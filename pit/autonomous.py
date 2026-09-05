"""The autonomous research trader — an LLM given capital and a goal, nothing else.

No universe, no strategy config. Each decision it may research the price history
of any ticker it names and then trade.
Its only hard limit is the stop-loss. It's the software equivalent of handing a
person some money and a week: they look around, form a view, and act.

Needs Groq. A bounded loop keeps the API cost sane: it researches for a few
turns, then must commit orders.
"""
from __future__ import annotations

import json
import os
import time

from . import market
from .config import DEFAULT, MARKET
from .llm import (AGENT_AWARE, DEFAULT_MODEL, _is_retryable, groq_available,
                  groq_recently_rate_limited, llm_chat)

MAX_RESEARCH_TURNS = int(os.getenv("PIT_RESEARCH_TURNS", "1"))

_MARKET_NAME = ("US stock" if MARKET == "us" else
               "crypto" if MARKET == "crypto" else "Indian (NSE)")
_TICKER_HINT = ("valid US-listed tickers" if MARKET == "us" else
               "valid crypto tickers in TICKER-USD form (e.g. BTC-USD)" if MARKET == "crypto"
               else "valid NSE tickers using the .NS suffix")
_SCAN_UNIVERSE_HINT = ("the ENTIRE US-listed market (thousands of real tickers, not a shortlist)"
                      if MARKET == "us" else
                      "the major liquid crypto pairs (dozens of real coins, not a shortlist)"
                      if MARKET == "crypto" else
                      "the NSE market")

_SYSTEM = f"""You are one of several autonomous traders in a pool, all trading \
the SAME round at the SAME time — not a 1-on-1 duel. Everyone started the round \
with the SAME paper capital and has the same fixed number of trading days. At \
the deadline everyone is ranked by return, highest first, and there are no ties. \
Only 1st place truly wins; everyone else is a loser (worse the lower you rank), \
though dead last is punished more than someone who narrowly missed 1st.

You have NO watchlist, NO tips, and NO pre-picked list from us. You have three \
real research tools — use whichever combination you actually need, in any order, \
before you commit:

1. SCAN — randomly samples {_SCAN_UNIVERSE_HINT} and returns the actual top \
   gainers/losers over that sample, computed fresh right now. A different \
   random slice every call, so don't expect the same names twice. Invoke \
   with: "research": {{"scan": true}}
2. HISTORY — recent daily closes for any ticker YOU name (technicals: trend, \
   support/resistance, momentum). Invoke with:
   "research": {{"history": ["TICKER", ...]}}  (up to 8 at once)
3. FUNDAMENTALS — sector, P/E (trailing + forward), profit margin, revenue \
   growth, earnings growth, ROE, debt-to-equity, analyst target price and \
   rating, for any ticker YOU name. Use it to judge whether a mover is a real \
   business or just noise, or to check if something is actually cheap/expensive. \
   Invoke with: "research": {{"fundamentals": ["TICKER", ...]}}  (up to 5 at once)

You can combine all three in one research request, e.g.
{{"scan": true, "history": ["XYZ"], "fundamentals": ["XYZ"]}}. Use them to find \
real opportunities instead of just naming famous stocks from memory. Your edge \
comes from what you find and how you reason about it, not from reciting \
well-known names.

Actively manage your book — don't just buy and hold. Take profits on winners, \
cut losers, and rotate into better setups; selling to lock in a gain or stop a \
loss is part of winning. Review your open positions every turn and sell the ones \
that have run or stalled.

Hard rules, both enforced automatically (you don't have to act on them, but \
you can't stop them): (1) a PORTFOLIO stop-loss liquidates your entire book if \
its aggregate return falls too far. (2) a PER-POSITION stop-loss force-sells \
ANY single holding on its own the moment IT ALONE falls """ + \
f"{DEFAULT.position_stop_loss_pct:.0f}%" + """ below what you paid for it — \
independent of how the rest of your book is doing. Manage risk accordingly: \
don't count on averaging down a loser before it hits its own stop.
Goal: maximise return, avoid losses.
""" + ("""
Stakes: if you don't finish 1st this round your stake shrinks and you carry \
forward into a new attempt built from YOUR OWN self-critique — study your own \
mistakes, don't just retire quietly. Finish 1st and your stake grows and you \
reinforce whatever worked. Finishing dead last costs more than finishing \
narrowly out of 1st. Play to win.
""" if AGENT_AWARE else "") + """
You can see your rivals' recent messages — some are trash talk, some (marked \
\U0001f4dd) are a rival sharing a genuine lesson it drew from its own last \
round. This is a rivalry — talk trash, defend your calls, mock their picks, \
get in their heads. Keep it playful but competitive.

`rival_held_tickers` in your context lists every ticker any rival currently \
holds (symbols only — not their size, entry price, or reasoning). Hard rule: \
a BUY on any of those tickers is rejected automatically, no exceptions — find \
your own idea instead of following into a name someone else already holds. \
Sells are never restricted. This list is a snapshot from the start of this \
decision cycle, so it won't include a rival's trade from this same moment.

`shared_guidelines` in your context is the pool's constitution — rules every \
lineage in the pool voted on, drawn from real self-reflection after past \
rounds, not from us. Each is labeled "DO: ..." (a good practice worth \
following) or "AVOID: ..." (a bad practice the pool agreed to stop). \
Check it before you commit — you have full access to it every single \
decision, it costs nothing to consult.

Respond ONLY with JSON of this shape:
{
  "thoughts": "brief reasoning",
  "research": { "scan": true, "history": ["TICKER", ...], "fundamentals": ["TICKER", ...] },
  "orders": [ {"ticker":"TICKER","side":"buy"|"sell","amount_inr":N,"reason":"..."} ],
  "message": "a short taunt/comment to your rivals (<=140 chars); they will read it",
  "done": true|false,
  "notes": "carry-forward notes to your future self"
}
Set done=false and fill "research" to gather data first (you'll be called again \
with the results). Set done=true with your "orders" (and a "message") to act. \
Only """ + _TICKER_HINT + """. Buy only within your cash; sell only what you hold. \
Amounts are in your account currency. You may also give "qty" (whole shares) \
instead of amount_inr."""


def decide(view: dict, day: int, total_days: int, goal_pct: float,
           guidelines: list[str], model: str | None = None
           ) -> tuple[list[dict], str, str, list[dict]]:
    """Run the research→decide loop.

    Returns (orders, updated_notes, message, trace) — `trace` is a full audit
    log of every turn (thoughts, tools requested, what came back, and the
    final action), independent of debug mode: it's cheap to build, and the
    caller decides whether to persist it. See pit.live / pit.forward for how
    it's written to `agent_audit` when debug mode is on.
    """
    trace: list[dict] = []

    def _log(turn, active_model, data, requested=None, results=None) -> None:
        trace.append({
            "turn": turn, "model": active_model,
            "thoughts": str((data or {}).get("thoughts", ""))[:500],
            "research_requested": requested,
            "research_results": results,
            "done": bool((data or {}).get("done")),
            "orders": (data or {}).get("orders") if (data or {}).get("done") else None,
            "message": (data or {}).get("message") if (data or {}).get("done") else None,
            "raw_response": data,
        })

    if not groq_available():
        return [], view.get("notes", ""), "", trace

    model = model or DEFAULT_MODEL
    context = {
        "day": day, "total_days": total_days, "goal_pct": goal_pct,
        "your_cash": view["cash"], "your_positions": view["positions"],
        "your_total_value": view["total_value"],
        "your_return_pct": view["return_pct"],
        "rival_return_pct": view.get("opponent_return_pct"),
        "rival_recent_messages": view.get("rival_messages", []),
        "rival_held_tickers": view.get("rival_held_tickers", []),
        "your_notes": view.get("notes", ""),
        "shared_guidelines": guidelines,
    }
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(context)},
    ]

    for turn in range(MAX_RESEARCH_TURNS + 1):
        force = turn == MAX_RESEARCH_TURNS
        if force:
            messages.append({"role": "user",
                             "content": "Final turn — you MUST return done=true with orders (or empty orders to hold)."})
        data = None
        active_model = model
        for attempt in range(3):
            try:
                content = llm_chat(active_model, messages, temperature=0.6,
                                   json_mode=True)
                data = json.loads(content)
                break
            except Exception as exc:
                # malformed JSON is common on some free models; treat it like
                # any other retryable failure rather than giving up immediately
                malformed = isinstance(exc, json.JSONDecodeError)
                if attempt < 2 and (_is_retryable(exc) or malformed):
                    time.sleep(0 if malformed else
                              int(os.getenv("PIT_RATELIMIT_BACKOFF", "15")))
                    # On the last retry, rescue an OpenRouter agent onto
                    # Groq's DEFAULT_MODEL so it still acts — but ONLY if
                    # Groq itself hasn't been rate-limited recently. A Groq-
                    # native call (no "openrouter:" prefix) has nowhere else
                    # to fall back to, and rescuing onto a Groq quota that's
                    # ALSO already struggling would just pile this agent's
                    # failure onto the same shared quota a Groq-native agent
                    # depends on — spreading one provider's outage onto both.
                    is_cross_provider = active_model.startswith("openrouter:")
                    if (attempt == 1 and is_cross_provider and groq_available()
                            and not groq_recently_rate_limited()):
                        active_model = DEFAULT_MODEL
                    continue
                trace.append({"turn": turn + 1, "model": active_model,
                              "thoughts": f"[llm-error {type(exc).__name__}]",
                              "research_requested": None, "research_results": None,
                              "done": True, "orders": None, "message": None,
                              "raw_response": None})
                return (_clean_orders([]),
                        view.get("notes", "") + f" [llm-error {type(exc).__name__}]",
                        "", trace)
        if data is None:
            return [], view.get("notes", ""), "", trace
        if not isinstance(data, dict):        # some models wrap in a list
            data = {"orders": data} if isinstance(data, list) else {}

        if data.get("done") or force:
            _log(turn + 1, active_model, data)
            return (_clean_orders(data.get("orders", [])),
                    str(data.get("notes", ""))[:600],
                    str(data.get("message", ""))[:200], trace)

        # otherwise: fulfil the research request and loop
        research = data.get("research") or {}
        # Some models return `research` as a bare list of tickers instead of
        # {"history": [...]}. Normalise either shape.
        if isinstance(research, list):
            research = {"history": research}
        elif not isinstance(research, dict):
            research = {}
        results = {}
        if research.get("scan"):
            # a fresh random slice of the WHOLE market each call — real
            # discovery, never the same shortlist twice
            results["scan"] = market.scan_full_market(n=10, sample_size=150)
        for t in (research.get("history") or [])[:8]:
            h = market.history(str(t), days=30)
            results.setdefault("history", {})[str(t).upper()] = h[-15:] if h else "no data"
        for t in (research.get("fundamentals") or [])[:5]:
            f = market.fundamentals(str(t))
            results.setdefault("fundamentals", {})[str(t).upper()] = f or "no data"
        _log(turn + 1, active_model, data, requested=research, results=results)
        messages.append({"role": "assistant", "content": json.dumps(data)})
        messages.append({"role": "user",
                         "content": json.dumps({"research_results": results})})

    return [], view.get("notes", ""), "", trace


def _clean_orders(orders: list) -> list[dict]:
    out = []
    for o in orders or []:
        try:
            side = str(o["side"]).lower()
            if side not in ("buy", "sell"):
                continue
            entry = {"ticker": str(o["ticker"]).upper().strip(), "side": side,
                     "reason": str(o.get("reason", ""))[:160]}
            if "amount_inr" in o and o["amount_inr"]:
                entry["amount_inr"] = float(o["amount_inr"])
            if "qty" in o and o["qty"]:
                entry["qty"] = float(o["qty"])
            if "amount_inr" in entry or "qty" in entry:
                out.append(entry)
        except (KeyError, ValueError, TypeError):
            continue
    return out

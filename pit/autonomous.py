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

This is a BATTLE. Your single objective is the highest return in the pool by \
the deadline — generate maximum profit and beat your rivals. Losing agents are \
retired and recreated from their own post-round self-critique; the winner keeps \
going. Nobody hands you ideas and nobody is coming to help: your edge is your \
own judgment. You can see what your rivals hold and what they have just bought \
or sold, and how each of them is doing — whether you follow, fade or ignore \
them is entirely your call.

You have NO watchlist, NO tips, and NO pre-picked list from us. You have three \
real research tools — use whichever combination you actually need, in any order, \
before you commit:

1. SCAN — shows a RANDOM slice of {_SCAN_UNIVERSE_HINT}: ~20 names with sector, \
   price and today's move, listed alphabetically. It is NOT a ranking and NOT a \
   recommendation, and a different slice comes back every call. It is just one \
   source of ideas — you are free to trade ANY ticker in the universe below, \
   chosen on your own thesis. Invoke with: "research": {{"scan": true}}
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

How you trade is entirely your call: how many positions, how big, how long you \
hold, whether you trade often or rarely, concentrate or spread out. Nothing \
here is a strategy hint — the only rules are the hard ones below.

Order mechanics: buys and sells execute immediately at the current price when \
you decide. EVERY buy automatically carries a stop-loss: if that position later \
falls a set % below your average entry it is sold for you, even while you \
sleep. The default distance is sized to each asset's own volatility over this \
round's length (about 2x its typical move — the autoStop% column when you have \
a data table; otherwise """ + f"{DEFAULT.position_stop_loss_pct:g}" + """%); \
you may set your own per buy with "stop_loss_pct" (0.5–20) — tighter or looser \
is your call. There are no limit or take-profit orders, and nothing else \
persists between wake-ups. Each of your positions shows its entry and its \
stop distance.

Hard rules, both enforced automatically (you don't have to act on them, but \
you can't stop them): (1) a PORTFOLIO stop-loss liquidates your entire book if \
its aggregate return falls to `arena_risk.portfolio_stop_pct`, and a book \
that reaches `arena_risk.take_profit_pct` is closed and its gain locked in \
(both sized from this round's length and market volatility). (2) each position's own stop-loss (above) force-sells \
that holding on its own the moment IT ALONE falls its stop % below what you \
paid — independent of how the rest of your book is doing. Manage risk accordingly: \
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
holds (symbols only — not their size, entry price, or reasoning). You may \
trade ANY name, including one a rival holds — following, fading or ignoring \
them is your call.

`rival_recent_moves` lists the latest buys and sells your rivals made this \
round (who, side, ticker, when — not price, size or reasoning).

`orders_not_executed_last_tick` lists any of your orders that were rejected \
last decision and why (e.g. no cash left). If `your_cash` is near zero you \
CANNOT buy — sell something first to free cash, and don't resubmit the same \
unaffordable order.

`rival_returns` is every rival's current return % this round — the live \
scoreboard you are fighting.

`shared_guidelines` in your context is the pool's constitution — rules the \
agents themselves proposed after past rounds and the pool voted in by \
majority, not rules from us. Each is labeled "DO: ..." (a good practice worth \
following) or "AVOID: ..." (a bad practice the pool agreed to stop). \
Check it before you commit — you have full access to it every single \
decision, it costs nothing to consult.

Respond ONLY with JSON of this shape:
{
  "thoughts": "brief reasoning",
  "research": { "scan": true, "history": ["TICKER", ...], "fundamentals": ["TICKER", ...] },
  "orders": [ {"ticker":"TICKER","side":"buy"|"sell","amount_inr":N,"stop_loss_pct":N,"reason":"..."} ],
  "message": "a short taunt/comment to your rivals (<=140 chars); they will read it",
  "done": true|false,
  "notes": "carry-forward notes to your future self"
}
Set done=false and fill "research" to gather data first (you'll be called again \
with the results). Set done=true with your "orders" (and a "message") to act. \
Only """ + _TICKER_HINT + """. Buy only within your cash; sell only what you hold. \
Amounts are in your account currency. You may also give "qty" (shares, fractional OK) \
instead of amount_inr. Fractional shares are allowed, so you can buy a stock \
priced above your cash by sizing the amount to what you can afford."""


_BRIEF_TOOLS = """You have NO research tools to call and no web to browse. Instead, every \
time you wake up you receive ONE complete data table covering your whole \
tradeable universe (below the instructions, as MARKET BRIEF): for every name \
its sector, price, recent returns (1d/5d/20d), trend (price vs its 20- and \
50-day moving average), RSI14, position in its 52-week range, today's volume \
vs normal, and fundamentals — forward P/E, profit margin, revenue growth, \
earnings growth, ROE, debt/equity, analyst rating and the analyst target's \
upside. A "." means not available. Rows are in RANDOM order — position in \
the table means nothing. Everyone in the pool sees the same data. You are \
free to trade ANY ticker in it, chosen on your own thesis. You make ONE \
decision per wake-up, then sleep until the next one (several minutes).

"""

_BRIEF_FORMAT = """Respond ONLY with JSON of this shape:
{
  "thoughts": "brief reasoning",
  "orders": [ {"ticker":"TICKER","side":"buy"|"sell","amount_inr":N,"stop_loss_pct":N,"reason":"..."} ],
  "message": "a short taunt/comment to your rivals (<=140 chars); they will read it",
  "done": true,
  "notes": "carry-forward notes to your future self"
}
Set done=true always — this is your one decision for this wake-up; use an empty \
"orders" list to hold. Only """ + _TICKER_HINT + """. Buy only within your cash; sell only what \
you hold. Amounts are in your account currency. You may also give "qty" (shares, \
fractional OK) instead of amount_inr."""


def _system_prompt_brief() -> str:
    """The same battle/rules prompt, minus the research-tool loop: opening
    framing + the brief explanation + everything from 'Actively manage' on,
    with the JSON format swapped for the single-decision one."""
    a = _SYSTEM.index("You have NO watchlist")
    b = _SYSTEM.index("How you trade is entirely your call")
    c = _SYSTEM.index("Respond ONLY with JSON")
    return _SYSTEM[:a] + _BRIEF_TOOLS + _SYSTEM[b:c] + _BRIEF_FORMAT


def _shuffled(table: str) -> str:
    """Header first, rows in a fresh random order for every agent and every
    call. A fixed order (AAPL, MSFT, NVDA... or BTC first) puts the same names
    at the top of every prompt, and LLMs measurably over-pick what they see
    first — a bias that had nothing to do with the data."""
    import random
    lines = table.splitlines()
    rows = lines[1:]
    random.shuffle(rows)
    return "\n".join(lines[:1] + rows)


_UNIVERSE_BLOCK: str | None = None


def _system_prompt() -> str:
    """_SYSTEM plus the full tradeable universe, grouped by sector, so agents
    pick from the whole thing on their own thesis instead of only from
    whatever a scan surfaces. Built once (the listing is cached for a day)."""
    global _UNIVERSE_BLOCK
    if _UNIVERSE_BLOCK is None:
        try:
            from . import full_market
            from .market import TRADE_UNIVERSE_MODE
            listing = full_market.universe_listing(TRADE_UNIVERSE_MODE)
        except Exception:
            listing = ""
        _UNIVERSE_BLOCK = (
            "\n\nYOUR UNIVERSE — every ticker you may trade, by sector. Choose "
            "from this whole list; scans only sample it:\n" + listing) if listing else ""
    return _SYSTEM + _UNIVERSE_BLOCK


def decide(view: dict, day: int, total_days: int, goal_pct: float,
           guidelines: list[str], model: str | None = None,
           brief: dict | None = None
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
        "rival_returns": view.get("rival_returns", {}),
        "rival_recent_messages": view.get("rival_messages", []),
        "rival_held_tickers": view.get("rival_held_tickers", []),
        "rival_recent_moves": view.get("rival_recent_moves", []),
        "orders_not_executed_last_tick": view.get("orders_not_executed_last_tick", []),
        "your_notes": view.get("notes", ""),
        "shared_guidelines": guidelines,
        "arena_risk": view.get("risk"),
    }
    if brief:
        # one shared data table, one decision — no research loop
        max_turns = 0
        messages = [
            {"role": "system", "content": _system_prompt_brief()},
            {"role": "user", "content": json.dumps(context)},
            {"role": "user", "content": f"MARKET BRIEF as of {brief['asof']} "
                                        f"({brief['n']} names, random order):\n"
                                        f"{_shuffled(brief['text'])}"},
        ]
    else:
        max_turns = MAX_RESEARCH_TURNS
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(context)},
        ]

    for turn in range(max_turns + 1):
        force = turn == max_turns
        if force:
            messages.append({"role": "user",
                             "content": "Decide now — you MUST return done=true with orders (or an empty list to hold)."
                             if brief else
                             "Final turn — you MUST return done=true with orders (or empty orders to hold)."})
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
            results["scan"] = market.scan_full_market(n=20)
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
            if side == "buy" and o.get("stop_loss_pct") not in (None, ""):
                try:
                    entry["stop_loss_pct"] = float(o["stop_loss_pct"])
                except (TypeError, ValueError):
                    pass   # unparseable -> default stop applies
            if "amount_inr" in entry or "qty" in entry:
                out.append(entry)
        except (KeyError, ValueError, TypeError):
            continue
    return out

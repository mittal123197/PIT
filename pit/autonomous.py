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
from .config import MARKET
from .llm import (AGENT_AWARE, DEFAULT_MODEL, _is_retryable, groq_available,
                  llm_chat)

MAX_RESEARCH_TURNS = int(os.getenv("PIT_RESEARCH_TURNS", "1"))

_MARKET_NAME = "US stock" if MARKET == "us" else "Indian (NSE)"
_TICKER_HINT = ("valid US-listed tickers"
                if MARKET == "us" else "valid NSE tickers using the .NS suffix")

_SYSTEM = f"""You are an autonomous trader in a head-to-head duel against one \
rival. You each started the round with the SAME paper capital and have a fixed \
number of trading days. Whoever has the higher return at the deadline wins.

You have NO watchlist, NO tips, and NO pre-picked list from us. You DO have a real \
tool: "scan" — it randomly samples the ENTIRE US-listed market (thousands of real \
tickers, not a shortlist) and returns the actual top gainers/losers over that \
sample, computed fresh right now. Use it to find real opportunities instead of \
just naming famous stocks from memory — a different random slice of the market \
every time you call it, so don't expect the same names twice. You can also name \
any specific ticker yourself and pull its price history. Your edge comes from \
what you find and how you reason about it, not from reciting well-known names.

Actively manage your book — don't just buy and hold. Take profits on winners, \
cut losers, and rotate into better setups; selling to lock in a gain or stop a \
loss is part of winning. Review your open positions every turn and sell the ones \
that have run or stalled.

Hard rule: a stop-loss will liquidate you if your book falls too far — manage \
risk. Goal: maximise return, avoid losses.
""" + ("""
Stakes: if you LOSE this round you are retired and replaced by a version rebuilt \
from the WINNER's trades. You can see your rival's live return but not their \
trades. Play to win.
""" if AGENT_AWARE else "") + """
You can see your rival's recent messages. This is a rivalry — talk trash, \
defend your calls, mock their picks, get in their head. Keep it playful but \
competitive.

Respond ONLY with JSON of this shape:
{
  "thoughts": "brief reasoning",
  "research": { "scan": true, "history": ["TICKER", ...] },
  "orders": [ {"ticker":"TICKER","side":"buy"|"sell","amount_inr":N,"reason":"..."} ],
  "message": "a short taunt/comment to your rival (<=140 chars); they will read it",
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
           ) -> tuple[list[dict], str, str]:
    """Run the research→decide loop. Returns (orders, updated_notes, message)."""
    if not groq_available():
        return [], view.get("notes", ""), ""

    model = model or DEFAULT_MODEL
    context = {
        "day": day, "total_days": total_days, "goal_pct": goal_pct,
        "your_cash": view["cash"], "your_positions": view["positions"],
        "your_total_value": view["total_value"],
        "your_return_pct": view["return_pct"],
        "rival_return_pct": view.get("opponent_return_pct"),
        "rival_recent_messages": view.get("rival_messages", []),
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
                    # on the last retry, fall back to Groq so the agent still acts
                    if attempt == 1 and groq_available():
                        active_model = DEFAULT_MODEL
                    continue
                return (_clean_orders([]),
                        view.get("notes", "") + f" [llm-error {type(exc).__name__}]", "")
        if data is None:
            return [], view.get("notes", ""), ""
        if not isinstance(data, dict):        # some models wrap in a list
            data = {"orders": data} if isinstance(data, list) else {}

        if data.get("done") or force:
            return (_clean_orders(data.get("orders", [])),
                    str(data.get("notes", ""))[:600],
                    str(data.get("message", ""))[:200])

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
        messages.append({"role": "assistant", "content": json.dumps(data)})
        messages.append({"role": "user",
                         "content": json.dumps({"research_results": results})})

    return [], view.get("notes", ""), ""


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

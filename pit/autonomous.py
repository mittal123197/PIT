"""The autonomous research trader — an LLM given capital and a goal, nothing else.

No universe, no strategy config. Each decision it may *research* (pull the
market movers, or the price history of any ticker it names) and then *trade*.
Its only hard limit is the stop-loss. It's the software equivalent of handing a
person some money and a week: they look around, form a view, and act.

Needs Groq. A bounded loop keeps the API cost sane: it researches for a few
turns, then must commit orders.
"""
from __future__ import annotations

import json
import os

from . import market
from .config import MARKET
from .llm import AGENT_AWARE, DEFAULT_MODEL, _client, _extra_for, groq_available

MAX_RESEARCH_TURNS = int(os.getenv("PIT_RESEARCH_TURNS", "2"))

_MARKET_NAME = "US stock" if MARKET == "us" else "Indian (NSE)"
_TICKER_HINT = ("US-listed tickers (e.g. AAPL, TSLA, NVDA, COIN)"
                if MARKET == "us" else "NSE tickers (suffix .NS)")

_SYSTEM = f"""You are an autonomous trader in a head-to-head duel against one \
rival. You each started the round with the SAME paper capital and have a fixed \
number of trading days. Whoever has the higher return at the deadline wins.

You have NO preset list of stocks and NO fixed strategy — you decide everything, \
like a human trader starting from scratch. Find opportunities yourself: you know \
the {_MARKET_NAME} market; you can also look at today's biggest movers. Research \
any ticker before trading it.

Hard rule: a stop-loss will liquidate you if your book falls too far — manage \
risk. Goal: maximise return, avoid losses.
""" + ("""
Stakes: if you LOSE this round you are retired and replaced by a version rebuilt \
from the WINNER's trades. You can see your rival's live return but not their \
trades. Play to win.
""" if AGENT_AWARE else "") + """
Respond ONLY with JSON of this shape:
{
  "thoughts": "brief reasoning",
  "research": { "movers": true, "history": ["TICKER.NS", ...] },
  "orders": [ {"ticker":"TICKER.NS","side":"buy"|"sell","amount_inr":N,"reason":"..."} ],
  "done": true|false,
  "notes": "carry-forward notes to your future self"
}
Set done=false and fill "research" to gather data first (you'll be called again \
with the results). Set done=true with your "orders" to act. Only """ + _TICKER_HINT + """. \
Buy only within your cash; sell only what you hold. Amounts are in your account \
currency. You may also give "qty" (whole shares) instead of amount_inr."""


def decide(view: dict, day: int, total_days: int, goal_pct: float,
           guidelines: list[str], model: str | None = None) -> tuple[list[dict], str]:
    """Run the research→decide loop. Returns (orders, updated_notes)."""
    if not groq_available():
        return [], view.get("notes", "")

    model = model or DEFAULT_MODEL
    context = {
        "day": day, "total_days": total_days, "goal_pct": goal_pct,
        "your_cash": view["cash"], "your_positions": view["positions"],
        "your_total_value": view["total_value"],
        "your_return_pct": view["return_pct"],
        "rival_return_pct": view.get("opponent_return_pct"),
        "your_notes": view.get("notes", ""),
        "shared_guidelines": guidelines,
        "market_movers": market.movers(n=10),   # always give a discovery surface
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
        try:
            resp = _client().chat.completions.create(
                model=model, messages=messages, temperature=0.6,
                response_format={"type": "json_object"}, **_extra_for(model),
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as exc:
            return [], view.get("notes", "") + f" [llm-error {type(exc).__name__}]"

        if data.get("done") or force:
            return _clean_orders(data.get("orders", [])), str(data.get("notes", ""))[:600]

        # otherwise: fulfil the research request and loop
        research = data.get("research") or {}
        results = {}
        if research.get("movers"):
            results["movers"] = context["market_movers"]
        for t in (research.get("history") or [])[:8]:
            h = market.history(str(t), days=30)
            results.setdefault("history", {})[str(t).upper()] = h[-15:] if h else "no data"
        messages.append({"role": "assistant", "content": json.dumps(data)})
        messages.append({"role": "user",
                         "content": json.dumps({"research_results": results})})

    return [], view.get("notes", "")


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

"""One agent decision: an LLM, its book, its charter, the shared data table.

Every wake-up the agent receives ONE market brief — the same table for every
agent (intraday + daily technicals + fundamentals for the whole universe, rows
in random order) — plus its own book, records and rivals' scores, and returns
one decision: orders (or none), a taunt, and notes to its future self. There
are no research tools or multi-turn loops: live trading and replays feed the
same table (pit/brief.py live, pit/replay.py from the market history).
"""
from __future__ import annotations

import json
import os
import random
import time

from .config import MARKET
from .llm import AGENT_AWARE, DEFAULT_MODEL, _is_retryable, llm_available, llm_chat

_TICKER_HINT = ("tickers from the table (US-listed stocks)" if MARKET == "us" else
                "tickers from the table, in TICKER-USD form (e.g. BTC-USD)")

SYSTEM = """You are one of several autonomous traders in a pool, all trading \
the SAME round at the SAME time with the SAME paper capital and the same \
deadline. At the deadline everyone is ranked by return, highest first — no \
ties. Only 1st place truly wins.

This is a BATTLE. Your single objective is the highest return in the pool by \
the deadline. Nobody hands you ideas: your edge is your own judgment. You can \
see how each rival is doing (their live return) but not what they hold or \
trade — your picks have to be your own.
""" + ("""
Stakes: if you don't finish 1st, this version of you is retired and your next \
generation is rebuilt from YOUR OWN post-round self-critique. Finish 1st and \
you keep going and reinforce what worked. Play to win.
""" if AGENT_AWARE else "") + """
YOUR DATA: every wake-up you receive ONE complete data table for your whole \
tradeable universe (as MARKET BRIEF, after these instructions): per name its \
sector and price; TODAY's session so far — open% (change since today's open), \
vsVWAP% (price vs today's volume-weighted average price), 15m% / 30m% (change \
over the last 15 / 30 minutes), dayRange% (0 = at today's low, 100 = at the \
high), volPace (today's volume vs a normal day at this hour); the daily view — \
1d/5d/20d%, price vs its 20- and 50-day moving average, RSI14, position in its \
52-week range, vol-x, dayVol% (typical daily move), autoStop% (your default \
stop for that name this round); and fundamentals — forward P/E, margin, \
revenue and earnings growth, ROE, debt/equity, analyst rating and target \
upside. A MARKET NOW line gives SPY/QQQ since the open and market breadth. \
"." means not available (some intraday columns right after the open). Rows \
are in RANDOM order. Everyone sees the same table. You make ONE decision per \
wake-up, then sleep until the next (several minutes).

How you trade is your call: how many positions, how big, how long you hold, \
how often you trade. Nothing here is a strategy hint — the only rules are the \
hard ones below and your own charter.

Order mechanics: orders execute immediately at the current price, and every \
fill pays half the bid/ask spread (buys a little above the quote, sells a \
little below), as with a real broker. EVERY buy carries a stop-loss: if that \
position later falls its stop % below your average entry, the arena sells it \
for you, even between wake-ups. The default stop is that name's autoStop% \
(about 2x its typical move over this round); set your own per buy with \
"stop_loss_pct" (0.5–20). No limit or take-profit orders. Each position shows \
its entry, stop, pos_pnl (P&L % vs entry) and held_min (minutes held).

Hard rules the arena enforces: (1) a PORTFOLIO stop liquidates your whole book \
if its return falls to `arena_risk.portfolio_stop_pct`, and a book reaching \
`arena_risk.take_profit_pct` is closed with the gain locked in (both sized \
from this round's length and market volatility); (2) each position's own stop \
(above) sells that holding alone. Don't count on averaging down a loser.

`your_charter` is the trading style YOU chose for yourself — trade by it \
(`your_charter_backtest` is how its checkable version did on ~60 days of real \
market data). When no name meets your charter, holding cash IS following it — \
a legitimate decision, not a failure. Every buy must carry "charter_fit": the \
charter conditions it meets, citing the table's numbers — or "deviation: <why>" \
if you knowingly break it (and `charter_enforced`, when present, means the \
arena rejects buys that don't meet your charter's entry conditions). \
`your_charter_record_this_round` shows your on- vs off-charter buys and what \
each made. You can change the charter after the round, not mid-round.

Your own records: `your_sell_record_this_round` (sells you chose, how many \
were profitable, what holding instead would be worth, costs paid), \
`arena_sold_for_you_this_round` (stops that fired), \
`orders_not_executed_last_tick` (rejected orders and why — if `your_cash` is \
near zero you CANNOT buy; sell first).

Rivals: `rival_returns` is the live scoreboard. Rivals' messages may be trash \
talk or a lesson (marked \U0001f4dd) from their last round; stock names in them \
are hidden ("[a stock]"). This is a rivalry — talk trash, defend your calls. \
Keep it playful.

`shared_guidelines` is the pool's rulebook: rules the agents proposed, the \
arena researched on market history, and the pool debated and voted in. "DO: \
..." is a practice worth following, "AVOID: ..." one to stop; each shows its \
evidence.

Respond ONLY with JSON of this shape:
{
  "thoughts": "brief reasoning",
  "orders": [ {"ticker":"TICKER","side":"buy"|"sell","amount":N,"stop_loss_pct":N,"charter_fit":"...","reason":"..."} ],
  "message": "a short taunt/comment to your rivals (<=140 chars); they will read it",
  "notes": "carry-forward notes to your future self"
}
Use an empty "orders" list to hold. Only """ + _TICKER_HINT + """. Buy only \
within your cash; sell only what you hold. Amounts are in dollars; you may give \
"qty" (fractional OK) instead of amount."""


def _shuffled(table: str) -> str:
    """Header first, rows in a fresh random order for every agent and every
    call — LLMs over-pick what they see first, a bias unrelated to the data."""
    lines = table.splitlines()
    rows = lines[1:]
    random.shuffle(rows)
    return "\n".join(lines[:1] + rows)


def decide(view: dict, decision_number: int, guidelines: list[str],
           brief: dict | None, model: str | None = None
           ) -> tuple[list[dict], str, str, list[dict]]:
    """One decision. Returns (orders, notes, message, trace) — `trace` is the
    audit record (thoughts, final action), persisted by the caller in debug
    mode. Without a brief or a reachable model the agent holds."""
    trace: list[dict] = []
    notes = view.get("notes", "")
    model = model or DEFAULT_MODEL
    if not brief or not llm_available(model):
        return [], notes, "", trace
    from .charter import enforce_on
    context = {
        "decision_number": decision_number,
        "your_cash": view["cash"], "your_positions": view["positions"],
        "your_total_value": view["total_value"],
        "your_return_pct": view["return_pct"],
        "rival_returns": view.get("rival_returns", {}),
        "rival_recent_messages": view.get("rival_messages", []),
        "orders_not_executed_last_tick": view.get("orders_not_executed_last_tick", []),
        "arena_sold_for_you_this_round": view.get("arena_sold_for_you_this_round", []),
        "your_sell_record_this_round": view.get("your_sell_record_this_round"),
        "your_charter_record_this_round": view.get("your_charter_record_this_round"),
        "your_charter": view.get("charter", ""),
        "your_charter_backtest": view.get("charter_backtest", ""),
        **({"charter_enforced": ("ON: any buy that doesn't meet your charter's entry "
                                 "conditions in the current table is rejected by the arena; "
                                 "sells are never restricted")} if enforce_on() else {}),
        "your_notes": notes,
        "shared_guidelines": guidelines,
        "arena_risk": view.get("risk"),
    }
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(context)},
        {"role": "user", "content": f"MARKET BRIEF as of {brief['asof']} "
                                    f"({brief['n']} names, random order):\n"
                                    + (f"{brief['market']}\n" if brief.get("market") else "")
                                    + _shuffled(brief["text"])},
    ]
    data = None
    for attempt in range(3):
        try:
            data = json.loads(llm_chat(model, messages, temperature=0.6, json_mode=True))
            break
        except Exception as exc:
            malformed = isinstance(exc, json.JSONDecodeError)
            if attempt < 2 and (_is_retryable(exc) or malformed):
                time.sleep(0 if malformed else int(os.getenv("PIT_RATELIMIT_BACKOFF", "15")))
                continue
            trace.append({"turn": 1, "model": model, "done": True, "orders": None,
                          "message": None, "raw_response": None,
                          "research_requested": None, "research_results": None,
                          "thoughts": f"[llm-error {type(exc).__name__}]"})
            return [], notes, "", trace
    if not isinstance(data, dict):        # some models wrap the reply in a list
        data = {"orders": data} if isinstance(data, list) else {}
    trace.append({"turn": 1, "model": model, "done": True,
                  "thoughts": str(data.get("thoughts", ""))[:2000],
                  "orders": data.get("orders"), "message": data.get("message"),
                  "research_requested": None, "research_results": None,
                  "raw_response": data})
    return (_clean_orders(data.get("orders", [])), str(data.get("notes", ""))[:600],
            str(data.get("message", ""))[:200], trace)


def _clean_orders(orders: list) -> list[dict]:
    """Validate the model's orders; mark buys that are off-charter (declared
    deviation) or that state no charter fit at all."""
    out = []
    for o in orders or []:
        try:
            side = str(o["side"]).lower()
            if side not in ("buy", "sell"):
                continue
            reason = str(o.get("reason", ""))[:160]
            fit = str(o.get("charter_fit", "")).strip()
            if side == "buy" and fit.lower().startswith("deviation"):
                reason = ("⚠ off-charter: " + fit[len("deviation"):].lstrip(" :—-")[:80]
                          + " | " + reason)[:200]
            elif side == "buy" and not fit:
                reason = ("⚠ no charter fit given | " + reason)[:200]
            entry = {"ticker": str(o["ticker"]).upper().strip(), "side": side,
                     "reason": reason}
            amt = o.get("amount") or o.get("amount_inr")   # old name still accepted
            if amt:
                entry["amount"] = float(amt)
            if "qty" in o and o["qty"]:
                entry["qty"] = float(o["qty"])
            if side == "buy" and o.get("stop_loss_pct") not in (None, ""):
                try:
                    entry["stop_loss_pct"] = float(o["stop_loss_pct"])
                except (TypeError, ValueError):
                    pass   # unparseable -> default stop applies
            if "amount" in entry or "qty" in entry:
                out.append(entry)
        except (KeyError, ValueError, TypeError):
            continue
    return out

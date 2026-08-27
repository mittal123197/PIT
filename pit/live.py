"""Live session — the agents trade and trash-talk in real time for a duration.

Unlike the daily forward step, this loops every `interval` seconds for `minutes`
total: each tick both agents research the real market, place paper trades, and
fire a message at their rival (which the rival sees next tick). Everything is
written to the DB as it happens, so the live web view (`/live`, auto-refresh)
updates while you watch. Paper money only — no real orders.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from . import autonomous, forward, market
from . import guidelines as gmod
from .config import CURRENCY, DEFAULT, MARKET, ArenaConfig


def _price_fn():
    cache: dict[str, float | None] = {}

    def price(t):
        if t not in cache:
            q = market.quote(t)
            cache[t] = q["price"] if q else None
        return cache[t]
    return price


def run_live(conn, config: ArenaConfig = DEFAULT, minutes: int = 120,
             interval: int = 180, verbose: bool = True) -> dict:
    rnd = forward._active_round(conn)
    if not rnd:
        forward.start_round(conn, config, days=10_000)  # time-based; no day-resolve
        rnd = forward._active_round(conn)
    if verbose:
        print(f"● LIVE — {MARKET.upper()} market, {minutes} min, a tick every "
              f"{interval}s. Two autonomous agents, {CURRENCY}"
              f"{config.base_capital:,.0f} paper each. Watch /live.\n", flush=True)

    end = time.time() + minutes * 60
    tick = 0
    while time.time() < end:
        tick += 1
        try:
            _tick(conn, config, rnd, tick, verbose)
        except Exception as exc:
            if verbose:
                print(f"  [tick error: {exc!r}]", flush=True)
        remaining = end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    conn.execute("UPDATE rounds SET status='ended', deadline=? WHERE id=?",
                 (forward._now(), rnd["id"]))
    conn.commit()
    return _standings(conn, rnd, verbose)


def _tick(conn, config, rnd, tick, verbose):
    price = _price_fn()
    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    guidelines = gmod.active_texts(conn)
    ts = datetime.now().strftime("%H:%M:%S")

    def value(st):
        h = json.loads(st["holdings"])
        return st["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())

    returns = {aid: (value(st) / st["starting_capital"] - 1) * 100
               for aid, st in states.items()}
    if verbose:
        print(f"[tick {tick} · {ts}]", flush=True)

    for aid, st in states.items():
        if st["status"] != "active":
            continue
        name = forward._name(conn, aid)
        if returns[aid] <= -config.stop_loss_pct:
            forward._liquidate(conn, rnd["id"], aid, st, price, ts)
            conn.commit()
            if verbose:
                print(f"  {name} STOPPED OUT at {returns[aid]:.1f}%", flush=True)
            continue

        agent = conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
        cfg = json.loads(agent["strategy_config"])
        holdings = json.loads(st["holdings"])
        positions = [{"ticker": t, "qty": q, "price": price(t),
                      "value": round(q * (price(t) or 0), 2)}
                     for t, q in holdings.items()]
        view = {
            "cash": round(st["current_capital"], 2), "positions": positions,
            "total_value": round(value(st), 2), "return_pct": round(returns[aid], 2),
            "opponent_return_pct": round(
                max((r for a, r in returns.items() if a != aid), default=0.0), 2),
            "notes": cfg.get("notes", ""),
            "rival_messages": forward._recent_messages(conn, rnd["id"], aid),
        }
        orders, notes, message = autonomous.decide(
            view, tick, 0, rnd["goal_pct"], guidelines)
        n = forward._execute(conn, rnd["id"], aid, st, orders, price, ts)
        cfg["notes"] = notes
        conn.execute("UPDATE agents SET strategy_config=? WHERE id=?",
                     (json.dumps(cfg), aid))
        conn.execute("UPDATE round_states SET current_capital=?, holdings=?, "
                     "final_return_pct=?, trade_count=trade_count WHERE round_id=? "
                     "AND agent_id=?",
                     (st["current_capital"], st["holdings"], round(returns[aid], 3),
                      rnd["id"], aid))
        if message:
            conn.execute("INSERT INTO agent_messages (round_id, agent_id, ts, "
                         "message) VALUES (?,?,?,?)",
                         (rnd["id"], aid, forward._now(), message))
        conn.commit()
        if verbose:
            arrow = "▲" if returns[aid] >= 0 else "▼"
            print(f"  {name:<6} {CURRENCY}{value(st):>10,.0f} ({returns[aid]:+.2f}%) {arrow}"
                  f"  {n} trade(s)", flush=True)
            for o in orders:
                print(f"         {o['side']:<4} {o['ticker']:<6} — {o.get('reason','')[:52]}",
                      flush=True)
            if message:
                print(f"         💬 \"{message}\"", flush=True)


def _standings(conn, rnd, verbose) -> dict:
    price = _price_fn()
    out = []
    for r in conn.execute("SELECT * FROM round_states WHERE round_id=?",
                          (rnd["id"],)).fetchall():
        h = json.loads(r["holdings"])
        total = r["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())
        out.append({"name": forward._name(conn, r["agent_id"]),
                    "value": round(total, 2),
                    "return_pct": round((total / r["starting_capital"] - 1) * 100, 2)})
    out.sort(key=lambda x: x["return_pct"], reverse=True)
    if verbose and out:
        print(f"\n● SESSION OVER — {out[0]['name']} leads at "
              f"{out[0]['return_pct']:+.2f}%", flush=True)
    return {"round_id": rnd["id"], "standings": out}

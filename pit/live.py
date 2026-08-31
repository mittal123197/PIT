"""Live session — the agents trade and trash-talk in real time for a duration.

Unlike the daily forward step, this loops every `interval` seconds for `minutes`
total: each tick both agents research the real market, place paper trades, and
fire a message at their rival (which the rival sees next tick). Everything is
written to the DB as it happens, so the live web view (`/live`, auto-refresh)
updates while you watch. Paper money only — no real orders.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

from . import autonomous, forward, market
from . import guidelines as gmod
from .config import CURRENCY, DEFAULT, MARKET, ArenaConfig


def _persist_audit(conn, round_id, agent_id, trace: list[dict]) -> None:
    """Write every turn of a decision (thoughts, tools used, results, final
    action) to `agent_audit`, for the debug-mode audit log on /live.

    Uses local wall-clock time (matching trades/fills, which use
    datetime.now() via _tick's `ts`) — not forward._now()'s UTC ISO. Mixing
    the two used to show audit/message timestamps hours off from the fills
    they belonged to (a full UTC-offset gap, e.g. IST is +5:30)."""
    ts = datetime.now().strftime("%H:%M:%S")
    for step in trace:
        conn.execute(
            """INSERT INTO agent_audit
               (round_id, agent_id, ts, turn, model, thoughts,
                research_requested, research_results, done, orders, message,
                raw_response)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (round_id, agent_id, ts, step["turn"], step.get("model"),
             step.get("thoughts"),
             json.dumps(step.get("research_requested")) if step.get("research_requested") is not None else None,
             json.dumps(step.get("research_results")) if step.get("research_results") is not None else None,
             int(step.get("done") or False),
             json.dumps(step.get("orders")) if step.get("orders") is not None else None,
             step.get("message"),
             json.dumps(step.get("raw_response")) if step.get("raw_response") is not None else None))
    conn.commit()


def _price_fn():
    cache: dict[str, float | None] = {}

    def price(t):
        if t not in cache:
            q = market.quote(t)
            cache[t] = q["price"] if q else None
        return cache[t]
    return price


def run_live(conn, config: ArenaConfig = DEFAULT, minutes: int = 180,
             interval: int = 900, refresh: int | None = None,
             verbose: bool = True, debug: bool = False) -> dict:
    """LLM *decisions* every `interval`s; cheap price/P&L *marks* every
    `refresh`s so the scoreboard moves continuously between decisions.

    debug=True persists every research turn (thoughts, tools used, results,
    final action) to `agent_audit` — see pit.autonomous.decide's `trace`."""
    refresh = refresh or int(os.getenv("PIT_MARK_REFRESH", "120"))
    if debug and verbose:
        print("[debug] full decision audit trail enabled — every thought, "
              "tool call, and action will be recorded.", flush=True)
    rnd = forward._active_round(conn)
    if not rnd:
        forward.start_round(conn, config, days=0)  # 0 = time-based; no day-resolve
        rnd = forward._active_round(conn)
    if verbose:
        print(f"● LIVE — {MARKET.upper()} market, {minutes} min. Decisions every "
              f"{interval // 60} min, prices marked every {refresh}s. "
              f"{CURRENCY}{config.base_capital:,.0f} paper each. Watch /live.\n",
              flush=True)

    end = time.time() + minutes * 60
    decision = 0
    decision_due = 0.0  # force a decision immediately
    while time.time() < end:
        now = time.time()
        try:
            if now >= decision_due:
                decision += 1
                _tick(conn, config, rnd, decision, verbose, debug)
                decision_due = time.time() + interval
            else:
                _refresh_marks(conn, config, rnd, verbose)
        except Exception as exc:
            if verbose:
                print(f"  [loop error: {exc!r}]", flush=True)
        remaining = end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(refresh, remaining))

    # Resolve for real: winner/loser cascade, ELO, stake, and — the whole
    # point of the arena — mutate the loser into its next generation from the
    # winner's trade log. Without this, a live session just stopped ('ended')
    # and never touched `lineages`/`round_results`, so wins/losses/generation
    # on the leaderboard never moved and no recreation ever happened.
    rnd_fresh = dict(conn.execute("SELECT * FROM rounds WHERE id=?",
                                  (rnd["id"],)).fetchone())
    if rnd_fresh["status"] == "live":
        price = _price_fn()
        outcome = forward._resolve(conn, rnd_fresh, config, price)
        if verbose:
            ranking = outcome.get("ranking") or []
            if outcome.get("both_stopped_out"):
                print(f"\n● DOUBLE STOP-OUT — everyone got liquidated. "
                      f"{outcome['winner']} {outcome['winner_return']:+.2f}% was "
                      f"merely less bad than the rest ({outcome['reason']})")
            elif outcome.get("passive_win"):
                print(f"\n● ROUND RESOLVED — {outcome['winner']} wins {outcome['winner_return']:+.2f}% "
                      f"on ZERO TRADES (capped stake bonus) ({outcome['reason']})")
            elif len(ranking) > 2:
                print(f"\n● ROUND RESOLVED — {len(ranking)}-way ({outcome['reason']})")
            else:
                print(f"\n● ROUND RESOLVED — WINNER {outcome['winner']} "
                      f"{outcome['winner_return']:+.2f}% beat {outcome['loser']} "
                      f"{outcome['loser_return']:+.2f}% ({outcome['reason']})")
            if len(ranking) > 2:
                # full N-way placement — the 2-line winner/loser summary above
                # would silently drop every middle finisher
                for r in ranking:
                    tag = ("reinforces" if r["rank"] == 1 and not outcome.get("both_stopped_out")
                          else "self-critiques")
                    print(f"  #{r['rank']} {r['name']:<8} {r['return_pct']:+.2f}%  "
                          f"{tag}: {r['note']}")
            else:
                verb = "self-critiques" if outcome.get("both_stopped_out") else "reinforces"
                print(f"  {outcome['winner']} {verb}: {outcome.get('winner_note', '')}")
                print(f"  {outcome['loser']} self-critiques: {outcome['mutation_note']}")
            if outcome.get("reflection"):
                r = outcome["reflection"]
                verdict = "ADOPTED" if r["accepted"] else "rejected"
                practice = r.get("practice", "good").upper()
                print(f"  reflection → proposal to {r['kind']} a {practice} PRACTICE "
                      f"guideline [{verdict}, {r['agree']}/{r['total']} agreed]: "
                      f"\"{r['text']}\"", flush=True)
    else:
        outcome = None
    standings = _standings(conn, rnd, verbose)
    standings["outcome"] = outcome
    return standings


def _refresh_marks(conn, config, rnd, verbose):
    """Cheap mark-to-market between decisions: revalue holdings at fresh prices,
    persist each agent's return, and enforce the stop-loss. No LLM calls."""
    price = _price_fn()
    line = []
    for r in conn.execute("SELECT * FROM round_states WHERE round_id=?",
                          (rnd["id"],)).fetchall():
        st = dict(r)
        if st["status"] == "active":
            forward.check_position_stops(conn, rnd["id"], st["agent_id"], st,
                                         price, datetime.now().strftime("%H:%M:%S"),
                                         config)
        h = json.loads(st["holdings"])
        total = st["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())
        ret = (total / st["starting_capital"] - 1) * 100
        if st["status"] == "active":
            stop = config.stop_loss_pct_for(rnd["length_days"])
            goal = rnd["goal_pct"]
            hit_reason = "stop-loss" if ret <= -stop else "goal-hit" if ret >= goal else None
            if hit_reason:
                forward._liquidate(conn, rnd["id"], st["agent_id"], st, price,
                                   datetime.now().strftime("%H:%M:%S"), reason=hit_reason)
                total = st["current_capital"]
                ret = (total / st["starting_capital"] - 1) * 100
        conn.execute("UPDATE round_states SET final_return_pct=? "
                     "WHERE round_id=? AND agent_id=?",
                     (round(ret, 3), rnd["id"], st["agent_id"]))
        line.append(f"{forward._name(conn, st['agent_id'])} {ret:+.2f}%")
    conn.commit()
    if verbose:
        print(f"  · mark {datetime.now().strftime('%H:%M:%S')}: {'  '.join(line)}",
              flush=True)


def _tick(conn, config, rnd, tick, verbose, debug=False):
    price = _price_fn()
    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    # snapshotted ONCE, before anyone this tick trades — see forward._held_by_rivals
    holdings_snapshot = {aid: set(json.loads(st["holdings"]).keys())
                        for aid, st in states.items()}
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
        n_stopped = forward.check_position_stops(conn, rnd["id"], aid, st, price, ts, config)
        if n_stopped:
            returns[aid] = (value(st) / st["starting_capital"] - 1) * 100
            if verbose:
                print(f"  {name}: {n_stopped} position(s) hit their per-position "
                      f"stop-loss", flush=True)
        stop = config.stop_loss_pct_for(rnd["length_days"])
        goal = rnd["goal_pct"]
        if returns[aid] <= -stop:
            forward._liquidate(conn, rnd["id"], aid, st, price, ts, reason="stop-loss")
            conn.commit()
            if verbose:
                print(f"  {name} STOPPED OUT at {returns[aid]:.1f}%", flush=True)
            continue
        if returns[aid] >= goal:
            forward._liquidate(conn, rnd["id"], aid, st, price, ts, reason="goal-hit")
            conn.commit()
            if verbose:
                print(f"  {name} BOOKED PROFIT at {returns[aid]:.1f}% (goal {goal:.1f}%)", flush=True)
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
            "rival_held_tickers": forward._held_by_rivals(holdings_snapshot, aid),
            "rival_messages": forward._recent_messages(conn, rnd["id"], aid),
        }
        orders, notes, message, trace = autonomous.decide(
            view, tick, 0, rnd["goal_pct"], guidelines, model=cfg.get("model"))
        orders, blocked = forward._drop_blocked_buys(orders, set(view["rival_held_tickers"]))
        if blocked and verbose:
            print(f"  {name}: blocked buy on {', '.join(blocked)} — "
                  f"already held by a rival", flush=True)
        if debug:
            _persist_audit(conn, rnd["id"], aid, trace)
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
            # local wall-clock ts (matches trades/fills), not forward._now()'s
            # UTC ISO — see _persist_audit for why mixing the two is a bug.
            conn.execute("INSERT INTO agent_messages (round_id, agent_id, ts, "
                         "message) VALUES (?,?,?,?)",
                         (rnd["id"], aid, ts, message))
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
            if debug:
                for step in trace:
                    tools = ", ".join(k for k in ("scan", "history", "fundamentals")
                                      if (step.get("research_requested") or {}).get(k))
                    print(f"         [debug] turn {step['turn']}"
                          f"{' · tools: ' + tools if tools else ''}: "
                          f"{step.get('thoughts','')[:100]}", flush=True)


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

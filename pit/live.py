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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import autonomous, brief as brief_mod, forward, market, replay, risk
from . import guidelines as gmod
from .config import CURRENCY, DEFAULT, MARKET, ArenaConfig


def _persist_audit(conn, round_id, agent_id, trace: list[dict]) -> None:
    """Write every turn of a decision (thoughts, tools used, results, final
    action) to `agent_audit`, for the debug-mode audit log on /live.

    Uses local wall-clock date+time (matching trades/fills, which use
    forward.local_ts() via _tick_body's `ts`) — not forward._now()'s UTC
    ISO. Mixing the two used to show audit/message timestamps hours off
    from the fills they belonged to (a full UTC-offset gap, e.g. IST is
    +5:30)."""
    ts = forward.local_ts()
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


# Orders each agent had rejected on its previous tick (agent_id -> reasons),
# fed back into its next decision. Without this an agent with $0 cash kept
# re-submitting the same unaffordable buy every tick, never learning why
# nothing happened.
_LAST_REJECTIONS: dict[int, list[str]] = {}


def benchmark_symbol() -> str:
    """Buy-and-hold reference for the race chart: BTC for crypto, SPY for US."""
    return os.getenv("PIT_BENCHMARK") or ("BTC-USD" if MARKET == "crypto" else "SPY")


def _mark_benchmark(conn, rnd, price) -> None:
    sym = benchmark_symbol()
    try:
        px = price(sym)
    except Exception:
        px = None
    if px:
        conn.execute("INSERT INTO benchmark_marks (round_id, ts, symbol, price) "
                     "VALUES (?,?,?,?)", (rnd["id"], forward.local_ts(), sym, px))


def _market_label() -> str:
    """Human label for what this session trades, shown on the leaderboard."""
    mode = market.TRADE_UNIVERSE_MODE
    base = ("Crypto" if MARKET == "crypto" else
            {"top100": "US top 100", "top500": "US S&P 500", "full": "US all-listed"}
            .get(mode, f"US {mode}") if MARKET == "us" else MARKET.upper())
    if replay._STATE and replay._STATE.get("day"):
        return f"{base} · replay {replay._STATE['day']}"
    return base


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
        # Each lineage's actual starting capital this round is its evolved
        # stake (round_states.starting_capital, set in forward.start_round
        # from lineages.current_stake), not the fixed config.base_capital —
        # that only applies to a lineage's very first-ever round. Printing
        # base_capital here used to claim "$100,000 paper each" even once
        # stakes had clearly diverged (a winner's grown, a loser's shrunk),
        # which is the whole point of the stake mechanic — so show what's
        # actually staked instead of a number that's wrong after round 1.
        stakes = sorted(r["starting_capital"] for r in conn.execute(
            "SELECT starting_capital FROM round_states WHERE round_id=?",
            (rnd["id"],)).fetchall())
        if stakes and min(stakes) != max(stakes):
            stake_txt = f"{CURRENCY}{min(stakes):,.0f}–{CURRENCY}{max(stakes):,.0f} paper (evolved stakes)"
        else:
            stake_txt = f"{CURRENCY}{(stakes[0] if stakes else config.base_capital):,.0f} paper each"
        print(f"● LIVE — {MARKET.upper()} market, {minutes} min. Decisions every "
              f"{interval // 60} min, prices marked every {refresh}s. "
              f"{stake_txt}. Watch /live.\n",
              flush=True)

    if not replay._STATE and brief_mod.brief_mode(market.TRADE_UNIVERSE_MODE):
        brief_mod.prefetch_fundamentals(brief_mod.symbols_for(market.TRADE_UNIVERSE_MODE),
                                        verbose)
    _set_risk_bands(conn, config, rnd, minutes, verbose)
    end = time.time() + minutes * 60
    conn.execute("UPDATE rounds SET ends_at=?, interval_s=?, market=? WHERE id=?",
                 (datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
                  int(interval), _market_label(), rnd["id"]))
    conn.commit()
    rnd = forward._active_round(conn)     # picks up ends_at + the risk bands
    for r in conn.execute("SELECT agent_id, final_return_pct FROM round_states "
                          "WHERE round_id=?", (rnd["id"],)).fetchall():
        conn.execute("INSERT INTO round_marks (round_id, agent_id, ts, return_pct) "
                     "VALUES (?,?,?,?)", (rnd["id"], r["agent_id"],
                                          forward.local_ts(), r["final_return_pct"] or 0.0))
    _mark_benchmark(conn, rnd, _price_fn())
    conn.commit()
    decision = 0
    decision_due = 0.0  # force a decision immediately
    while time.time() < end:
        now = time.time()
        try:
            if now >= decision_due:
                decision += 1
                tick_started = time.time()
                _tick(conn, config, rnd, decision, verbose, debug)
                _refresh_marks(conn, config, rnd, verbose)   # returns update even if ticks run back-to-back
                # from tick START: a tick that takes longer than `interval`
                # (slow local model) runs the next one straight away, instead
                # of idling another full `interval` after it finishes — that
                # made a "1 minute" cadence really ~4 minutes.
                decision_due = tick_started + interval
            else:
                _refresh_marks(conn, config, rnd, verbose)
        except Exception as exc:
            if verbose:
                print(f"  [loop error: {exc!r}]", flush=True)
        remaining = end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(refresh, remaining))

    # A tick's own LLM decision latency can by itself exceed the whole
    # session's wall-clock budget (3 agents x multi-turn research calls is
    # not fast) — when it does, the loop above exits with zero marks ever
    # having run, and every agent's return is still exactly the value it had
    # the instant its own order filled (0.00%, since a fill is priced at the
    # same quote used to compute post-trade value). Resolving straight off
    # that means the outcome is decided entirely by the trade-count/drawdown
    # tie-break cascade rather than any actual price movement. One guaranteed
    # mark-to-market pass here — after the loop, right before resolution —
    # ensures the round always reflects at least one real price refresh.
    if decision > 0:
        try:
            _refresh_marks(conn, config, rnd, verbose)
        except Exception as exc:
            if verbose:
                print(f"  [final mark error: {exc!r}]", flush=True)

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
            for r in (outcome.get("reflections")
                      or ([outcome["reflection"]] if outcome.get("reflection") else [])):
                label = "DO" if r.get("practice", "good") == "good" else "AVOID"
                if r.get("action"):     # re-test of an existing rule
                    from .guidelines import evidence_label
                    print(f"  rulebook → {label}: \"{r['text']}\" {r['action'].upper()} "
                          f"by the ledger ({evidence_label(r['evidence'])})", flush=True)
                    continue
                verdict = ("ADOPTED" + (f" on {r['tier']}" if r.get("tier") else "")
                           if r["accepted"] else
                           r["note"] if r.get("note") else
                           f"rejected — needed {r.get('required', 'a majority')}")
                who = f"{r['proposer']} proposes" if r.get("proposer") else "proposal"
                from .guidelines import evidence_label
                bt = evidence_label(r["ledger"]) if r.get("ledger") else "no backtest"
                print(f"  rulebook → {who} to {r['kind']} {label}: \"{r['text']}\"\n"
                      f"      backtest: {bt}\n"
                      f"      [{verdict}; {r['agree']}/{r['total']} agreed]", flush=True)
    else:
        outcome = None
    standings = _standings(conn, rnd, verbose)
    standings["outcome"] = outcome
    return standings


def _risk_view(rnd, config) -> dict:
    left = None
    try:
        if rnd["ends_at"]:
            left = max(0, round((datetime.strptime(rnd["ends_at"], "%Y-%m-%d %H:%M:%S")
                                 - datetime.now()).total_seconds() / 60, 1))
    except (KeyError, IndexError, ValueError):
        pass
    return {"portfolio_stop_pct": forward.round_stop_pct(rnd, config),
            "take_profit_pct": rnd["goal_pct"], "minutes_left": left,
            "expected_1sigma_move_over_round_pct": _row_get(rnd, "sigma_pct")}


def _row_get(row, key):
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _set_risk_bands(conn, config, rnd, minutes, verbose) -> None:
    """Size this round's stops and take-profit from its length and the
    universe's real volatility (pit/risk.py). Without volatility data (replay,
    or a universe with no brief) the flat config bands stay in force."""
    risk.clear()
    if replay._STATE or not brief_mod.brief_mode(market.TRADE_UNIVERSE_MODE):
        return
    try:
        b = brief_mod.get_brief(market.TRADE_UNIVERSE_MODE)
    except Exception:
        b = None
    vols = {s: f.get("dvol") for s, f in ((b or {}).get("features") or {}).items()}
    mkt = "crypto" if market.TRADE_UNIVERSE_MODE == "crypto" else "us"
    bd = risk.bands(vols, minutes, mkt, config.risk_reward_ratio)
    if not bd:
        return
    risk.set_round(minutes, mkt, vols)
    conn.execute("UPDATE rounds SET stop_pct=?, goal_pct=?, sigma_pct=?, auto_stops=? "
                 "WHERE id=?", (bd["stop"], bd["goal"], bd["sigma"],
                                json.dumps(bd["auto_stops"]), rnd["id"]))
    conn.commit()
    if verbose:
        st = sorted(bd["auto_stops"].values())
        print(f"[risk] {minutes}-min round: typical move ±{bd['sigma']:.2f}% (1σ) → "
              f"portfolio stop −{bd['stop']:.2f}%, take-profit +{bd['goal']:.2f}%, "
              f"position stops {st[0]:.2f}–{st[-1]:.2f}% by asset volatility",
              flush=True)


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
                                         price, forward.local_ts(), config)
        h = json.loads(st["holdings"])
        total = st["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())
        ret = (total / st["starting_capital"] - 1) * 100
        if st["status"] == "active":
            stop = forward.round_stop_pct(rnd, config)
            goal = rnd["goal_pct"]
            hit_reason = "stop-loss" if ret <= -stop else "goal-hit" if ret >= goal else None
            if hit_reason:
                forward._liquidate(conn, rnd["id"], st["agent_id"], st, price,
                                   forward.local_ts(), reason=hit_reason)
                total = st["current_capital"]
                ret = (total / st["starting_capital"] - 1) * 100
        held_now = json.loads(st["holdings"])
        conn.execute("UPDATE round_states SET final_return_pct=?, mark_prices=? "
                     "WHERE round_id=? AND agent_id=?",
                     (round(ret, 3), json.dumps({t: price(t) for t in held_now}),
                      rnd["id"], st["agent_id"]))
        conn.execute("INSERT INTO round_marks (round_id, agent_id, ts, return_pct) "
                     "VALUES (?,?,?,?)", (rnd["id"], st["agent_id"], forward.local_ts(),
                                          round(ret, 3)))
        line.append(f"{forward._name(conn, st['agent_id'])} {ret:+.2f}%")
    _mark_benchmark(conn, rnd, price)
    conn.commit()
    if verbose:
        print(f"  · mark {datetime.now().strftime('%H:%M:%S')}: {'  '.join(line)}",
              flush=True)


def _tick(conn, config, rnd, tick, verbose, debug=False):
    # The replay clock is NOT paused here. It used to be (so slow LLM
    # thinking couldn't burn simulated market time), but with agents deciding
    # concurrently and ticks scheduled from their start, a tick can fill the
    # whole interval — freezing the clock for the entire run: zero price
    # marks and every return stuck at 0.00% (seen live). Consistency inside a
    # tick doesn't need a frozen clock anyway: positions, fills and returns
    # all use the per-tick price cache (_price_fn), and the per-tick snapshot
    # keeps every agent's view identical. At sensible compression (compress
    # >= ~10) a tick is a few simulated minutes, nowhere near end-of-day.
    _tick_body(conn, config, rnd, tick, verbose, debug)


def _tick_body(conn, config, rnd, tick, verbose, debug=False):
    price = _price_fn()
    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    # snapshotted ONCE, before anyone this tick trades — see forward._held_by_rivals
    holdings_snapshot = {aid: set(json.loads(st["holdings"]).keys())
                        for aid, st in states.items()}
    guidelines = gmod.active_texts(conn)
    ts = forward.local_ts()

    def value(st):
        h = json.loads(st["holdings"])
        return st["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())

    returns = {aid: (value(st) / st["starting_capital"] - 1) * 100
               for aid, st in states.items()}
    conn.execute("UPDATE rounds SET last_tick_at=? WHERE id=?", (ts, rnd["id"]))
    conn.commit()
    if verbose:
        print(f"[tick {tick} · {ts}]", flush=True)

    jobs: list[dict] = []
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
        stop = forward.round_stop_pct(rnd, config)
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
        cb = json.loads(st.get("cost_basis") or "{}")
        sp = json.loads(st.get("stop_pcts") or "{}")
        positions = [{"ticker": t, "qty": q, "price": price(t),
                      "value": round(q * (price(t) or 0), 2),
                      "avg_entry": cb.get(t),
                      "stop_pct": sp.get(t, risk.auto_stop(t, config.position_stop_loss_pct)),
                      "stop_price": (round(cb[t] * (1 - sp.get(t, risk.auto_stop(t, config.position_stop_loss_pct)) / 100), 8)
                                     if cb.get(t) else None)}
                     for t, q in holdings.items()]
        view = {
            "risk": _risk_view(rnd, config),
            "cash": round(st["current_capital"], 2), "positions": positions,
            "total_value": round(value(st), 2), "return_pct": round(returns[aid], 2),
            "opponent_return_pct": round(
                max((r for a, r in returns.items() if a != aid), default=0.0), 2),
            "notes": cfg.get("notes", ""),
            "rival_held_tickers": forward._held_by_rivals(holdings_snapshot, aid),
            "rival_messages": forward._recent_messages(conn, rnd["id"], aid),
            "rival_recent_moves": forward._rival_recent_moves(conn, rnd["id"], aid),
            "rival_returns": {forward._name(conn, a): round(r, 2)
                              for a, r in returns.items() if a != aid},
            "orders_not_executed_last_tick": _LAST_REJECTIONS.get(aid, []),
        }
        jobs.append({"aid": aid, "st": st, "name": name, "cfg": cfg, "view": view})

    # Every agent decides from the SAME pre-tick snapshot (same messages,
    # same prices, same rival holdings). Previously agents decided one after
    # another and each one's trash talk was committed before the next agent
    # read messages, so whoever went last saw more than whoever went first
    # — contradicting "everyone trades the same round at the same time".
    # Threads overlap the LLM waits where the backend allows it (a hosted
    # API, or Ollama with OLLAMA_NUM_PARALLEL>1); a single-slot local server
    # just queues them, which costs nothing.
    # one shared market brief per tick (fundamentals + technicals for the
    # whole universe) -> every agent makes a single decision from it, instead
    # of a multi-turn research loop. Not in replay (needs historical data).
    brief = None
    if (jobs and not replay._STATE and brief_mod.brief_mode(market.TRADE_UNIVERSE_MODE)):
        try:
            brief = brief_mod.get_brief(market.TRADE_UNIVERSE_MODE)
        except Exception as exc:
            if verbose:
                print(f"  [brief unavailable, falling back to research mode: {exc!r}]",
                      flush=True)
    if verbose and brief:
        print(f"  [brief: {brief['n']} names, as of {brief['asof']}]", flush=True)
    kw = {"brief": brief} if brief else {}

    def _decide(job):
        return autonomous.decide(job["view"], tick, 0, rnd["goal_pct"], guidelines,
                                 model=job["cfg"].get("model"), **kw)
    if jobs:
        workers = len(jobs) if os.getenv("PIT_PARALLEL_AGENTS", "1") != "0" else 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            decisions = list(ex.map(_decide, jobs))
    else:
        decisions = []

    for job, (orders, notes, message, trace) in zip(jobs, decisions):
        aid, st, name, cfg, view = (job["aid"], job["st"], job["name"],
                                    job["cfg"], job["view"])
        orders, blocked = forward._drop_blocked_buys(orders, set(view["rival_held_tickers"]))
        if blocked and verbose:
            print(f"  {name}: blocked buy on {', '.join(blocked)} — "
                  f"already held by a rival", flush=True)
        if debug:
            _persist_audit(conn, rnd["id"], aid, trace)
        # Stamp fills/messages when this agent's decision actually lands —
        # `ts` above is the tick START, and a local model can spend a minute
        # or more thinking per agent, so tick-start stamps made fills look
        # earlier than the audit entries that produced them (and overstated
        # every position's "held" time).
        ts = forward.local_ts()
        n = forward._execute(conn, rnd["id"], aid, st, orders, price, ts,
                             features=(brief or {}).get("features"))
        _LAST_REJECTIONS[aid] = list(st.get("rejected_orders", []))
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
            for rej in st.get("rejected_orders", []):
                print(f"         ✗ not executed — {rej}", flush=True)
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

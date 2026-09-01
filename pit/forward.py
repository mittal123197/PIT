"""Forward (live) paper trading over real calendar days.

Unlike the compressed-replay engine, a forward round stays OPEN across real
trading days. Each day (call `step_round` after the NSE close) every agent
researches the real market and trades paper money; positions carry to the next
day. After `length_days` processed trading days the round resolves with the same
cascade + stakes + ELO + mutation as the replay engine.

Paper only: fills are at the real last close, no real orders anywhere.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from . import autonomous, market, rating
from . import guidelines as gmod
from .config import DEFAULT, ArenaConfig
from .resolve import AgentSummary, resolve


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().date().isoformat()


# ---- seeding two autonomous agents ------------------------------------

# Each agent gets a DIFFERENT brain on a DIFFERENT provider — not just a
# different model name, real infra diversity. OpenRouter's free ":free"
# models share one rate-limited pool across ALL of OpenRouter's free users
# (20/min, 50/day) — repeatedly exhausted this session — so RONIN and VIPER
# now run on DeepSeek's own paid API instead: cheap, not shared with anyone
# else's traffic. "deepseek:<model>" / "openrouter:<model>" prefixes route in
# llm.llm_chat; no prefix routes straight to Groq. Override any of these via
# env if you like.
_RONIN_MODEL = os.getenv("PIT_RONIN_MODEL", "deepseek:deepseek-v4-flash")
# Both default to the cheap model (flash, not pro — pro runs ~3x the cost)
# while spend is being kept deliberately small. Set PIT_VIPER_MODEL=
# deepseek:deepseek-v4-pro yourself once you're comfortable with the cost —
# that's the "different reasoning depth" experiment, just opt-in for now.
_VIPER_MODEL = os.getenv("PIT_VIPER_MODEL", "deepseek:deepseek-v4-flash")
# Plain Groq — no prefix routes straight to Groq (see llm.llm_chat) — a
# second, independent provider alongside DeepSeek.
_LYNX_MODEL = os.getenv("PIT_LYNX_MODEL", "openai/gpt-oss-20b")

AUTONOMOUS_SEED = [
    ("RONIN", {"mode": "autonomous", "notes": "", "model": _RONIN_MODEL}),
    ("VIPER", {"mode": "autonomous", "notes": "", "model": _VIPER_MODEL}),
    ("LYNX", {"mode": "autonomous", "notes": "", "model": _LYNX_MODEL}),
]


def ensure_autonomous_lineages(conn: sqlite3.Connection,
                               config: ArenaConfig = DEFAULT) -> list[int]:
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]
    if len(ids) >= len(AUTONOMOUS_SEED):
        return ids
    have = {r["name"] for r in conn.execute("SELECT name FROM lineages")}
    for name, cfg in AUTONOMOUS_SEED:
        if name not in have:
            _create_lineage(conn, name, cfg, config)
    return [r["id"] for r in conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]


def _create_lineage(conn, name, cfg, config) -> int:
    cur = conn.execute(
        "INSERT INTO lineages (name, current_stake, created_at) VALUES (?,?,?)",
        (name, config.base_capital, _now()))
    lid = cur.lastrowid
    aid = conn.execute(
        """INSERT INTO agents (lineage_id, generation, strategy_config,
           activity_profile, rating, created_at) VALUES (?,?,?,?,?,?)""",
        (lid, 1, json.dumps(cfg), json.dumps({"mode": "autonomous"}),
         config.elo_base, _now())).lastrowid
    conn.execute("UPDATE lineages SET current_agent_id=? WHERE id=?", (aid, lid))
    conn.commit()
    return lid


# ---- starting a forward round -----------------------------------------

def start_round(conn: sqlite3.Connection, config: ArenaConfig = DEFAULT,
                days: int | None = None) -> int:
    """Open a new live round with EVERY current lineage in the pool trading
    simultaneously — not a rotating 1v1 (that was tried and explicitly
    rejected: "3 agents should fight every battle"). `round_states` is the
    real roster for a round (queried by round_id everywhere); agent_a_id/
    agent_b_id are kept as legacy columns (first two lineages) only for old
    code/queries that still read them directly."""
    if _active_round(conn):
        raise RuntimeError("a live round is already in progress "
                           "(resolve/cancel it first)")
    ids = ensure_autonomous_lineages(conn, config)
    lineages = [conn.execute("SELECT * FROM lineages WHERE id=?", (i,)).fetchone()
               for i in ids]
    rnum = (conn.execute("SELECT COALESCE(MAX(round_number),0) n FROM rounds")
            .fetchone()["n"] + 1)
    # 0 means "time-based session, no fixed day count" (used by live sessions) —
    # `days or config...` would wrongly treat 0 as falsy and override it.
    length = days if days is not None else config.start_round_days
    rid = conn.execute(
        """INSERT INTO rounds (round_number, agent_a_id, agent_b_id, start_at,
           deadline, length_days, goal_pct, status, created_at)
           VALUES (?,?,?,?,?,?,?, 'live', ?)""",
        (rnum, lineages[0]["current_agent_id"], lineages[1]["current_agent_id"],
         _today(), "", length, config.goal_pct_for(length), _now())).lastrowid
    for lin in lineages:
        conn.execute(
            """INSERT INTO round_states (round_id, agent_id, starting_capital,
               current_capital, holdings) VALUES (?,?,?,?, '{}')""",
            (rid, lin["current_agent_id"], lin["current_stake"], lin["current_stake"]))
    conn.commit()
    return rid


def _active_round(conn):
    return conn.execute(
        "SELECT * FROM rounds WHERE status='live' ORDER BY id DESC LIMIT 1").fetchone()


# ---- stepping one real trading day ------------------------------------

def step_round(conn: sqlite3.Connection, config: ArenaConfig = DEFAULT,
               trade_date: str | None = None, verbose: bool = True,
               debug: bool = False) -> dict:
    """Process one trading day for the active live round. Idempotent per date."""
    rnd = _active_round(conn)
    if not rnd:
        return {"status": "no_active_round"}
    date = trade_date or _today()
    if not market.is_trading_day():
        return {"status": "not_a_trading_day", "date": date}
    already = conn.execute(
        "SELECT 1 FROM live_days WHERE round_id=? AND trade_date=?",
        (rnd["id"], date)).fetchone()
    if already:
        return {"status": "already_processed", "date": date}

    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    # snapshotted ONCE, before anyone this cycle trades — see _held_by_rivals
    holdings_snapshot = {aid: set(json.loads(st["holdings"]).keys())
                        for aid, st in states.items()}
    guidelines = gmod.active_texts(conn)
    quote_cache: dict[str, float | None] = {}

    def price(t):
        if t not in quote_cache:
            q = market.quote(t)
            quote_cache[t] = q["price"] if q else None
        return quote_cache[t]

    def value(st):
        h = json.loads(st["holdings"])
        return st["current_capital"] + sum(
            qty * (price(t) or 0.0) for t, qty in h.items())

    # start-of-day scoreboard (snapshot before anyone trades)
    returns = {aid: (value(st) / st["starting_capital"] - 1) * 100
               for aid, st in states.items()}

    day_index = conn.execute(
        "SELECT COUNT(*) n FROM live_days WHERE round_id=?", (rnd["id"],)
    ).fetchone()["n"] + 1
    logs = []

    for aid, st in states.items():
        if st["status"] != "active":
            continue
        # per-position hard stop — independent of the portfolio-level one
        # below: cuts a single collapsing holding on its own, before the
        # AGGREGATE book has to fall this far to react.
        n_stopped = check_position_stops(conn, rnd["id"], aid, st, price, date, config)
        if n_stopped:
            logs.append(f"{_name(conn, aid)}: {n_stopped} position(s) hit their "
                        f"per-position stop-loss")
            returns[aid] = (value(st) / st["starting_capital"] - 1) * 100
        # stop-loss / take-profit (both hard constraints, portfolio-level)
        if returns[aid] <= -config.stop_loss_pct_for(rnd["length_days"]):
            _liquidate(conn, rnd["id"], aid, st, price, date, reason="stop-loss")
            logs.append(f"{_name(conn, aid)} stopped out at {returns[aid]:.1f}%")
            continue
        if returns[aid] >= rnd["goal_pct"]:
            _liquidate(conn, rnd["id"], aid, st, price, date, reason="goal-hit")
            logs.append(f"{_name(conn, aid)} booked profit at {returns[aid]:.1f}%")
            continue

        agent = conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
        cfg = json.loads(agent["strategy_config"])
        holdings = json.loads(st["holdings"])
        positions = [{"ticker": t, "qty": q, "price": price(t),
                      "value": round(q * (price(t) or 0), 2)}
                     for t, q in holdings.items()]
        view = {
            "cash": round(st["current_capital"], 2),
            "positions": positions,
            "total_value": round(value(st), 2),
            "return_pct": round(returns[aid], 2),
            "opponent_return_pct": round(
                max((r for a, r in returns.items() if a != aid), default=0.0), 2),
            "notes": cfg.get("notes", ""),
            "rival_held_tickers": _held_by_rivals(holdings_snapshot, aid),
        }
        view["rival_messages"] = _recent_messages(conn, rnd["id"], aid)
        orders, notes, message, trace = autonomous.decide(
            view, day_index, rnd["length_days"], rnd["goal_pct"], guidelines,
            model=cfg.get("model"))
        orders, blocked = _drop_blocked_buys(orders, set(view["rival_held_tickers"]))
        if blocked:
            logs.append(f"{_name(conn, aid)}: blocked buy on "
                        f"{', '.join(blocked)} — already held by a rival")
        if debug:
            from .live import _persist_audit
            _persist_audit(conn, rnd["id"], aid, trace)
        n_exec = _execute(conn, rnd["id"], aid, st, orders, price, date)
        if message:
            conn.execute("INSERT INTO agent_messages (round_id, agent_id, ts, "
                         "message) VALUES (?,?,?,?)", (rnd["id"], aid, _now(), message))
        cfg["notes"] = notes
        conn.execute("UPDATE agents SET strategy_config=? WHERE id=?",
                     (json.dumps(cfg), aid))
        conn.execute("UPDATE round_states SET current_capital=?, holdings=? "
                     "WHERE round_id=? AND agent_id=?",
                     (st["current_capital"], json.dumps(json.loads(st["holdings"])),
                      rnd["id"], aid))
        logs.append(f"{_name(conn, aid)}: {n_exec} trades, ret {returns[aid]:+.2f}%")

    # mark drawdown + persist day
    for aid, st in states.items():
        _update_drawdown(conn, rnd["id"], aid, returns[aid])
    conn.execute("INSERT INTO live_days (round_id, trade_date, processed_at) "
                 "VALUES (?,?,?)", (rnd["id"], date, _now()))
    conn.commit()

    done = conn.execute("SELECT COUNT(*) n FROM live_days WHERE round_id=?",
                        (rnd["id"],)).fetchone()["n"]
    if verbose:
        for l in logs:
            print("  " + l)
    if done >= rnd["length_days"]:
        outcome = _resolve(conn, rnd, config, price)
        return {"status": "resolved", "day": done, "date": date,
                "logs": logs, "outcome": outcome}
    return {"status": "stepped", "day": done, "of": rnd["length_days"],
            "date": date, "logs": logs}


def _execute(conn, round_id, agent_id, st, orders, price, date) -> int:
    n = 0
    holdings = json.loads(st["holdings"])
    cost_basis = json.loads(st.get("cost_basis") or "{}")
    cash = st["current_capital"]
    for o in orders:
        px = price(o["ticker"])
        if not px or px <= 0:
            continue
        if o["side"] == "buy":
            budget = o.get("amount_inr") or (o.get("qty", 0) * px)
            qty = float(int(min(budget, cash) // px))
            if qty <= 0:
                continue
            cash -= qty * px
            prev_qty = holdings.get(o["ticker"], 0.0)
            prev_cost = cost_basis.get(o["ticker"], px)
            # weighted-average entry price across all buys of this ticker —
            # drives the per-position stop-loss below
            cost_basis[o["ticker"]] = ((prev_qty * prev_cost + qty * px)
                                       / (prev_qty + qty))
            holdings[o["ticker"]] = prev_qty + qty
        else:  # sell
            held = holdings.get(o["ticker"], 0.0)
            qty = float(int(o.get("qty", held) if "qty" in o
                            else (o.get("amount_inr", held * px) / px)))
            qty = min(qty, held)
            if qty <= 0:
                continue
            cash += qty * px
            rem = held - qty
            if rem <= 1e-9:
                holdings.pop(o["ticker"], None)
                cost_basis.pop(o["ticker"], None)
            else:
                holdings[o["ticker"]] = rem
                # avg cost basis is unchanged by a partial sell
        conn.execute(
            """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
               price, capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)""",
            (round_id, agent_id, date, o["ticker"], o["side"], qty, px, cash,
             o.get("reason", "")))
        n += 1
    st["current_capital"] = cash
    st["holdings"] = json.dumps(holdings)
    st["cost_basis"] = json.dumps(cost_basis)
    conn.execute("UPDATE round_states SET trade_count=trade_count+?, cost_basis=? "
                 "WHERE round_id=? AND agent_id=?",
                 (n, json.dumps(cost_basis), round_id, agent_id))
    return n


def check_position_stops(conn, round_id, agent_id, st, price, date, config) -> int:
    """Per-position hard stop: force-sell any SINGLE holding that has fallen
    `position_stop_loss_pct` below its own average entry price — independent
    of the portfolio-level stop-loss, which only fires on the AGGREGATE book.
    Returns how many positions were force-sold. Mutates `st` in place."""
    holdings = json.loads(st["holdings"])
    if not holdings:
        return 0
    cost_basis = json.loads(st.get("cost_basis") or "{}")
    cash = st["current_capital"]
    sold = 0
    for ticker, qty in list(holdings.items()):
        entry = cost_basis.get(ticker)
        if not entry:
            continue
        px = price(ticker)
        if not px or px <= 0:
            continue
        if px <= entry * (1 - config.position_stop_loss_pct / 100.0):
            cash += qty * px
            holdings.pop(ticker, None)
            cost_basis.pop(ticker, None)
            conn.execute(
                """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
                   price, capital_after, reason) VALUES (?,?,?,?,'sell',?,?,?,?)""",
                (round_id, agent_id, date, ticker, qty, px, cash,
                 f"position stop-loss (entry {entry:.2f}, -{config.position_stop_loss_pct:.0f}%)"))
            sold += 1
    if sold:
        st["current_capital"] = cash
        st["holdings"] = json.dumps(holdings)
        st["cost_basis"] = json.dumps(cost_basis)
        conn.execute("UPDATE round_states SET current_capital=?, holdings=?, "
                     "cost_basis=?, trade_count=trade_count+? "
                     "WHERE round_id=? AND agent_id=?",
                     (cash, st["holdings"], st["cost_basis"], sold, round_id, agent_id))
    return sold


def close_out_all_positions(conn, round_id, agent_id, st, price, date,
                            reason="round-end close") -> None:
    """Force-sell every open position for real at round end — final_return_pct
    should reflect REALIZED cash, not a paper mark on positions nobody
    actually sold. Mutates `st` in place. Safe to call on an agent already
    fully in cash (no-op)."""
    holdings = json.loads(st["holdings"])
    if not holdings:
        return
    cash = st["current_capital"]
    for ticker, qty in list(holdings.items()):
        px = price(ticker)
        if not px or px <= 0:
            continue  # can't close without a price; leave it (rare, e.g. delisted)
        cash += qty * px
        conn.execute(
            """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
               price, capital_after, reason) VALUES (?,?,?,?,'sell',?,?,?,?)""",
            (round_id, agent_id, date, ticker, qty, px, cash, reason))
        holdings.pop(ticker, None)
    st["current_capital"] = cash
    st["holdings"] = json.dumps(holdings)
    st["cost_basis"] = "{}"
    conn.execute("UPDATE round_states SET current_capital=?, holdings=?, "
                 "cost_basis='{}' WHERE round_id=? AND agent_id=?",
                 (cash, st["holdings"], round_id, agent_id))


def _liquidate(conn, round_id, agent_id, st, price, date, reason="stop-loss"):
    """Force-close every open position. `reason` is either 'stop-loss' (hard
    downside constraint) or 'goal-hit' (hard take-profit constraint) — both
    freeze the agent for the rest of the round, same mechanism either way."""
    holdings = json.loads(st["holdings"])
    cash = st["current_capital"]
    for t, qty in list(holdings.items()):
        px = price(t)
        if px:
            cash += qty * px
            conn.execute(
                """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
                   price, capital_after, reason) VALUES (?,?,?,?, 'sell', ?,?,?,?)""",
                (round_id, agent_id, date, t, qty, px, cash, reason))
    st["current_capital"] = cash
    st["holdings"] = "{}"
    status = "goal_hit" if reason == "goal-hit" else "liquidated"
    conn.execute("UPDATE round_states SET current_capital=?, holdings='{}', "
                 "status=?, liquidated_at=? WHERE round_id=? AND agent_id=?",
                 (cash, status, date, round_id, agent_id))


def _update_drawdown(conn, round_id, agent_id, ret_pct):
    row = conn.execute("SELECT max_drawdown_pct FROM round_states "
                       "WHERE round_id=? AND agent_id=?", (round_id, agent_id)).fetchone()
    dd = max(row["max_drawdown_pct"], -ret_pct if ret_pct < 0 else 0.0)
    conn.execute("UPDATE round_states SET max_drawdown_pct=? "
                 "WHERE round_id=? AND agent_id=?", (dd, round_id, agent_id))


# ---- resolution -------------------------------------------------------

def _rank_participants(summaries: list, eps: float) -> list[int]:
    """Full ranking, best to worst, no ties possible — same cascade as the
    original 2-agent case (higher return, then fewer trades, then lower
    drawdown), generalized to N. The final tiebreak is agent_id: a full
    N-way "sudden death" mini-round doesn't generalize cleanly past 2, so an
    exact tie this deep just falls to a deterministic order instead."""
    def key(s):
        bucket = round(s.return_pct / eps) if eps else s.return_pct
        return (-bucket, s.trade_count, s.max_drawdown_pct, s.agent_id)
    return [s.agent_id for s in sorted(summaries, key=key)]


def _resolve(conn, rnd, config, price) -> dict:
    """Resolve a round with EVERY participant (2 or more) ranked at once —
    not just a single winner/loser pair. Rank 1 reinforces; EVERY other rank
    is a loser and self-critiques into a new generation (explicit design
    choice: only 1st truly wins), but only dead-last takes the full stake
    penalty — anyone strictly in the middle takes a smaller one."""
    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    summaries = {}
    for aid, st in states.items():
        # Force-close every open position for real at the deadline — a
        # final_return_pct built on a PAPER mark of positions nobody actually
        # sold is not a realized result. No-op for an agent already fully in
        # cash (e.g. already stopped/goal-hit earlier in the round).
        close_out_all_positions(conn, rnd["id"], aid, st, price, _now())
        h = json.loads(st["holdings"])
        total = st["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())
        ret = (total / st["starting_capital"] - 1) * 100
        conn.execute("UPDATE round_states SET final_return_pct=? "
                     "WHERE round_id=? AND agent_id=?", (ret, rnd["id"], aid))
        summaries[aid] = AgentSummary(aid, ret, st["trade_count"],
                                      st["max_drawdown_pct"],
                                      st["status"] in ("liquidated", "goal_hit"))

    ranking = _rank_participants(list(summaries.values()), config.tie_epsilon_pct)
    winner_id, loser_id = ranking[0], ranking[-1]
    # reuse the pairwise cascade purely to get a human-readable "reason"
    # string for the top-vs-bottom placement (return / trade_count / drawdown)
    reason = resolve(summaries[winner_id], summaries[loser_id],
                     config.tie_epsilon_pct).reason
    agents_by_id = {aid: conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
                    for aid in ranking}

    # Pairwise ELO: every pair implied by the ranking gets a standard 2-player
    # update (K scaled down by the number of pairs, so a round with more
    # participants doesn't move ratings proportionally further). Deltas are
    # summed and applied once per agent at the end.
    n_pairs = len(ranking) * (len(ranking) - 1) // 2
    k_per_pair = config.elo_k / max(1, n_pairs)
    net_delta = {aid: 0.0 for aid in ranking}
    for i, hi in enumerate(ranking):
        for lo in ranking[i + 1:]:
            _, _, d = rating.update(agents_by_id[hi]["rating"] + net_delta[hi],
                                    agents_by_id[lo]["rating"] + net_delta[lo],
                                    k_per_pair)
            net_delta[hi] += d
            net_delta[lo] -= d
    for aid in ranking:
        conn.execute("UPDATE agents SET rating=? WHERE id=?",
                     (agents_by_id[aid]["rating"] + net_delta[aid], aid))

    # Everyone stopped out — even the "winner" — means nobody actually
    # succeeded; rank 1 gets loser treatment too instead of a false
    # "reinforcement" story (generalizes the old both_stopped_out rule).
    all_stopped_out = all(states[a]["status"] == "liquidated" for a in ranking)
    # A win earned with zero trades didn't beat anyone's trading — it just
    # outlasted a rival who lost on its own. Cap the reward accordingly.
    passive_win = states[winner_id]["trade_count"] == 0

    stake_mult, notes = {}, {}
    if all_stopped_out:
        notes[winner_id] = _reflect_loser(conn, rnd["id"], agents_by_id[winner_id],
                                          agents_by_id[winner_id]["rating"] + net_delta[winner_id],
                                          summaries[winner_id].return_pct)
    else:
        notes[winner_id] = _reflect_winner(conn, rnd["id"], agents_by_id[winner_id],
                                           summaries[winner_id].return_pct)
    wd = (config.passive_win_stake_bonus_pct if passive_win
         else config.win_stake_bonus_pct) / 100.0
    stake_mult[winner_id] = 1 + wd
    _apply_lineage(conn, agents_by_id[winner_id]["lineage_id"],
                   summaries[winner_id].return_pct, True, stake_mult[winner_id])

    for aid in ranking[1:]:
        is_last = aid == loser_id
        penalty_pct = (config.loss_stake_penalty_pct if is_last
                      else config.middle_place_penalty_pct)
        stake_mult[aid] = 1 - penalty_pct / 100.0
        notes[aid] = _reflect_loser(conn, rnd["id"], agents_by_id[aid],
                                    agents_by_id[aid]["rating"] + net_delta[aid],
                                    summaries[aid].return_pct)
        _apply_lineage(conn, agents_by_id[aid]["lineage_id"],
                       summaries[aid].return_pct, False, stake_mult[aid])

    for rank_pos, aid in enumerate(ranking, start=1):
        conn.execute(
            """INSERT INTO round_rankings (round_id, agent_id, rank, return_pct,
               note, stake_mult, rating_delta, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (rnd["id"], aid, rank_pos, summaries[aid].return_pct, notes[aid],
             stake_mult[aid], net_delta[aid], _now()))

    # round_results is kept only for backward-compat display (a single
    # winner/loser pair) — round_rankings above is the full, real record.
    conn.execute(
        """INSERT INTO round_results (round_id, winner_agent_id, loser_agent_id,
           resolution_reason, winner_return_pct, loser_return_pct, rating_delta,
           winner_note, loser_note, both_stopped_out, passive_win, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rnd["id"], winner_id, loser_id, reason, summaries[winner_id].return_pct,
         summaries[loser_id].return_pct, net_delta[winner_id], notes[winner_id],
         notes[loser_id], int(all_stopped_out), int(passive_win), _now()))

    # Shared "constitution": every reflection_interval rounds, look across ALL
    # lineages for a pattern that keeps recurring (not a one-off) and let the
    # pool vote on codifying it. Accepted guidelines are injected into every
    # agent's context from then on — the "commonly agreed good practices"
    # layer, independent of and in addition to each lineage's own reflection.
    # Everyone stopping out ALWAYS forces this pass early, regardless of the
    # interval — the whole pool just failed together in the same conditions,
    # which is exactly the kind of shared signal worth discussing immediately.
    reflection = None
    if all_stopped_out or (config.reflection_enabled
            and rnd["round_number"] % config.reflection_interval == 0):
        from . import guidelines as gmod
        from .llm import groq_available
        reflection = gmod.open_and_resolve(conn, config, rnd["id"],
                                           use_llm=groq_available())

    conn.execute("UPDATE rounds SET status='resolved', deadline=? WHERE id=?",
                 (_today(), rnd["id"]))
    conn.commit()
    return {"winner": _name(conn, winner_id), "loser": _name(conn, loser_id),
            "reason": reason, "winner_return": round(summaries[winner_id].return_pct, 2),
            "loser_return": round(summaries[loser_id].return_pct, 2),
            "winner_note": notes[winner_id], "mutation_note": notes[loser_id],
            "both_stopped_out": all_stopped_out, "passive_win": passive_win,
            "ranking": [{"agent_id": aid, "name": _name(conn, aid),
                        "rank": i + 1, "return_pct": round(summaries[aid].return_pct, 2),
                        "note": notes[aid]} for i, aid in enumerate(ranking)],
            "reflection": reflection}


def _apply_lineage(conn, lid, ret, won, mult):
    lin = conn.execute("SELECT * FROM lineages WHERE id=?", (lid,)).fetchone()
    new_stake = max(1.0, lin["current_stake"] * mult)
    new_cum = ((1 + lin["cumulative_return_pct"] / 100) * (1 + ret / 100) - 1) * 100
    conn.execute("""UPDATE lineages SET current_stake=?, cumulative_return_pct=?,
                 wins=wins+?, losses=losses+? WHERE id=?""",
                 (round(new_stake, 2), round(new_cum, 4), int(won), int(not won), lid))


def _own_trades(conn, round_id, agent_id) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT ts,symbol,side,qty,price,reason FROM trades "
        "WHERE round_id=? AND agent_id=? ORDER BY id", (round_id, agent_id))]


def _reflect_winner(conn, round_id, wa, return_pct) -> str:
    """Reinforcement, not recreation: the winner studies its OWN trades and
    writes notes on what specifically worked, so it keeps doing it. Same
    agent row, same generation — a win doesn't spawn a new agent, it just
    sharpens the current one."""
    trades = _own_trades(conn, round_id, wa["id"])
    cfg = json.loads(wa["strategy_config"])
    note = _self_reflection(cfg.get("notes", ""), trades, return_pct, won=True)
    cfg["notes"] = note
    conn.execute("UPDATE agents SET strategy_config=? WHERE id=?",
                 (json.dumps(cfg), wa["id"]))
    _share_lesson(conn, round_id, wa["id"], note)
    return note[:200]


def _reflect_loser(conn, round_id, la, carried_rating, return_pct) -> str:
    """Self-critique, not recreation: the loser studies its OWN trades — not
    the winner's — and writes corrective notes on its own mistakes. Still
    advances to a new generation (this attempt is retired), but seeded from
    itself, not from copying whoever beat it."""
    trades = _own_trades(conn, round_id, la["id"])
    old_cfg = json.loads(la["strategy_config"])
    note = _self_reflection(old_cfg.get("notes", ""), trades, return_pct, won=False)
    # Preserve the model across generations — this was dropped for a long
    # time, silently falling back to decide()'s `model or DEFAULT_MODEL`
    # (Groq) for every agent that had ever lost even once, regardless of
    # which model it was actually supposed to run.
    new_cfg = {"mode": "autonomous", "notes": note, "model": old_cfg.get("model")}
    new_id = conn.execute(
        """INSERT INTO agents (lineage_id, generation, strategy_config,
           activity_profile, rating, seeded_from_trade_agent_id, mutation_note,
           created_at) VALUES (?,?,?,?,?,?,?,?)""",
        (la["lineage_id"], la["generation"] + 1, json.dumps(new_cfg),
         json.dumps({"mode": "autonomous"}), carried_rating, la["id"],
         note[:200], _now())).lastrowid
    conn.execute("UPDATE lineages SET current_agent_id=? WHERE id=?",
                 (new_id, la["lineage_id"]))
    _share_lesson(conn, round_id, la["id"], note)
    return note[:200]


def _share_lesson(conn, round_id, agent_id, note) -> None:
    """Post a self-reflection note as a message the RIVAL will actually see
    on its next decision (via _recent_messages) — this is the "communication"
    the pool draws its shared guidelines from: agents don't just silently
    self-improve, they broadcast what they learned.

    Local wall-clock time, not _now()'s UTC ISO — matches trash-talk messages
    and fills, which both display local time; mixing bases made lessons show
    hours off from everything else on the same page."""
    conn.execute(
        "INSERT INTO agent_messages (round_id, agent_id, ts, kind, message) "
        "VALUES (?,?,?,'lesson',?)",
        (round_id, agent_id, datetime.now().strftime("%H:%M:%S"),
         f"\U0001f4dd Lesson from this round: {note}"))


def _self_reflection(old_notes, own_trades, return_pct, won: bool) -> str:
    """Learn from one's OWN trades (LLM if available, else a plain summary).
    Never looks at the opponent's trades — this is self-critique/reinforcement,
    not copying whoever won."""
    if not own_trades:
        # Nothing to reflect on. If this was a "win", flag plainly that it
        # wasn't earned by trading — either a genuine all-cash hold, or a
        # decision loop that silently failed (LLM/network error) and never
        # got the chance to act. Either way, no false "what worked" note.
        if won:
            return ("Made zero trades this round — this win wasn't earned by "
                    "beating the rival's trading, it just avoided their loss. "
                    "Investigate whether this was a deliberate hold or a failed "
                    "decision call before assuming this is a real strategy.")
        return "Made zero trades and still lost — likely a decision/data failure, not a strategy choice."
    try:
        from .llm import DEFAULT_MODEL, _client, _extra_for, groq_available
        if groq_available():
            import json as _j
            if won:
                instruction = (
                    f"You WON this round at {return_pct:+.2f}%. Study your OWN "
                    "trades below and identify what specifically worked (entry "
                    "timing, ticker selection, fundamentals check, exit "
                    "discipline). Write concise reinforcement notes so you keep "
                    "doing this. <=400 chars. Return JSON {\"notes\":\"...\"}.")
            else:
                instruction = (
                    f"You LOST this round at {return_pct:+.2f}%. Study your OWN "
                    "trades below and identify your own mistakes (bad entries, "
                    "poor timing, ignoring risk, chasing hype, wrong sizing). "
                    "Write concise corrective notes for next time — self-critique "
                    "only, do not reference any other trader. <=400 chars. "
                    "Return JSON {\"notes\":\"...\"}.")
            prompt = {"your_own_trades": own_trades[:40],
                      "your_old_notes": old_notes, "instruction": instruction}
            r = _client().chat.completions.create(
                model=DEFAULT_MODEL, response_format={"type": "json_object"},
                messages=[{"role": "system",
                          "content": "You are a trader reflecting on your own "
                                     "performance. JSON only."},
                          {"role": "user", "content": _j.dumps(prompt)}],
                temperature=0.8, **_extra_for(DEFAULT_MODEL))
            return "LLM: " + str(_j.loads(r.choices[0].message.content).get("notes", ""))[:380]
    except Exception:
        pass
    syms = ", ".join(sorted({t["symbol"] for t in own_trades})[:6]) or "nothing"
    verb = "Worked" if won else "Rethink"
    return f"{verb}: traded {syms} at {return_pct:+.2f}%."


def _held_by_rivals(holdings_snapshot: dict, self_aid: int) -> list[str]:
    """Every ticker any OTHER participant held as of THIS tick's start — the
    "no duplicate stock" hard rule. Snapshotted once per tick/day, before
    anyone this cycle has traded — not updated as agents execute within the
    same cycle, so processing order can't hand one agent a race-condition
    advantage over the others (whoever a for-loop happens to reach first)."""
    held = set()
    for aid, tickers in holdings_snapshot.items():
        if aid != self_aid:
            held |= tickers
    return sorted(held)


def _drop_blocked_buys(orders: list[dict], blocked: set) -> tuple[list[dict], list[str]]:
    """Strip any BUY for a ticker a rival already holds (see
    _held_by_rivals) — sells are never blocked. Server-side enforcement,
    independent of whether the agent's own reasoning respected the rule."""
    clean, rejected = [], []
    for o in orders:
        if o.get("side") == "buy" and o.get("ticker") in blocked:
            rejected.append(o["ticker"])
            continue
        clean.append(o)
    return clean, rejected


def _recent_messages(conn, round_id, agent_id, limit=4) -> list[str]:
    """ALL OTHER PARTICIPANTS' recent messages this round (not this agent's
    own), oldest-first: this round's banter/lessons, plus each rival's most
    recent lesson from any round if it isn't already in that window.

    round_states is the real roster for a round (works for 2 or N agents) —
    NOT rounds.agent_a_id/agent_b_id, which only ever holds two lineages and
    would silently drop rivals in a 3+-way battle. Lessons are posted at
    round-resolution time (see _share_lesson), scoped to the round that just
    ENDED — without the fallback, the very next round would never surface
    the self-reflection a rival just shared."""
    rivals = conn.execute(
        "SELECT a.id, a.lineage_id FROM round_states rs "
        "JOIN agents a ON a.id = rs.agent_id "
        "WHERE rs.round_id=? AND rs.agent_id!=?", (round_id, agent_id)).fetchall()
    if not rivals:
        return []
    rival_ids = [r["id"] for r in rivals]
    rival_lineages = {r["lineage_id"] for r in rivals}
    placeholders = ",".join("?" * len(rival_ids))
    rows = conn.execute(
        f"SELECT m.message, l.name FROM agent_messages m "
        f"JOIN agents a ON a.id=m.agent_id JOIN lineages l ON l.id=a.lineage_id "
        f"WHERE m.round_id=? AND m.agent_id IN ({placeholders}) "
        f"ORDER BY m.id DESC LIMIT ?",
        (round_id, *rival_ids, limit)).fetchall()
    msgs = [f"{r['name']}: {r['message']}" for r in reversed(rows)]
    for lineage_id in rival_lineages:
        last_lesson = conn.execute(
            "SELECT m.message, l.name FROM agent_messages m "
            "JOIN agents a ON a.id=m.agent_id JOIN lineages l ON l.id=a.lineage_id "
            "WHERE a.lineage_id=? AND m.kind='lesson' ORDER BY m.id DESC LIMIT 1",
            (lineage_id,)).fetchone()
        if last_lesson:
            tag = f"{last_lesson['name']}: {last_lesson['message']}"
            if tag not in msgs:
                msgs.append(tag)
    return msgs


def _name(conn, agent_id) -> str:
    return conn.execute(
        "SELECT l.name FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE a.id=?", (agent_id,)).fetchone()["name"]


def status(conn) -> dict | None:
    rnd = _active_round(conn)
    if not rnd:
        return None
    done = conn.execute("SELECT COUNT(*) n FROM live_days WHERE round_id=?",
                        (rnd["id"],)).fetchone()["n"]
    return {"round_id": rnd["id"], "round_number": rnd["round_number"],
            "days_done": done, "length_days": rnd["length_days"],
            "goal_pct": rnd["goal_pct"], "started": rnd["start_at"]}

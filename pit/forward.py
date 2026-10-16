"""A round's lifecycle: the pool, orders and fills, stops, resolution, evolution.

  start_round      every lineage's current agent gets the same flat stake
  _execute         an agent's orders -> fills (sells first, buys scaled to
                   cash, half the spread paid, per-position stop set)
  check_position_stops / _liquidate / close_out_all_positions
                   the arena's forced exits (position stop, portfolio stop,
                   take-profit, deadline)
  _resolve         rank the pool, ELO, stakes, each agent's self-reflection
                   (winner reinforces, everyone else becomes a new
                   generation), charter revision, then the rulebook pass

pit/live.py drives a round in real (or replayed) time; this module owns what
happens to money and to agents. Paper only — no real orders anywhere.
"""
from __future__ import annotations

import json
import math
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


def local_ts() -> str:
    """Local wall-clock date+time for anything persisted to the trade ledger,
    messages, or audit log — date included, not just time, so a trade in the
    ledger is unambiguous on its own instead of relying on "well it must have
    been today." Every writer of a persisted timestamp in this codebase uses
    this (or forward._now()'s UTC ISO, deliberately, for a handful of
    non-trade metadata rows) — see the module docstring notes on past bugs
    from mixing UTC and local timestamps on the same page."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---- seeding autonomous agents -----------------------------------------

# Each lineage's model, provider-prefixed (see llm.llm_chat). Set in .env;
# an existing agent is switched to the configured model at the next round.
_RONIN_MODEL = os.getenv("PIT_RONIN_MODEL", "deepseek:deepseek-v4-flash")
_VIPER_MODEL = os.getenv("PIT_VIPER_MODEL", "deepseek:deepseek-v4-flash")
_LYNX_MODEL = os.getenv("PIT_LYNX_MODEL", "deepseek:deepseek-v4-flash")

AUTONOMOUS_SEED = [
    ("RONIN", {"mode": "autonomous", "notes": "", "model": _RONIN_MODEL}),
    ("VIPER", {"mode": "autonomous", "notes": "", "model": _VIPER_MODEL}),
    ("LYNX", {"mode": "autonomous", "notes": "", "model": _LYNX_MODEL}),
]


def ensure_autonomous_lineages(conn: sqlite3.Connection,
                               config: ArenaConfig = DEFAULT) -> list[int]:
    have = {r["name"] for r in conn.execute("SELECT name FROM lineages")}
    for name, cfg in AUTONOMOUS_SEED:
        if name not in have:
            _create_lineage(conn, name, cfg, config)
    _sync_models(conn)
    return [r["id"] for r in conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]


def _sync_models(conn) -> None:
    """The model is stored on each agent (and carried to its successors), so
    a changed PIT_<NAME>_MODEL in .env would otherwise never take effect for
    an existing pool. The configured model wins at every round start."""
    want = dict((n, c["model"]) for n, c in AUTONOMOUS_SEED)
    for r in conn.execute("""SELECT l.name, a.id, a.strategy_config FROM lineages l
                             JOIN agents a ON a.id = l.current_agent_id""").fetchall():
        cfg = json.loads(r["strategy_config"] or "{}")
        if r["name"] in want and cfg.get("model") != want[r["name"]]:
            cfg["model"] = want[r["name"]]
            conn.execute("UPDATE agents SET strategy_config=? WHERE id=?",
                         (json.dumps(cfg), r["id"]))
    conn.commit()


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
            (rid, lin["current_agent_id"], config.base_capital, config.base_capital))
    conn.commit()
    return rid


def _active_round(conn):
    return conn.execute(
        "SELECT * FROM rounds WHERE status='live' ORDER BY id DESC LIMIT 1").fetchone()


# ---- stepping one real trading day ------------------------------------



# Bounds on an agent-chosen per-position stop (stop_loss_pct on a buy).
STOP_PCT_MIN, STOP_PCT_MAX = 0.5, 20.0


def _stop_pct_for(order: dict, default: float) -> float:
    try:
        v = float(order.get("stop_loss_pct"))
    except (TypeError, ValueError):
        return default
    return min(STOP_PCT_MAX, max(STOP_PCT_MIN, v))


# Smallest order worth executing. Without it agents made $0.02 "trades"
# (0.0001 shares) that count as trades and clutter the ledger.
MIN_ORDER_USD = 1.0


def _position_state(conn, round_id, agent_id, symbol, px, date, avg) -> dict:
    """Where the position stood when the agent chose to sell it: P&L vs its
    average entry and minutes held since the first buy this round — the
    fields exit rules ('AVOID selling when pos_pnl% > -0.3 and held_min < 15')
    are written over."""
    out = {}
    if avg:
        out["pos_pnl"] = round((px / avg - 1) * 100, 3)
    first = conn.execute("SELECT MIN(ts) t FROM trades WHERE round_id=? AND agent_id=? "
                         "AND symbol=? AND side='buy'", (round_id, agent_id, symbol)).fetchone()
    try:
        from datetime import datetime
        f = datetime.strptime(first["t"][:19], "%Y-%m-%d %H:%M:%S")
        d = datetime.strptime(str(date)[:19], "%Y-%m-%d %H:%M:%S")
        out["held_min"] = round((d - f).total_seconds() / 60, 1)
    except Exception:
        pass
    return out


def record_end_prices(conn, round_id, price) -> None:
    """Every symbol traded this round, priced at the deadline — the yardstick
    for 'was that sell better than holding?' (exit-rule backtests)."""
    syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM trades WHERE round_id=?",
                                       (round_id,)).fetchall()]
    out = {}
    for s_ in syms:
        try:
            v = price(s_)
        except Exception:
            v = None
        if v:
            out[s_] = v
    conn.execute("UPDATE rounds SET end_prices=? WHERE id=?", (json.dumps(out), round_id))


def _fill(px: float, side: str) -> float:
    """The price a fill actually gets: the quote plus/minus half the spread."""
    b = DEFAULT.spread_bps / 10_000
    return px * (1 + b) if side == "buy" else px * (1 - b)


def _amount(o: dict) -> float | None:
    """Order size in account currency ("amount"; "amount_inr" is the old name)."""
    return o.get("amount") or o.get("amount_inr")


def _floor_qty(x: float) -> float:
    """Floor to 4 decimals (fractional shares) — never rounds UP, so a buy
    can't exceed the cash/budget it was sized from."""
    return math.floor(x * 10_000 + 1e-9) / 10_000


def round_stop_pct(rnd, config) -> float:
    """The round's portfolio stop: volatility-scaled when the session set one
    (rounds.stop_pct, see pit/risk.py), else the flat sqrt(days) band."""
    try:
        v = rnd["stop_pct"]
    except (KeyError, IndexError):
        v = None
    return v if v else config.stop_loss_pct_for(rnd["length_days"])


def _execute(conn, round_id, agent_id, st, orders, price, date,
             features: dict | None = None) -> int:
    from . import risk
    n = 0
    rejected = []   # surfaced to the live console/page; used to be silent
    holdings = json.loads(st["holdings"])
    cost_basis = json.loads(st.get("cost_basis") or "{}")
    stop_pcts = json.loads(st.get("stop_pcts") or "{}")
    cash = st["current_capital"]
    # Sells first (they free cash for this decision's buys), then buys. If the
    # buys together ask for more than the cash on hand, scale every buy down
    # pro rata — the agent's intended split survives. Executing in order used
    # to let the first buy take everything and reject the rest (an agent that
    # split $1,000 over three coins got one coin and two rejections).
    orders = ([o for o in orders if o.get("side") == "sell"] +
              [o for o in orders if o.get("side") == "buy"])
    _scale = None
    sell_avg = None
    for o in orders:
        px = price(o["ticker"])
        if not px or px <= 0:
            rejected.append(f"{o['side']} {o['ticker']}: no price available")
            continue
        px = _fill(px, o["side"])
        if o["side"] == "buy":
            if _scale is None:      # first buy: cash now includes this decision's sells
                wants = sum((_amount(b) or (b.get("qty", 0) * (price(b["ticker"]) or 0)))
                            for b in orders if b.get("side") == "buy")
                _scale = min(1.0, cash / wants) if wants > 0 else 1.0
            budget = (_amount(o) or (o.get("qty", 0) * px)) * _scale
            # Fractional shares: at a $1,000-scale stake, whole-share sizing
            # silently dropped every order for a stock priced above the
            # agent's budget (e.g. LITE at $1,114 vs a $560 budget -> 0
            # shares -> order discarded, agent retries it every tick).
            qty = _floor_qty(min(budget, cash) / px)
            if qty <= 0 or qty * px < MIN_ORDER_USD:
                rejected.append(f"buy {o['ticker']}: ${min(budget, cash):,.2f} "
                                f"available, nothing to buy at ${px:,.2f}")
                continue
            cash -= qty * px
            prev_qty = holdings.get(o["ticker"], 0.0)
            prev_cost = cost_basis.get(o["ticker"], px)
            # weighted-average entry price across all buys of this ticker —
            # drives the per-position stop-loss below
            cost_basis[o["ticker"]] = ((prev_qty * prev_cost + qty * px)
                                       / (prev_qty + qty))
            holdings[o["ticker"]] = prev_qty + qty
            # every buy carries a stop by default; the agent may set its own
            # distance (bounded) — a later buy of the same name can reset it
            if "stop_loss_pct" in o or o["ticker"] not in stop_pcts:
                stop_pcts[o["ticker"]] = _stop_pct_for(
                    o, risk.auto_stop(o["ticker"], DEFAULT.position_stop_loss_pct))
        else:  # sell
            sell_avg = cost_basis.get(o["ticker"])
            held = holdings.get(o["ticker"], 0.0)
            qty = _floor_qty(o.get("qty", held) if "qty" in o
                             else ((_amount(o) or held * px) / px))
            qty = min(qty, held)
            # selling (almost) the whole position closes it: flooring to 4
            # decimals otherwise strands dust (0.0001 shares) in holdings
            if (held - qty) * px < MIN_ORDER_USD:
                qty = held
            if qty <= 0:
                rejected.append(f"sell {o['ticker']}: nothing held to sell")
                continue
            cash += qty * px
            rem = held - qty
            if rem <= 1e-9:
                holdings.pop(o["ticker"], None)
                cost_basis.pop(o["ticker"], None)
                stop_pcts.pop(o["ticker"], None)
            else:
                holdings[o["ticker"]] = rem
                # avg cost basis is unchanged by a partial sell
        feat = (features or {}).get(o["ticker"])
        if o["side"] == "sell" and feat is not None:
            feat = dict(feat, **_position_state(conn, round_id, agent_id, o["ticker"],
                                                 px, date, sell_avg))
        conn.execute(
            """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
               price, capital_after, reason, features) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (round_id, agent_id, date, o["ticker"], o["side"], qty, px, cash,
             o.get("reason", ""), json.dumps(feat) if feat else None))
        n += 1
    st["current_capital"] = cash
    st["holdings"] = json.dumps(holdings)
    st["cost_basis"] = json.dumps(cost_basis)
    st["rejected_orders"] = rejected
    st["stop_pcts"] = json.dumps(stop_pcts)
    conn.execute("UPDATE round_states SET trade_count=trade_count+?, cost_basis=?, "
                 "stop_pcts=? WHERE round_id=? AND agent_id=?",
                 (n, json.dumps(cost_basis), st["stop_pcts"], round_id, agent_id))
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
    stop_pcts = json.loads(st.get("stop_pcts") or "{}")
    cash = st["current_capital"]
    sold = 0
    for ticker, qty in list(holdings.items()):
        entry = cost_basis.get(ticker)
        if not entry:
            continue
        px = price(ticker)
        if not px or px <= 0:
            continue
        pct = stop_pcts.get(ticker, config.position_stop_loss_pct)
        if px <= entry * (1 - pct / 100.0):
            px = _fill(px, "sell")
            cash += qty * px
            holdings.pop(ticker, None)
            cost_basis.pop(ticker, None)
            stop_pcts.pop(ticker, None)
            conn.execute(
                """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
                   price, capital_after, reason) VALUES (?,?,?,?,'sell',?,?,?,?)""",
                (round_id, agent_id, date, ticker, qty, px, cash,
                 f"position stop-loss (entry {entry:.6g}, -{pct:g}%)"))
            sold += 1
    if sold:
        st["current_capital"] = cash
        st["holdings"] = json.dumps(holdings)
        st["cost_basis"] = json.dumps(cost_basis)
        st["stop_pcts"] = json.dumps(stop_pcts)
        conn.execute("UPDATE round_states SET current_capital=?, holdings=?, "
                     "cost_basis=?, stop_pcts=?, trade_count=trade_count+? "
                     "WHERE round_id=? AND agent_id=?",
                     (cash, st["holdings"], st["cost_basis"], st["stop_pcts"], sold,
                      round_id, agent_id))
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
        px = _fill(px, "sell")
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
            px = _fill(px, "sell")
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
    record_end_prices(conn, rnd["id"], price)
    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    summaries = {}
    for aid, st in states.items():
        # Force-close every open position for real at the deadline — a
        # final_return_pct built on a PAPER mark of positions nobody actually
        # sold is not a realized result. No-op for an agent already fully in
        # cash (e.g. already stopped/goal-hit earlier in the round).
        # Local wall-clock time, not _now()'s UTC ISO — every other trade
        # this round (fills, messages, audit) already uses datetime.now()
        # (see _persist_audit in live.py), so a round-end close-out stamped
        # in UTC showed up hours earlier than the ticks that preceded it in
        # the same trade ledger (e.g. "08:50" right after fills at "14:15").
        close_out_ts = local_ts()
        close_out_all_positions(conn, rnd["id"], aid, st, price, close_out_ts)
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

    notes = {}
    if all_stopped_out:
        notes[winner_id] = _reflect_loser(conn, rnd["id"], agents_by_id[winner_id],
                                          agents_by_id[winner_id]["rating"] + net_delta[winner_id],
                                          summaries[winner_id].return_pct)
    else:
        notes[winner_id] = _reflect_winner(conn, rnd["id"], agents_by_id[winner_id],
                                           summaries[winner_id].return_pct)
    # Every agent starts every round with the same base capital (flat stakes:
    # compounding let early luck snowball into a capital edge). Rank 1
    # reinforces; every other rank self-critiques into a new generation.
    stake_mult = {aid: 1.0 for aid in ranking}
    _apply_lineage(conn, agents_by_id[winner_id]["lineage_id"],
                   summaries[winner_id].return_pct, True)
    for aid in ranking[1:]:
        notes[aid] = _reflect_loser(conn, rnd["id"], agents_by_id[aid],
                                    agents_by_id[aid]["rating"] + net_delta[aid],
                                    summaries[aid].return_pct)
        _apply_lineage(conn, agents_by_id[aid]["lineage_id"],
                       summaries[aid].return_pct, False)

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

    # The pool's rulebook pass (review -> propose -> research -> debate ->
    # vote, see guidelines.pool_constitution) — after every round by default,
    # always when everyone stopped out. Needs at least one agent's model to be
    # reachable; otherwise the rulebook simply waits for the next round.
    reflection, reflections = None, []
    if all_stopped_out or (config.reflection_enabled
            and rnd["round_number"] % config.reflection_interval == 0):
        from . import guidelines as gmod
        from .llm import llm_available
        if any(llm_available(l.get("model")) for l in
               (json.loads(r["strategy_config"] or "{}") for r in conn.execute(
                   "SELECT a.strategy_config FROM lineages l "
                   "JOIN agents a ON a.id = l.current_agent_id").fetchall())):
            reflections = gmod.pool_constitution(conn, config, rnd["id"])
            reflection = next((r for r in reflections if r.get("kind")), None)

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
            "reflection": reflection, "reflections": reflections}


def _apply_lineage(conn, lid, ret, won):
    lin = conn.execute("SELECT * FROM lineages WHERE id=?", (lid,)).fetchone()
    new_cum = ((1 + lin["cumulative_return_pct"] / 100) * (1 + ret / 100) - 1) * 100
    conn.execute("""UPDATE lineages SET cumulative_return_pct=?,
                 wins=wins+?, losses=losses+? WHERE id=?""",
                 (round(new_cum, 4), int(won), int(not won), lid))




FORCED_REASONS = ("round-end close", "stop-loss", "goal-hit")


def is_forced(reason: str | None) -> bool:
    """A fill the ARENA made (deadline close-out, portfolio stop, take-profit,
    per-position stop) rather than the agent."""
    r = reason or ""
    return r in FORCED_REASONS or r.startswith("position stop")


def _agent_ts(ts):
    from .replay import agent_ts
    return agent_ts(ts)


def sell_record(conn, round_id, agent_id, price) -> dict:
    """A factual mirror of the agent's OWN sell decisions this round (not
    advice): how many it made, how many were profitable, what they netted,
    and what simply holding those shares would be worth at `price` now.
    Arena-forced exits are excluded — the agent didn't choose them."""
    rows = conn.execute(
        "SELECT symbol, side, qty, price, reason FROM trades "
        "WHERE round_id=? AND agent_id=? ORDER BY id", (round_id, agent_id)).fetchall()
    held, cost = {}, {}
    n = wins = 0
    net = held_instead = paid = 0.0
    b = DEFAULT.spread_bps / 10_000
    for r in rows:
        sym, q, px = r["symbol"], r["qty"], r["price"]
        paid += q * px * b / (1 + b)
        if r["side"] == "buy":
            tot = held.get(sym, 0.0) + q
            cost[sym] = (held.get(sym, 0.0) * cost.get(sym, px) + q * px) / tot
            held[sym] = tot
            continue
        avg = cost.get(sym, px)
        held[sym] = max(0.0, held.get(sym, 0.0) - q)
        if is_forced(r["reason"]):
            continue
        pnl = (px - avg) * q
        n += 1
        wins += pnl > 0
        net += pnl
        now = price(sym)
        held_instead += ((now - avg) * q) if now else pnl
    return {"sells_you_chose": n, "profitable": wins,
            "net_from_those_sells": round(net, 2),
            "if_you_had_held_those_shares_instead": round(held_instead, 2),
            "trading_costs_paid_so_far": round(paid, 2)}


def _rival_charters(conn, lineage_id) -> dict:
    out = {}
    for r in conn.execute("""SELECT l.name, json_extract(a.strategy_config, '$.charter') c
                             FROM lineages l JOIN agents a ON a.id = l.current_agent_id
                             WHERE l.id != ?""", (lineage_id,)).fetchall():
        if r["c"]:
            out[r["name"]] = r["c"]
    return out


def _quote_px(sym):
    from . import market
    try:
        return (market.quote(sym) or {}).get("price")
    except Exception:
        return None


def _is_off_charter(reason: str | None) -> bool:
    r = reason or ""
    return r.startswith("⚠ off-charter") or r.startswith("⚠ no charter")


def charter_record(conn, round_id, agent_id, price) -> dict:
    """The agent's charter discipline this round, as plain numbers: buys that
    fit its charter vs. buys it marked (or left) off-charter, and what each
    group has made so far (exit = its later sells, else the current price).
    An agent whose setup never appeared went 13-for-13 off-charter, lost
    most, and kept the charter unchanged — this puts that in front of it."""
    rows = conn.execute("SELECT id, symbol, side, qty, price, reason FROM trades "
                        "WHERE round_id=? AND agent_id=? ORDER BY id",
                        (round_id, agent_id)).fetchall()
    groups = {"on_charter": {"buys": 0, "net_usd": 0.0},
              "off_charter": {"buys": 0, "net_usd": 0.0}}
    for b in rows:
        if b["side"] != "buy":
            continue
        later = [r for r in rows if r["side"] == "sell" and r["symbol"] == b["symbol"]
                 and r["id"] > b["id"]]
        q = sum(r["qty"] for r in later)
        exit_px = (sum(r["qty"] * r["price"] for r in later) / q) if q else price(b["symbol"])
        if not exit_px:
            continue
        g = groups["off_charter" if _is_off_charter(b["reason"]) else "on_charter"]
        g["buys"] += 1
        g["net_usd"] += b["qty"] * (exit_px - b["price"])
    for g in groups.values():
        g["net_usd"] = round(g["net_usd"], 2)
    return groups


def _own_trades(conn, round_id, agent_id) -> list[dict]:
    """The agent's fills this round, each flagged forced_by_arena where the
    arena made it. Unflagged, the deadline close-out read like the agent's own
    choice — agents critiqued themselves for 'rushing to close at round-end'
    and voted in rules about it, though they can't control it."""
    rows = [dict(r) for r in conn.execute(
        "SELECT ts,symbol,side,qty,price,reason FROM trades "
        "WHERE round_id=? AND agent_id=? ORDER BY id", (round_id, agent_id))]
    from .replay import agent_ts
    from . import market
    end_px: dict = {}
    for r in rows:
        r["forced_by_arena"] = is_forced(r["reason"])
        r["ts"] = agent_ts(r["ts"])
        # score each sell the agent CHOSE against simply holding to the end
        if r["side"] == "sell" and not r["forced_by_arena"]:
            if r["symbol"] not in end_px:
                try:
                    end_px[r["symbol"]] = (market.quote(r["symbol"]) or {}).get("price")
                except Exception:
                    end_px[r["symbol"]] = None
            e = end_px[r["symbol"]]
            if e and r["price"]:
                r["price_at_round_end"] = round(e, 4)
                r["holding_instead_vs_selling_pct"] = round((e / r["price"] - 1) * 100, 2)
    return rows


def _reflect_winner(conn, round_id, wa, return_pct) -> str:
    """Reinforcement, not recreation: the winner studies its OWN trades and
    writes notes on what specifically worked, so it keeps doing it. Same
    agent row, same generation — a win doesn't spawn a new agent, it just
    sharpens the current one."""
    trades = _own_trades(conn, round_id, wa["id"])
    cfg = json.loads(wa["strategy_config"])
    note = _self_reflection(cfg.get("notes", ""), trades, return_pct, won=True,
                           model=cfg.get("model"), has_charter=bool(cfg.get("charter")))
    cfg["notes"] = note
    if cfg.get("charter"):
        from .charter import revise_charter
        n = conn.execute("SELECT COUNT(*) FROM round_states WHERE round_id=?", (round_id,)).fetchone()[0]
        from .charter import accept as _accept_charter
        _accept_charter(cfg, revise_charter(cfg.get("model"), cfg["charter"], trades, return_pct, 1, n,
                                           charter_record(conn, round_id, wa["id"], _quote_px),
                                           card=cfg.get("charter_card"), spec=cfg.get("charter_spec"),
                                           rivals=_rival_charters(conn, wa["lineage_id"])))
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
    note = _self_reflection(old_cfg.get("notes", ""), trades, return_pct, won=False,
                           model=old_cfg.get("model"), has_charter=bool(old_cfg.get("charter")))
    # Preserve the model across generations — this was dropped for a long
    # time, silently falling back to decide()'s `model or DEFAULT_MODEL`
    # (Groq) for every agent that had ever lost even once, regardless of
    # which model it was actually supposed to run.
    new_cfg = {"mode": "autonomous", "notes": note, "model": old_cfg.get("model")}
    if old_cfg.get("charter"):
        from .charter import revise_charter
        # rankings are written after reflection — rank from the round's returns
        n = conn.execute("SELECT COUNT(*) FROM round_states WHERE round_id=?", (round_id,)).fetchone()[0]
        better = conn.execute("SELECT COUNT(*) FROM round_states WHERE round_id=? "
                              "AND final_return_pct > ?", (round_id, return_pct)).fetchone()[0]
        from .charter import accept as _accept_charter
        for k in ("charter", "charter_spec", "charter_card", "charter_last_good"):
            if k in old_cfg:
                new_cfg[k] = old_cfg[k]
        _accept_charter(new_cfg, revise_charter(old_cfg.get("model"), old_cfg["charter"], trades,
                                               return_pct, min(n, better + 1), n,
                                               charter_record(conn, round_id, la["id"], _quote_px),
                                               card=old_cfg.get("charter_card"),
                                               spec=old_cfg.get("charter_spec"),
                                               rivals=_rival_charters(conn, la["lineage_id"])))
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
        (round_id, agent_id, local_ts(),
         f"\U0001f4dd Lesson from this round: {_redact_tickers(note)}"))


def _self_reflection(old_notes, own_trades, return_pct, won: bool,
                     model: str | None = None, has_charter: bool = False) -> str:
    """Learn from one's OWN trades (LLM if available, else a plain summary).
    Never looks at the opponent's trades — this is self-critique/reinforcement,
    not copying whoever won."""
    if not own_trades:
        # Nothing to reflect on. If this was a "win", flag plainly that it
        # wasn't earned by trading — either a genuine all-cash hold, or a
        # decision loop that silently failed (LLM/network error) and never
        # got the chance to act. Either way, no false "what worked" note.
        # With a charter, sitting in cash is usually deliberate (no setup met
        # its conditions) — say that rather than "a data failure".
        if has_charter:
            return ((old_notes + " | " if old_notes else "") +
                    "Held cash all round: no name met my charter's entry conditions. "
                    "If that keeps happening, my setup may be too rare for these round "
                    "lengths.")[-600:]
        if won:
            return ("Made zero trades this round — this win wasn't earned by "
                    "beating the rival's trading, it just avoided their loss. "
                    "Investigate whether this was a deliberate hold or a failed "
                    "decision call before assuming this is a real strategy.")
        return "Made zero trades and still lost — likely a decision/data failure, not a strategy choice."
    try:
        from .llm import DEFAULT_MODEL, llm_available, llm_chat
        if llm_available(model):
            import json as _j
            # The agent reflects with its OWN brain (its own model), and the
            # question is open-ended. The old prompt handed every agent the
            # same checklist ("entry timing, ticker selection, exit discipline"
            # / "bad entries, chasing hype, wrong sizing"), so every critique
            # came back in the same words regardless of what actually happened
            # — a lens we imposed, not a lesson the agent drew.
            outcome = "won" if won else "did not win"
            instruction = (
                f"The round is over: you {outcome} at {return_pct:+.2f}%. Look "
                "at your OWN trades below and your notes from before. In your "
                "own words: what does this round actually tell you about how you "
                "should trade next time — what to keep doing and what to change? "
                "Be specific to these trades and this market, not generic advice. "
                "Trades marked forced_by_arena were made by the arena (the "
                "deadline close-out or a stop-loss), not by you — judge your "
                "entries, sizing and your own sells, not those. Each of your "
                "own sells shows the price at round end and "
                "holding_instead_vs_selling_pct (positive = holding would have "
                "done better than selling). "
                "Self-critique only, don't reference other traders. <=400 chars. "
                "Return JSON {\"notes\":\"...\"}.")
            prompt = {"your_own_trades": own_trades[:40],
                      "your_old_notes": old_notes, "instruction": instruction}
            content = llm_chat(model or DEFAULT_MODEL,
                               [{"role": "system",
                                 "content": "You are a trader reflecting on your "
                                            "own performance. JSON only."},
                                {"role": "user", "content": _j.dumps(prompt)}],
                               temperature=0.7, json_mode=True)
            return "LLM: " + str(_j.loads(content).get("notes", ""))[:380]
    except Exception:
        pass
    syms = ", ".join(sorted({t["symbol"] for t in own_trades})[:6]) or "nothing"
    verb = "Worked" if won else "Rethink"
    return f"{verb}: traded {syms} at {return_pct:+.2f}%."






def _recent_messages(conn, round_id, agent_id, limit=4,
                     this_round_only: bool = False) -> list[str]:
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
        f"SELECT m.message, m.kind, l.name FROM agent_messages m "
        f"JOIN agents a ON a.id=m.agent_id JOIN lineages l ON l.id=a.lineage_id "
        f"WHERE m.round_id=? AND m.agent_id IN ({placeholders}) "
        f"ORDER BY m.id DESC LIMIT ?",
        (round_id, *rival_ids, limit)).fetchall()
    # stock names are hidden in every rival message — taunts ("watch META
    # run") leaked picks just as lessons did, now that holdings are private
    msgs = [f"{r['name']}: {_redact_tickers(r['message'])}" for r in reversed(rows)]
    if this_round_only:   # replay: lessons from earlier rounds may know how the day ends
        return msgs
    for lineage_id in rival_lineages:
        last_lesson = conn.execute(
            "SELECT m.message, l.name FROM agent_messages m "
            "JOIN agents a ON a.id=m.agent_id JOIN lineages l ON l.id=a.lineage_id "
            "WHERE a.lineage_id=? AND m.kind='lesson' ORDER BY m.id DESC LIMIT 1",
            (lineage_id,)).fetchone()
        if last_lesson:
            tag = f"{last_lesson['name']}: {_redact_tickers(last_lesson['message'])}"
            if tag not in msgs:
                msgs.append(tag)
    return msgs


def _redact_tickers(text: str) -> str:
    """A rival's lesson should pass on the PRINCIPLE, not the pick. Lessons
    like "CRM/META/INTU were green at close" became tips: next round all three
    agents opened in META and CRM (100% book overlap). Ticker symbols from the
    tradeable universe are replaced with "[a stock]"; the agent's own notes
    are untouched."""
    import re
    try:
        from .guidelines import _universe_symbols
        syms = _universe_symbols()
    except Exception:
        syms = set()
    if not syms:
        return text
    def _sub(m):
        tok = m.group(0)
        if tok not in syms:
            return tok
        # "20-day MA", "50d MA": a moving average, not Mastercard
        if tok == "MA" and re.search(r"\d+\s*-?\s*(d|day)s?\s*$", text[max(0, m.start() - 10):m.start()]):
            return tok
        return "[a stock]"
    return re.sub(r"\b[A-Z][A-Z0-9.\-]{0,9}\b", _sub, text)


def _name(conn, agent_id) -> str:
    return conn.execute(
        "SELECT l.name FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE a.id=?", (agent_id,)).fetchone()["name"]


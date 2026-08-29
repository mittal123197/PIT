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

# Each agent gets a DIFFERENT frontier brain (via OpenRouter) — same goal,
# same freedom, different reasoning. Override with env if you like.
_RONIN_MODEL = os.getenv("PIT_RONIN_MODEL",
                         "openrouter:nvidia/nemotron-3-super-120b-a12b:free")
# A finance-tuned model vs. a general frontier model — a real experiment in
# whether domain specialization actually helps here.
_VIPER_MODEL = os.getenv("PIT_VIPER_MODEL",
                         "openrouter:inclusionai/ling-3.0-flash-fin:free")

AUTONOMOUS_SEED = [
    ("RONIN", {"mode": "autonomous", "notes": "", "model": _RONIN_MODEL}),
    ("VIPER", {"mode": "autonomous", "notes": "", "model": _VIPER_MODEL}),
]


def ensure_autonomous_lineages(conn: sqlite3.Connection,
                               config: ArenaConfig = DEFAULT) -> list[int]:
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]
    if len(ids) >= 2:
        return ids
    for name, cfg in AUTONOMOUS_SEED:
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
    """Open a new live round between the two current agents. Returns round id."""
    if _active_round(conn):
        raise RuntimeError("a live round is already in progress "
                           "(resolve/cancel it first)")
    ids = ensure_autonomous_lineages(conn, config)
    la = conn.execute("SELECT * FROM lineages WHERE id=?", (ids[0],)).fetchone()
    lb = conn.execute("SELECT * FROM lineages WHERE id=?", (ids[1],)).fetchone()
    # 0 means "time-based session, no fixed day count" (used by live sessions) —
    # `days or config...` would wrongly treat 0 as falsy and override it.
    length = days if days is not None else config.start_round_days
    rnum = (conn.execute("SELECT COALESCE(MAX(round_number),0) n FROM rounds")
            .fetchone()["n"] + 1)
    rid = conn.execute(
        """INSERT INTO rounds (round_number, agent_a_id, agent_b_id, start_at,
           deadline, length_days, goal_pct, status, created_at)
           VALUES (?,?,?,?,?,?,?, 'live', ?)""",
        (rnum, la["current_agent_id"], lb["current_agent_id"], _today(),
         "", length, config.goal_pct_for(length), _now())).lastrowid
    for lin in (la, lb):
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
        # stop-loss (hard constraint)
        if returns[aid] <= -config.stop_loss_pct:
            _liquidate(conn, rnd["id"], aid, st, price, date)
            logs.append(f"{_name(conn, aid)} stopped out at {returns[aid]:.1f}%")
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
        }
        view["rival_messages"] = _recent_messages(conn, rnd["id"], aid)
        orders, notes, message, trace = autonomous.decide(
            view, day_index, rnd["length_days"], rnd["goal_pct"], guidelines,
            model=cfg.get("model"))
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
            holdings[o["ticker"]] = holdings.get(o["ticker"], 0.0) + qty
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
            else:
                holdings[o["ticker"]] = rem
        conn.execute(
            """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
               price, capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)""",
            (round_id, agent_id, date, o["ticker"], o["side"], qty, px, cash,
             o.get("reason", "")))
        n += 1
    st["current_capital"] = cash
    st["holdings"] = json.dumps(holdings)
    conn.execute("UPDATE round_states SET trade_count=trade_count+? "
                 "WHERE round_id=? AND agent_id=?", (n, round_id, agent_id))
    return n


def _liquidate(conn, round_id, agent_id, st, price, date):
    holdings = json.loads(st["holdings"])
    cash = st["current_capital"]
    for t, qty in list(holdings.items()):
        px = price(t)
        if px:
            cash += qty * px
            conn.execute(
                """INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty,
                   price, capital_after, reason) VALUES (?,?,?,?, 'sell', ?,?,?, 'stop-loss')""",
                (round_id, agent_id, date, t, qty, px, cash))
    st["current_capital"] = cash
    st["holdings"] = "{}"
    conn.execute("UPDATE round_states SET current_capital=?, holdings='{}', "
                 "status='liquidated', liquidated_at=? WHERE round_id=? AND agent_id=?",
                 (cash, date, round_id, agent_id))


def _update_drawdown(conn, round_id, agent_id, ret_pct):
    row = conn.execute("SELECT max_drawdown_pct FROM round_states "
                       "WHERE round_id=? AND agent_id=?", (round_id, agent_id)).fetchone()
    dd = max(row["max_drawdown_pct"], -ret_pct if ret_pct < 0 else 0.0)
    conn.execute("UPDATE round_states SET max_drawdown_pct=? "
                 "WHERE round_id=? AND agent_id=?", (dd, round_id, agent_id))


# ---- resolution -------------------------------------------------------

def _resolve(conn, rnd, config, price) -> dict:
    states = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_states WHERE round_id=?", (rnd["id"],)).fetchall()}
    summaries = {}
    for aid, st in states.items():
        h = json.loads(st["holdings"])
        total = st["current_capital"] + sum(q * (price(t) or 0) for t, q in h.items())
        ret = (total / st["starting_capital"] - 1) * 100
        conn.execute("UPDATE round_states SET final_return_pct=? "
                     "WHERE round_id=? AND agent_id=?", (ret, rnd["id"], aid))
        summaries[aid] = AgentSummary(aid, ret, st["trade_count"],
                                      st["max_drawdown_pct"],
                                      st["status"] == "liquidated")
    a, b = list(summaries.values())
    res = resolve(a, b, config.tie_epsilon_pct)
    winner_id = res.winner_id or min(summaries)  # extreme-tie fallback
    loser_id = res.loser_id or max(summaries)
    reason = res.reason

    wa = conn.execute("SELECT * FROM agents WHERE id=?", (winner_id,)).fetchone()
    la = conn.execute("SELECT * FROM agents WHERE id=?", (loser_id,)).fetchone()
    nw, nl, delta = rating.update(wa["rating"], la["rating"], config.elo_k)
    conn.execute("UPDATE agents SET rating=? WHERE id=?", (nw, winner_id))
    conn.execute("UPDATE agents SET rating=? WHERE id=?", (nl, loser_id))

    wd = config.win_stake_bonus_pct / 100.0
    ld = config.loss_stake_penalty_pct / 100.0
    _apply_lineage(conn, wa["lineage_id"], summaries[winner_id].return_pct, True, 1 + wd)
    _apply_lineage(conn, la["lineage_id"], summaries[loser_id].return_pct, False, 1 - ld)

    conn.execute(
        """INSERT INTO round_results (round_id, winner_agent_id, loser_agent_id,
           resolution_reason, winner_return_pct, loser_return_pct, rating_delta,
           created_at) VALUES (?,?,?,?,?,?,?,?)""",
        (rnd["id"], winner_id, loser_id, reason, summaries[winner_id].return_pct,
         summaries[loser_id].return_pct, delta, _now()))

    note = _mutate_loser(conn, rnd["id"], la, wa, nl)
    conn.execute("UPDATE rounds SET status='resolved', deadline=? WHERE id=?",
                 (_today(), rnd["id"]))
    conn.commit()
    return {"winner": _name(conn, winner_id), "loser": _name(conn, loser_id),
            "reason": reason, "winner_return": round(summaries[winner_id].return_pct, 2),
            "loser_return": round(summaries[loser_id].return_pct, 2),
            "mutation_note": note}


def _apply_lineage(conn, lid, ret, won, mult):
    lin = conn.execute("SELECT * FROM lineages WHERE id=?", (lid,)).fetchone()
    new_stake = max(1.0, lin["current_stake"] * mult)
    new_cum = ((1 + lin["cumulative_return_pct"] / 100) * (1 + ret / 100) - 1) * 100
    conn.execute("""UPDATE lineages SET current_stake=?, cumulative_return_pct=?,
                 wins=wins+?, losses=losses+? WHERE id=?""",
                 (round(new_stake, 2), round(new_cum, 4), int(won), int(not won), lid))


def _mutate_loser(conn, round_id, la, wa, carried_rating) -> str:
    winner_trades = [dict(r) for r in conn.execute(
        "SELECT ts,symbol,side,qty,price,reason FROM trades "
        "WHERE round_id=? AND agent_id=? ORDER BY id", (round_id, wa["id"]))]
    note = _autonomous_mutation(json.loads(la["strategy_config"]), winner_trades)
    new_cfg = {"mode": "autonomous", "notes": note,
               "persona": "rebuilt from the trader who beat me"}
    new_id = conn.execute(
        """INSERT INTO agents (lineage_id, generation, strategy_config,
           activity_profile, rating, seeded_from_trade_agent_id, mutation_note,
           created_at) VALUES (?,?,?,?,?,?,?,?)""",
        (la["lineage_id"], la["generation"] + 1, json.dumps(new_cfg),
         json.dumps({"mode": "autonomous"}), carried_rating, wa["id"],
         note[:200], _now())).lastrowid
    conn.execute("UPDATE lineages SET current_agent_id=? WHERE id=?",
                 (new_id, la["lineage_id"]))
    return note[:200]


def _autonomous_mutation(loser_cfg, winner_trades) -> str:
    """Rewrite the loser's carry-forward notes by learning from the winner's
    trades (LLM if available, else a plain summary)."""
    try:
        from .llm import DEFAULT_MODEL, _client, _extra_for, groq_available
        if groq_available() and winner_trades:
            import json as _j
            prompt = {"winner_trades": winner_trades[:40],
                      "your_old_notes": loser_cfg.get("notes", ""),
                      "instruction": "You lost. Study the winner's trades and write "
                      "concise trading notes for your next attempt — learn from them "
                      "but don't blindly copy. <=400 chars. Return JSON {\"notes\":\"...\"}."}
            r = _client().chat.completions.create(
                model=DEFAULT_MODEL, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You evolve traders. JSON only."},
                          {"role": "user", "content": _j.dumps(prompt)}],
                temperature=0.8, **_extra_for(DEFAULT_MODEL))
            return "LLM: " + str(_j.loads(r.choices[0].message.content).get("notes", ""))[:380]
    except Exception:
        pass
    syms = ", ".join(sorted({t["symbol"] for t in winner_trades})[:6]) or "nothing"
    return f"Winner traded {syms}. Rethink entries and risk."


def _recent_messages(conn, round_id, agent_id, limit=4) -> list[str]:
    """The RIVAL's recent messages (not this agent's own), oldest-first."""
    rows = conn.execute(
        "SELECT m.message, l.name FROM agent_messages m "
        "JOIN agents a ON a.id=m.agent_id JOIN lineages l ON l.id=a.lineage_id "
        "WHERE m.round_id=? AND m.agent_id!=? ORDER BY m.id DESC LIMIT ?",
        (round_id, agent_id, limit)).fetchall()
    return [f"{r['name']}: {r['message']}" for r in reversed(rows)]


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

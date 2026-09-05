"""Read-only query helpers shared by the dashboard (and handy in a REPL).

Everything here returns plain dicts/lists so templates and any future JSON API
can consume them without touching sqlite3.Row semantics.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .config import DEFAULT


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def leaderboard(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(conn.execute(
        """SELECT l.id, l.name, l.wins, l.losses, l.cumulative_return_pct,
                  l.current_stake, a.generation, a.rating
           FROM lineages l JOIN agents a ON a.id = l.current_agent_id
           ORDER BY a.rating DESC"""
    ))
    for i, r in enumerate(rows, 1):
        r["position"] = i
        r["rating_timeline"] = rating_timeline(conn, r["id"])
    return rows


def rating_timeline(conn: sqlite3.Connection, lineage_id: int) -> list[float]:
    """Reconstruct a lineage's ELO over rounds from result deltas.

    Ratings are stored per current generation (mutated in place), so the
    per-round history is rebuilt from round_results, starting at the base ELO.
    """
    results = conn.execute(
        """SELECT rr.winner_agent_id, rr.loser_agent_id, rr.rating_delta,
                  wa.lineage_id AS win_lin, la.lineage_id AS lose_lin
           FROM round_results rr
           JOIN rounds r ON r.id = rr.round_id
           JOIN agents wa ON wa.id = rr.winner_agent_id
           JOIN agents la ON la.id = rr.loser_agent_id
           ORDER BY r.round_number"""
    ).fetchall()
    timeline = [DEFAULT.elo_base]
    rating = DEFAULT.elo_base
    for row in results:
        if row["win_lin"] == lineage_id:
            rating += row["rating_delta"]
            timeline.append(round(rating, 1))
        elif row["lose_lin"] == lineage_id:
            rating -= row["rating_delta"]
            timeline.append(round(rating, 1))
    return timeline


def recent_rounds(conn: sqlite3.Connection, limit: int = 25) -> list[dict]:
    """Full 1st/2nd/3rd placement per round, not just winner-vs-loser —
    `round_results` only ever records the top and bottom finisher, which
    silently drops anyone in the middle once a round has 3+ participants
    (every round, now that the whole pool trades together)."""
    rounds = _rows(conn.execute(
        """SELECT r.id, r.round_number, r.length_days, r.goal_pct, r.status,
                  rr.resolution_reason
           FROM rounds r
           LEFT JOIN round_results rr ON rr.round_id = r.id
           ORDER BY r.round_number DESC LIMIT ?""",
        (limit,),
    ))
    if not rounds:
        return rounds
    ids = tuple(r["id"] for r in rounds)
    placeholders = ",".join("?" * len(ids))
    by_round: dict[int, list] = {}
    for row in _rows(conn.execute(
        f"""SELECT rr.round_id, rr.rank, rr.return_pct, l.name AS lineage,
                   l.id AS lineage_id
            FROM round_rankings rr
            JOIN agents a ON a.id = rr.agent_id
            JOIN lineages l ON l.id = a.lineage_id
            WHERE rr.round_id IN ({placeholders})
            ORDER BY rr.round_id, rr.rank""",
        ids,
    )):
        by_round.setdefault(row["round_id"], []).append(row)
    for r in rounds:
        r["rankings"] = by_round.get(r["id"], [])
    return rounds


def round_detail(conn: sqlite3.Connection, round_id: int) -> dict | None:
    rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
    if not rnd:
        return None
    rnd = dict(rnd)
    rnd["time_based"] = not rnd["length_days"]  # 0 => live/replay, not a real day count
    states = _rows(conn.execute(
        """SELECT rs.*, l.name AS lineage, l.id AS lineage_id,
                  a.generation, a.rating
           FROM round_states rs
           JOIN agents a ON a.id = rs.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE rs.round_id=? ORDER BY rs.final_return_pct DESC""",
        (round_id,),
    ))
    trades = _rows(conn.execute(
        """SELECT t.*, l.name AS lineage
           FROM trades t
           JOIN agents a ON a.id = t.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE t.round_id=? ORDER BY t.id""",
        (round_id,),
    ))
    result = conn.execute(
        """SELECT rr.*, wl.name AS winner, ll.name AS loser
           FROM round_results rr
           LEFT JOIN agents wa ON wa.id = rr.winner_agent_id
           LEFT JOIN agents la ON la.id = rr.loser_agent_id
           LEFT JOIN lineages wl ON wl.id = wa.lineage_id
           LEFT JOIN lineages ll ON ll.id = la.lineage_id
           WHERE rr.round_id=?""",
        (round_id,),
    ).fetchone()
    rankings = _rows(conn.execute(
        """SELECT rr.rank, rr.return_pct, rr.note, rr.stake_mult, rr.rating_delta,
                  l.name AS lineage, l.id AS lineage_id
           FROM round_rankings rr
           JOIN agents a ON a.id = rr.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE rr.round_id=? ORDER BY rr.rank""",
        (round_id,),
    ))
    return {
        "round": rnd,
        "states": states,
        "trades": trades,
        "result": dict(result) if result else None,
        "rankings": rankings,  # full N-way placement; empty for old pre-feature rounds
    }


def lineage_detail(conn: sqlite3.Connection, lineage_id: int) -> dict | None:
    lin = conn.execute("SELECT * FROM lineages WHERE id=?", (lineage_id,)).fetchone()
    if not lin:
        return None
    lin = dict(lin)
    generations = _rows(conn.execute(
        """SELECT id, generation, rating, strategy_config, activity_profile,
                  mutation_note, seeded_from_trade_agent_id, created_at
           FROM agents WHERE lineage_id=? ORDER BY generation DESC""",
        (lineage_id,),
    ))
    for g in generations:
        try:
            g["config_pretty"] = json.dumps(json.loads(g["strategy_config"]), indent=2)
        except Exception:
            g["config_pretty"] = g["strategy_config"]
    return {
        "lineage": lin,
        "generations": generations,
        "rating_timeline": rating_timeline(conn, lineage_id),
        "current_agent_id": lin["current_agent_id"],
    }


def _fmt_duration(start_iso: str, end_iso: str) -> str | None:
    """Wall-clock duration between two forward._now()-style UTC ISO
    timestamps, as a compact "Xd Yh" / "Yh Zm" / "Zm" / "Zs" string. Used for
    "how long did the latest round actually take" — length_days alone reads
    as "—" for every live/replay session (length_days=0 is their "time-based,
    not a real day count" sentinel), which is most rounds in practice, so a
    bare day-count is the wrong unit almost all the time."""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except (TypeError, ValueError):
        return None
    secs = (end - start).total_seconds()
    if secs < 0:
        return None
    secs = int(secs)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def arena_summary(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) c FROM rounds").fetchone()["c"]
    resolved = conn.execute(
        "SELECT COUNT(*) c FROM rounds WHERE status='resolved'").fetchone()["c"]
    trades = conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"]
    lineages = conn.execute("SELECT COUNT(*) c FROM lineages").fetchone()["c"]
    # Actual wall-clock time the latest RESOLVED round took, from its own
    # start to its own resolution — works the same for a 6-minute replay
    # session and a real 7-day forward round, unlike length_days (which is
    # 0, a sentinel, for every live/replay session).
    last_round = conn.execute(
        """SELECT r.created_at AS started, rr.created_at AS resolved
           FROM rounds r JOIN round_results rr ON rr.round_id = r.id
           WHERE r.status='resolved'
           ORDER BY r.round_number DESC LIMIT 1"""
    ).fetchone()
    duration = (_fmt_duration(last_round["started"], last_round["resolved"])
               if last_round else None)
    return {
        "total_rounds": total,
        "resolved_rounds": resolved,
        "total_trades": trades,
        "lineages": lineages,
        "last_round_duration": duration,
    }


# Distinct lineage accent colours (deliberately avoiding the semantic
# green/red used for up/down returns).
LINEAGE_COLORS = ["#8C7BFB", "#3FD0E6", "#F5B841", "#F98BB0",
                  "#A6E15A", "#F0A054", "#5AD1A8", "#C77DFF"]


def lineage_color(lineage_id: int) -> str:
    return LINEAGE_COLORS[(int(lineage_id) - 1) % len(LINEAGE_COLORS)]


def live_status(conn: sqlite3.Connection) -> dict | None:
    r = conn.execute("SELECT * FROM rounds WHERE status='live' "
                     "ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None
    done = conn.execute("SELECT COUNT(*) n FROM live_days WHERE round_id=?",
                        (r["id"],)).fetchone()["n"]
    return {"round_id": r["id"], "round_number": r["round_number"],
            "days_done": done, "length_days": r["length_days"],
            "time_based": not r["length_days"],  # 0 => live/replay, no fixed length
            "goal_pct": r["goal_pct"]}


def live_view(conn: sqlite3.Connection) -> dict | None:
    """The active (or most recent) live session — agents, positions, banter."""
    r = conn.execute("SELECT * FROM rounds WHERE status IN ('live','ended') "
                     "ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None
    agents = _rows(conn.execute(
        """SELECT rs.agent_id, rs.starting_capital, rs.current_capital, rs.holdings,
                  rs.final_return_pct, rs.status, rs.trade_count,
                  l.name, l.id AS lineage_id
           FROM round_states rs JOIN agents a ON a.id = rs.agent_id
           JOIN lineages l ON l.id = a.lineage_id WHERE rs.round_id=?""",
        (r["id"],)))
    for a in agents:
        try:
            a["holdings"] = json.loads(a["holdings"])
        except Exception:
            a["holdings"] = {}
        ret = a["final_return_pct"] or 0.0
        a["value"] = round(a["starting_capital"] * (1 + ret / 100), 2)
        a["ret"] = ret
    agents.sort(key=lambda x: x["ret"], reverse=True)
    messages = _rows(conn.execute(
        """SELECT m.ts, m.message, m.kind, l.name, l.id AS lineage_id
           FROM agent_messages m JOIN agents a ON a.id = m.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE m.round_id=? ORDER BY m.id""", (r["id"],)))
    return {"round": dict(r), "agents": agents, "messages": messages,
            "is_live": r["status"] == "live"}


def _ts_seconds(ts: str):
    """Best-effort seconds-of-day from either 'HH:MM:SS' or an ISO timestamp."""
    try:
        if "T" in ts or (len(ts) > 10 and "-" in ts):
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace("Z", ""))
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        h, m, s = ts.split(":")[:3]
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


def _fmt_hold(a, b):
    if a is None or b is None:
        return "—"
    secs = b - a
    if secs < 0:
        secs += 86400
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


def trade_analysis(conn: sqlite3.Connection, round_id: int) -> dict:
    """FIFO-match fills into completed round-trips (buy→sell with P&L + holding
    time) plus the still-open positions. Answers 'what did they buy, at what
    price, sell price, how long held'."""
    from collections import defaultdict, deque
    rows = _rows(conn.execute(
        """SELECT t.*, l.name FROM trades t JOIN agents a ON a.id = t.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE t.round_id=? ORDER BY t.id""", (round_id,)))
    lots = defaultdict(deque)   # (agent_id, symbol) -> open buy lots
    round_trips, names = [], {}
    for t in rows:
        names[t["agent_id"]] = t["name"]
        key = (t["agent_id"], t["symbol"])
        if t["side"] == "buy":
            lots[key].append({"qty": t["qty"], "price": t["price"], "ts": t["ts"],
                              "reason": t["reason"] or ""})
        else:  # sell — match FIFO
            qty = t["qty"]
            while qty > 1e-9 and lots[key]:
                lot = lots[key][0]
                m = min(qty, lot["qty"])
                pnl_pct = (t["price"] / lot["price"] - 1) * 100
                round_trips.append({
                    "name": t["name"], "symbol": t["symbol"], "qty": m,
                    "buy_price": lot["price"], "sell_price": t["price"],
                    "buy_ts": lot["ts"], "sell_ts": t["ts"],
                    "hold": _fmt_hold(_ts_seconds(lot["ts"]), _ts_seconds(t["ts"])),
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_amount": round((t["price"] - lot["price"]) * m, 2),
                    "buy_reason": lot["reason"], "sell_reason": t["reason"] or "",
                })
                lot["qty"] -= m
                qty -= m
                if lot["qty"] <= 1e-9:
                    lots[key].popleft()

    open_positions = []
    now = _ts_seconds(__import__("datetime").datetime.now().strftime("%H:%M:%S"))
    for (agent_id, symbol), q in lots.items():
        total_qty = sum(l["qty"] for l in q)
        if total_qty <= 1e-9:
            continue
        cost = sum(l["qty"] * l["price"] for l in q)
        first_ts = q[0]["ts"]
        open_positions.append({
            "name": names.get(agent_id, "?"), "symbol": symbol,
            "qty": total_qty, "avg_price": round(cost / total_qty, 2),
            "entry_ts": first_ts, "held": _fmt_hold(_ts_seconds(first_ts), now),
            "reason": q[0].get("reason", ""),
        })
    return {"round_trips": round_trips, "open_positions": open_positions,
            "fills": rows}


def latest_head_to_head(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT id FROM rounds WHERE status='resolved' "
        "ORDER BY round_number DESC LIMIT 1"
    ).fetchone()
    return round_detail(conn, row["id"]) if row else None


def audit_log(conn: sqlite3.Connection, round_id: int, limit: int = 300) -> list[dict]:
    """Every recorded decision turn (debug mode only) for a round: thoughts,
    which research tools were used, what came back, and the final action if
    that turn was the decision's last. Newest first."""
    rows = _rows(conn.execute(
        """SELECT a.*, l.name, l.id AS lineage_id
           FROM agent_audit a
           JOIN agents ag ON ag.id = a.agent_id
           JOIN lineages l ON l.id = ag.lineage_id
           WHERE a.round_id=? ORDER BY a.id DESC LIMIT ?""",
        (round_id, limit)))
    for r in rows:
        for field in ("research_requested", "research_results", "orders"):
            if r.get(field):
                try:
                    r[field] = json.loads(r[field])
                except (TypeError, ValueError):
                    pass
        r["tools_used"] = [k for k in ("scan", "history", "fundamentals")
                           if isinstance(r.get("research_requested"), dict)
                           and r["research_requested"].get(k)]
    return rows


def has_audit_log(conn: sqlite3.Connection, round_id: int) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM agent_audit WHERE round_id=? LIMIT 1", (round_id,)
    ).fetchone())


def guidelines_overview(conn: sqlite3.Connection) -> dict:
    active = _rows(conn.execute(
        "SELECT * FROM guidelines WHERE status='active' ORDER BY id"))
    proposals = _rows(conn.execute(
        """SELECT p.id, p.kind, p.practice, p.proposed_text, p.resolution,
                  SUM(v.vote='agree') AS agree, COUNT(v.id) AS total
           FROM guideline_proposals p
           LEFT JOIN guideline_votes v ON v.proposal_id = p.id
           GROUP BY p.id ORDER BY p.id DESC LIMIT 8"""))
    return {"active": active, "proposals": proposals}


def sparkline(values: list[float], width: int = 120, height: int = 28) -> str:
    """Return an inline SVG polyline — no JS, works offline and in any theme."""
    if not values or len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = i / (n - 1) * (width - 4) + 2
        y = height - 2 - (v - lo) / span * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    up = values[-1] >= values[0]
    color = "var(--alpha)" if up else "var(--liq)"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'class="spark" preserveAspectRatio="none">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linejoin="round"/></svg>'
    )

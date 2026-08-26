"""Read-only query helpers shared by the dashboard (and handy in a REPL).

Everything here returns plain dicts/lists so templates and any future JSON API
can consume them without touching sqlite3.Row semantics.
"""
from __future__ import annotations

import json
import sqlite3

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
    return _rows(conn.execute(
        """SELECT r.id, r.round_number, r.length_days, r.goal_pct, r.status,
                  rr.resolution_reason, rr.winner_return_pct, rr.loser_return_pct,
                  rr.rating_delta,
                  wl.name AS winner, ll.name AS loser
           FROM rounds r
           LEFT JOIN round_results rr ON rr.round_id = r.id
           LEFT JOIN agents wa ON wa.id = rr.winner_agent_id
           LEFT JOIN agents la ON la.id = rr.loser_agent_id
           LEFT JOIN lineages wl ON wl.id = wa.lineage_id
           LEFT JOIN lineages ll ON ll.id = la.lineage_id
           ORDER BY r.round_number DESC LIMIT ?""",
        (limit,),
    ))


def round_detail(conn: sqlite3.Connection, round_id: int) -> dict | None:
    rnd = conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
    if not rnd:
        return None
    rnd = dict(rnd)
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
    return {
        "round": rnd,
        "states": states,
        "trades": trades,
        "result": dict(result) if result else None,
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


def arena_summary(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) c FROM rounds").fetchone()["c"]
    resolved = conn.execute(
        "SELECT COUNT(*) c FROM rounds WHERE status='resolved'").fetchone()["c"]
    trades = conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"]
    lineages = conn.execute("SELECT COUNT(*) c FROM lineages").fetchone()["c"]
    last_len = conn.execute(
        "SELECT length_days FROM rounds ORDER BY round_number DESC LIMIT 1"
    ).fetchone()
    return {
        "total_rounds": total,
        "resolved_rounds": resolved,
        "total_trades": trades,
        "lineages": lineages,
        "last_round_days": last_len["length_days"] if last_len else None,
    }


def guidelines_overview(conn: sqlite3.Connection) -> dict:
    active = _rows(conn.execute(
        "SELECT * FROM guidelines WHERE status='active' ORDER BY id"))
    proposals = _rows(conn.execute(
        """SELECT p.id, p.kind, p.proposed_text, p.resolution,
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

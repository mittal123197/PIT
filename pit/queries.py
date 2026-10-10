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
    stats = lineage_stats(conn)
    for i, r in enumerate(rows, 1):
        r["position"] = i
        r["rating_timeline"] = rating_timeline(conn, r["id"])
        st = stats.get(r["id"], {"played": 0, "podium": {}, "avg": None, "best": None})
        r["played"], r["podium"] = st["played"], st["podium"]
        r["avg_return"], r["best_return"] = st["avg"], st["best"]
    return rows


def rating_timeline(conn: sqlite3.Connection, lineage_id: int) -> list[float]:
    """A lineage's ELO after every round it played, from round_rankings (each
    participant's own rating_delta). The old version rebuilt it from the
    2-agent round_results table — in a 3-way round it skipped the middle
    finisher entirely and gave the loser the winner's delta mirrored."""
    rows = conn.execute(
        """SELECT rr.rating_delta FROM round_rankings rr
           JOIN rounds r ON r.id = rr.round_id
           JOIN agents a ON a.id = rr.agent_id
           WHERE a.lineage_id=? ORDER BY r.round_number""", (lineage_id,)).fetchall()
    timeline = [DEFAULT.elo_base]
    for row in rows:
        timeline.append(round(timeline[-1] + (row["rating_delta"] or 0), 1))
    return timeline


def lineage_stats(conn: sqlite3.Connection) -> dict[int, dict]:
    """Per lineage: rounds played, podium finishes, average / best round return."""
    out: dict[int, dict] = {}
    for r in conn.execute(
            """SELECT a.lineage_id, rr.rank, rr.return_pct FROM round_rankings rr
               JOIN agents a ON a.id = rr.agent_id""").fetchall():
        d = out.setdefault(r["lineage_id"], {"played": 0, "podium": {}, "rets": []})
        d["played"] += 1
        d["podium"][r["rank"]] = d["podium"].get(r["rank"], 0) + 1
        d["rets"].append(r["return_pct"] or 0.0)
    for d in out.values():
        d["avg"] = round(sum(d["rets"]) / len(d["rets"]), 2) if d["rets"] else None
        d["best"] = round(max(d["rets"]), 2) if d["rets"] else None
    return out


def elo_chart(conn: sqlite3.Connection) -> str:
    series = []
    for r in conn.execute("SELECT id, name FROM lineages ORDER BY id").fetchall():
        tl = rating_timeline(conn, r["id"])
        series.append({"name": r["name"], "color": lineage_color(r["id"]),
                       "points": [(i, v) for i, v in enumerate(tl)]})
    if max((len(s["points"]) for s in series), default=0) < 2:
        return ""
    n = max(len(s["points"]) for s in series) - 1
    ticks = list(range(0, n + 1)) if n <= 14 else [0, n // 2, n]
    return line_chart(series, y_fmt=lambda v: f"{v:.0f}", x_fmt=lambda x, x0: f"R{int(x)}" if x else "start",
                      end_fmt=lambda v: f"{v:.0f}", height=210, zero_line=DEFAULT.elo_base,
                      aria="ELO rating after each round, per agent", x_ticks=ticks)


def recent_rounds(conn: sqlite3.Connection, limit: int = 25) -> list[dict]:
    """Full 1st/2nd/3rd placement per round, not just winner-vs-loser —
    `round_results` only ever records the top and bottom finisher, which
    silently drops anyone in the middle once a round has 3+ participants
    (every round, now that the whole pool trades together)."""
    rounds = _rows(conn.execute(
        """SELECT r.id, r.round_number, r.length_days, r.goal_pct, r.status,
                  r.market, r.created_at, rr.resolution_reason,
                  rr.created_at AS resolved_at
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
        r["duration"] = (_fmt_duration(r["created_at"], r["resolved_at"])
                         if r.get("resolved_at") else None)
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
    # per-position live value + unrealized P&L from the last mark (no network)
    st_extra = {r["agent_id"]: r for r in _rows(conn.execute(
        "SELECT agent_id, cost_basis, mark_prices FROM round_states WHERE round_id=?",
        (r["id"],)))}
    for a in agents:
        ex = st_extra.get(a["agent_id"], {})
        cost = _json(ex.get("cost_basis"))
        marks = _json(ex.get("mark_prices"))
        pos = []
        for sym, qty in a["holdings"].items():
            px = marks.get(sym) or cost.get(sym)
            c = cost.get(sym)
            pos.append({"symbol": sym, "qty": qty, "price": px, "cost": c,
                        "value": round(qty * (px or 0), 2),
                        "pnl_pct": round((px / c - 1) * 100, 2) if px and c else None})
        pos.sort(key=lambda p: p["value"], reverse=True)
        a["positions"] = pos
        total = a["value"] or 1.0
        invested = sum(p["value"] for p in pos)
        unreal = sum(p["value"] - p["qty"] * (p["cost"] or p["price"] or 0) for p in pos)
        a["exposure_pct"] = round(invested / total * 100) if total else 0
        a["unrealized"] = round(unreal, 2)
        a["realized"] = round((a["value"] - a["starting_capital"]) - unreal, 2)
        a["alloc"] = ([{"label": p["symbol"], "pct": max(0.0, p["value"] / total * 100)}
                       for p in pos] +
                      [{"label": "cash", "pct": max(0.0, a["current_capital"] / total * 100)}])
    messages = _rows(conn.execute(
        """SELECT m.ts, m.message, m.kind, l.name, l.id AS lineage_id
           FROM agent_messages m JOIN agents a ON a.id = m.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE m.round_id=? ORDER BY m.id""", (r["id"],)))
    rd = dict(r)
    timing = _session_timing(rd)
    series = race_series(conn, r["id"])
    dd = _max_drawdowns(conn, r["id"])
    for a in agents:
        a["max_dd"] = dd.get(a["agent_id"])
    bench = next((s for s in series if s.get("dash")), None)
    bench_now = round(bench["points"][-1][1], 2) if bench else None
    return {"round": rd, "agents": agents, "messages": messages,
            "is_live": r["status"] == "live", "timing": timing,
            "race_svg": race_chart(series), "risk": _risk_bands(rd),
            "bench_name": bench["name"] if bench else None, "bench_now": bench_now}


def _max_drawdowns(conn, round_id: int) -> dict[int, float]:
    """Worst peak-to-trough fall of each agent's account this round, in %."""
    out: dict[int, tuple[float, float]] = {}
    for r in conn.execute("SELECT agent_id, return_pct FROM round_marks "
                          "WHERE round_id=? ORDER BY id", (round_id,)).fetchall():
        eq = 1 + (r["return_pct"] or 0) / 100
        peak, worst = out.get(r["agent_id"], (eq, 0.0))
        peak = max(peak, eq)
        out[r["agent_id"]] = (peak, min(worst, (eq / peak - 1) * 100))
    return {k: round(v[1], 2) for k, v in out.items()}


def _json(v) -> dict:
    try:
        return json.loads(v) if v else {}
    except Exception:
        return {}


def _parse_local(ts):
    from datetime import datetime as _dt
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            d = _dt.strptime(ts, fmt)
            return d.astimezone().replace(tzinfo=None) if d.tzinfo else d
        except (TypeError, ValueError):
            continue
    return None


def _session_timing(rd: dict) -> dict | None:
    """Elapsed / remaining for a live session and the next agent wake-up."""
    from datetime import datetime as _dt, timedelta
    start = _parse_local(rd.get("created_at"))
    end = _parse_local(rd.get("ends_at"))
    if not start or not end or end <= start:
        return None
    now = _dt.now()
    total = (end - start).total_seconds()
    done = min(max((now - start).total_seconds(), 0), total)
    out = {"pct": round(done / total * 100, 1),
           "remaining": _fmt_secs(max(0, (end - now).total_seconds())),
           "elapsed": _fmt_secs(done), "next_wake": None}
    last = _parse_local(rd.get("last_tick_at"))
    if last and rd.get("interval_s"):
        nxt = last + timedelta(seconds=rd["interval_s"])
        secs = (nxt - now).total_seconds()
        out["next_wake"] = "deciding now" if secs <= 0 else f"in {_fmt_secs(secs)}"
    return out


def _fmt_secs(secs: float) -> str:
    secs = int(secs)
    return f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"


def _risk_bands(rd: dict) -> dict:
    return {"stop": DEFAULT.stop_loss_pct_for(rd.get("length_days") or 0),
            "goal": rd.get("goal_pct"),
            "position_stop": DEFAULT.position_stop_loss_pct}


def race_series(conn, round_id: int) -> list[dict]:
    rows = _rows(conn.execute(
        """SELECT m.agent_id, m.ts, m.return_pct, l.name, l.id AS lineage_id
           FROM round_marks m JOIN agents a ON a.id = m.agent_id
           JOIN lineages l ON l.id = a.lineage_id
           WHERE m.round_id=? ORDER BY m.id""", (round_id,)))
    by: dict[int, dict] = {}
    for r in rows:
        t = _parse_local(r["ts"])
        if not t:
            continue
        s = by.setdefault(r["agent_id"], {"name": r["name"],
                                          "color": lineage_color(r["lineage_id"]),
                                          "points": []})
        s["points"].append((t.timestamp(), r["return_pct"]))
    # buy/sell markers on each agent's line (forced closes excluded — not a decision)
    from .forward import is_forced
    for r in _rows(conn.execute(
            "SELECT agent_id, ts, side, symbol, reason FROM trades WHERE round_id=? ORDER BY id",
            (round_id,))):
        t = _parse_local(r["ts"])
        if t and r["agent_id"] in by and not is_forced(r["reason"]):
            by[r["agent_id"]].setdefault("marks", []).append(
                (t.timestamp(), r["side"], r["symbol"]))
    series = list(by.values())
    bench = _rows(conn.execute(
        "SELECT ts, symbol, price FROM benchmark_marks WHERE round_id=? ORDER BY id",
        (round_id,)))
    if len(bench) >= 2 and bench[0]["price"]:
        p0 = bench[0]["price"]
        pts = [(_parse_local(b["ts"]).timestamp(), (b["price"] / p0 - 1) * 100)
               for b in bench if _parse_local(b["ts"])]
        series.append({"name": bench[0]["symbol"].replace("-USD", "") + " hold",
                       "color": "#8b9099", "dash": True, "points": pts})
    return series


def race_chart(series: list[dict], width: int = 860, height: int = 230) -> str:
    """Every agent's return % over a session (x = elapsed time)."""
    return line_chart(series, y_fmt=lambda v: f"{v:+.2f}%",
                      x_fmt=lambda t, t0: _fmt_secs(t - t0) if t > t0 else "start",
                      end_fmt=lambda v: f"{v:+.2f}%", width=width, height=height,
                      zero_line=0.0, aria="Return over time for each agent")


def line_chart(series: list[dict], y_fmt, x_fmt, end_fmt, width: int = 860,
               height: int = 230, zero_line: float | None = 0.0,
               aria: str = "chart", x_ticks: list | None = None) -> str:
    """Server-rendered multi-line SVG: one line per series in its colour, a
    highlighted baseline, end-of-line labels nudged apart. No JS."""
    pts = [p for s in series for p in s["points"]]
    if len({p[0] for p in pts}) < 2:      # need at least two moments in time
        return ""
    t0, t1 = min(p[0] for p in pts), max(p[0] for p in pts)
    lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
    if zero_line is not None:
        lo, hi = min(lo, zero_line), max(hi, zero_line)
    pad = max((hi - lo) * 0.15, 0.05)
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 52, 118, 14, 26
    w, h = width - L - R, height - T - B
    X = lambda t: L + (t - t0) / ((t1 - t0) or 1) * w
    Y = lambda v: T + (hi - v) / (hi - lo) * h
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" '
           f'role="img" aria-label="{aria}" style="display:block;max-height:{height + 30}px;">']
    step = _nice_step((hi - lo) / 4)
    v = (lo // step) * step
    while v <= hi:
        y = Y(v)
        if T - 1 <= y <= T + h + 1:
            out.append(f'<line x1="{L}" x2="{L + w}" y1="{y:.1f}" y2="{y:.1f}" '
                       'stroke="#23262e" stroke-width="1" stroke-dasharray="2 4"/>')
            out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" '
                       f'font-size="10" fill="#6C7076" font-family="IBM Plex Mono">{y_fmt(v)}</text>')
        v += step
    if zero_line is not None:
        y = Y(zero_line)
        out.append(f'<line x1="{L}" x2="{L + w}" y1="{y:.1f}" y2="{y:.1f}" '
                   'stroke="#6C7076" stroke-width="1"/>')
    for t in (x_ticks if x_ticks is not None else [t0 + (t1 - t0) * f for f in (0, 0.5, 1)]):
        out.append(f'<text x="{X(t):.1f}" y="{height - 6}" text-anchor="middle" '
                   f'font-size="10" fill="#6C7076" font-family="IBM Plex Mono">{x_fmt(t, t0)}</text>')
    ends = []
    for s in series:
        p = s["points"]
        if not p:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(t):.1f},{Y(v):.1f}" for i, (t, v) in enumerate(p))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" '
                   f'stroke-width="{1.6 if s.get("dash") else 2.2}"{dash} '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
        for (mt, side, sym) in s.get("marks", []):
            if not (t0 <= mt <= t1):
                continue
            mv = _interp(p, mt)
            cx, cy = X(mt), Y(mv)
            tri = (f"M{cx:.1f},{cy - 9:.1f} l-5,8 h10 z" if side == "buy"
                   else f"M{cx:.1f},{cy + 9:.1f} l-5,-8 h10 z")
            fill = "#3ddc97" if side == "buy" else "#f0563c"
            out.append(f'<path d="{tri}" fill="{fill}" stroke="#0e1014" stroke-width="1">'
                       f'<title>{side} {sym}</title></path>')
        if len(p) <= 30 and not s.get("marks"):   # few points (e.g. one per round): mark each one
            out += [f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{s["color"]}"/>'
                    for t, v in p]
        ends.append([Y(p[-1][1]), s, p[-1]])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, s, (t, v) in ends:
        out.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{s["color"]}"/>')
        out.append(f'<text x="{L + w + 10}" y="{y + 4:.1f}" font-size="11.5" '
                   f'fill="{s["color"]}" font-family="IBM Plex Mono" font-weight="600">'
                   f'{s["name"]} {end_fmt(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def _interp(points, t) -> float:
    """y of a polyline at x=t (clamped to the ends)."""
    if t <= points[0][0]:
        return points[0][1]
    for (t1, v1), (t2, v2) in zip(points, points[1:]):
        if t1 <= t <= t2:
            return v1 if t2 == t1 else v1 + (v2 - v1) * (t - t1) / (t2 - t1)
    return points[-1][1]


def _nice_step(raw: float) -> float:
    import math
    if raw <= 0:
        return 0.1
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


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
    marks = {r["agent_id"]: _json(r["mark_prices"]) for r in _rows(conn.execute(
        "SELECT agent_id, mark_prices FROM round_states WHERE round_id=?", (round_id,)))}
    now = _ts_seconds(__import__("datetime").datetime.now().strftime("%H:%M:%S"))
    for (agent_id, symbol), q in lots.items():
        total_qty = sum(l["qty"] for l in q)
        if total_qty <= 1e-9:
            continue
        cost = sum(l["qty"] * l["price"] for l in q)
        first_ts = q[0]["ts"]
        open_positions.append({
            "name": names.get(agent_id, "?"), "symbol": symbol,
            "qty": total_qty, "avg_price": round(cost / total_qty, 8),
            "now_price": marks.get(agent_id, {}).get(symbol),
            "pnl_pct": (round((marks[agent_id][symbol] / (cost / total_qty) - 1) * 100, 2)
                        if marks.get(agent_id, {}).get(symbol) else None),
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
    retired = _rows(conn.execute(
        "SELECT * FROM guidelines WHERE status='retired' ORDER BY id DESC LIMIT 6"))
    status_by_text = {g["text"]: g["status"] for g in _rows(conn.execute(
        "SELECT text, status FROM guidelines"))}
    proposals = _rows(conn.execute(
        """SELECT p.id, p.kind, p.practice, p.proposed_text, p.resolution,
                  p.proposer, SUM(v.vote='agree') AS agree, COUNT(v.id) AS total
           FROM guideline_proposals p
           LEFT JOIN guideline_votes v ON v.proposal_id = p.id
           GROUP BY p.id ORDER BY p.id DESC LIMIT 10"""))
    for p in proposals:
        # an adopted rule can be retired later — say so instead of a bare "adopted"
        p["now"] = status_by_text.get(p["proposed_text"]) if p["kind"] == "add" else None
    return {"active": active, "retired": retired, "proposals": proposals}


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

"""Flask dashboard — a read-only window onto the arena's SQLite state.

    python3 -m pit.web            # http://127.0.0.1:5001

Read-only by design: it never runs a round or mutates state, so it's safe to
point at a live DB while the orchestrator writes to it. Sparklines are
server-rendered inline SVG, so the page needs no JS and works offline.
"""
from __future__ import annotations

import os

from flask import Flask, abort, render_template
from markupsafe import Markup

from . import db as dbm
from . import queries as q

app = Flask(__name__)


def _conn():
    conn = dbm.connect()
    dbm.init_db(conn)
    return conn


@app.template_filter("pct")
def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%"


@app.template_filter("money")
def _money(value) -> str:
    if value is None:
        return "—"
    return f"₹{value:,.0f}"


@app.template_filter("spark")
def _spark(values) -> Markup:
    return Markup(q.sparkline(values or []))


@app.route("/")
def index():
    conn = _conn()
    return render_template(
        "index.html",
        summary=q.arena_summary(conn),
        board=q.leaderboard(conn),
        rounds=q.recent_rounds(conn),
    )


@app.route("/round/<int:round_id>")
def round_view(round_id: int):
    conn = _conn()
    data = q.round_detail(conn, round_id)
    if not data:
        abort(404)
    return render_template("round.html", **data)


@app.route("/lineage/<int:lineage_id>")
def lineage_view(lineage_id: int):
    conn = _conn()
    data = q.lineage_detail(conn, lineage_id)
    if not data:
        abort(404)
    return render_template("lineage.html", **data)


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


def main():
    port = int(os.getenv("PIT_WEB_PORT", "5001"))
    host = os.getenv("PIT_WEB_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=os.getenv("PIT_WEB_DEBUG") == "1")


if __name__ == "__main__":
    main()

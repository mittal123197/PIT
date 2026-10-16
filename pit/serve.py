"""Production entrypoint: serve the dashboard with waitress.

    python3 -m pit.serve        # dashboard on $PORT (default 8080)

Battles are driven separately (scripts/run_arena.sh or `python -m pit.cli
live`) against the same SQLite file; the dashboard only reads it. The old
always-on heuristic engine this used to start alongside it is gone — the
arena is the live three-agent battle now.
"""
from __future__ import annotations

import os

from .web import app


def main() -> None:
    from waitress import serve
    serve(app, host=os.getenv("PIT_WEB_HOST", "0.0.0.0"),
          port=int(os.getenv("PORT", "8080")))


if __name__ == "__main__":
    main()

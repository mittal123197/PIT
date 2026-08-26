"""Production entrypoint for the deployed container.

Starts the always-on arena loop in a background thread, then serves the
dashboard with waitress (a real WSGI server, unlike Flask's dev server). One
process, one SQLite file (WAL mode) — which is all a single Fly.io Machine on a
persistent volume needs.

    python3 -m pit.serve        # arena daemon + dashboard on $PORT (default 8080)

Set PIT_DAEMON=0 to serve the dashboard only (no auto-running rounds).
"""
from __future__ import annotations

import os
import threading

from .web import app


def _start_daemon() -> None:
    from .daemon import run_forever
    t = threading.Thread(target=run_forever, name="arena-daemon", daemon=True)
    t.start()


def main() -> None:
    if os.getenv("PIT_DAEMON", "1") not in ("0", "", "false"):
        _start_daemon()

    host = os.getenv("PIT_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("PIT_WEB_PORT", "8080")))
    try:
        from waitress import serve
        print(f"[serve] dashboard on http://{host}:{port}", flush=True)
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        print("[serve] waitress not installed, using Flask dev server", flush=True)
        app.run(host=host, port=port)


if __name__ == "__main__":
    main()

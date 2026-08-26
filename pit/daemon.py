"""The always-on arena loop — the deployed heartbeat.

Runs rounds forever with a configurable pause between them. Locally, rounds are
7 *simulated* days replayed in seconds; the pause (`PIT_ROUND_INTERVAL`) just
controls how often a new duel kicks off so the dashboard is watchable rather
than a blur. Seeds the two default lineages on first run if the DB is empty.

Run standalone (`python3 -m pit.daemon`) or let `pit.serve` start it in a
background thread alongside the dashboard.
"""
from __future__ import annotations

import os
import time

from . import db as dbm
from .config import DEFAULT
from .engine import Engine

SEED_LINEAGES = [
    ("CONTANGO", {"type": "momentum", "lookback": 5, "buy_threshold_pct": 1.0,
                  "sell_threshold_pct": -1.5, "position_frac": 0.25,
                  "heartbeat_minutes": 60}),
    ("BACKWARDATION", {"type": "mean_reversion", "lookback": 10, "band_pct": 2.0,
                       "position_frac": 0.25, "heartbeat_minutes": 120}),
]


def _ensure_seeded(eng: Engine) -> list[int]:
    ids = [r["id"] for r in eng.conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]
    if len(ids) >= 2:
        return ids
    for name, cfg in SEED_LINEAGES:
        eng.create_lineage(name, cfg)
    return [r["id"] for r in eng.conn.execute(
        "SELECT id FROM lineages ORDER BY id").fetchall()]


def run_forever() -> None:
    interval = int(os.getenv("PIT_ROUND_INTERVAL", "300"))  # seconds between rounds
    bars = int(os.getenv("PIT_BARS", "0")) or None
    use_llm = os.getenv("PIT_USE_LLM", "0") not in ("0", "", "false")

    conn = dbm.connect()
    dbm.init_db(conn)
    from .llm import mutate_with_llm
    from .policies import SimplePolicy

    def policy_factory(agent_row):
        if use_llm:
            from .llm import LLMPolicy, groq_available
            if groq_available():
                return LLMPolicy()
        return SimplePolicy()

    eng = Engine(conn, config=DEFAULT, policy_factory=policy_factory,
                 mutate_fn=mutate_with_llm, use_llm=use_llm)
    ids = _ensure_seeded(eng)

    print(f"[daemon] arena loop up — round every {interval}s, "
          f"llm={use_llm}, lineages={ids}", flush=True)
    while True:
        from .feeds import SyntheticFeed
        n = eng._next_round_number()

        def factory(length_days, _n=n):
            b = bars or DEFAULT.bars_for_days(length_days)
            return SyntheticFeed(DEFAULT.universe, seed=1000 + _n, bars=b,
                                 bar_minutes=DEFAULT.bar_minutes)
        try:
            out = eng.run_round(ids[0], ids[1], feed_factory=factory)
            print(f"[daemon] round {out.round_number}: {out.winner_lineage} "
                  f"beat {out.loser_lineage} ({out.reason})", flush=True)
        except Exception as exc:  # never let one bad round kill the loop
            print(f"[daemon] round error: {exc!r}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    run_forever()

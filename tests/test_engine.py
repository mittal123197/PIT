"""End-to-end integration: a full round must run, resolve, and mutate.

Uses an in-memory SQLite DB and the synthetic feed so it's fast, offline, and
deterministic. Proves the plumbing the plan's integration check calls for:
trades log, a winner is produced, and the loser's next generation is a distinct
(mutated) config — not an identical clone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import sqlite3  # noqa: E402

from pit import db as dbm  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402
from pit.engine import Engine  # noqa: E402
from pit.feeds import SyntheticFeed  # noqa: E402


def _fresh_engine():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    return Engine(conn, config=ArenaConfig())


def _seed(eng):
    a = eng.create_lineage("A", {"type": "momentum", "lookback": 5,
                                 "buy_threshold_pct": 0.5, "sell_threshold_pct": -0.5,
                                 "position_frac": 0.3, "heartbeat_minutes": 30})
    b = eng.create_lineage("B", {"type": "mean_reversion", "lookback": 8,
                                 "band_pct": 1.5, "position_frac": 0.3,
                                 "heartbeat_minutes": 60})
    return a, b


def test_round_runs_and_produces_a_winner():
    eng = _fresh_engine()
    a, b = _seed(eng)
    feed = SyntheticFeed(eng.config.universe, bars=150, seed=7)
    out = eng.run_round(a, b, feed=feed)
    assert out.winner_agent_id != out.loser_agent_id
    assert out.reason in ("return", "trade_count", "drawdown", "sudden_death")


def test_trades_are_logged():
    eng = _fresh_engine()
    a, b = _seed(eng)
    feed = SyntheticFeed(eng.config.universe, bars=150, seed=7)
    out = eng.run_round(a, b, feed=feed)
    n = eng.conn.execute(
        "SELECT COUNT(*) c FROM trades WHERE round_id=?", (out.round_id,)
    ).fetchone()["c"]
    assert n > 0, "expected at least one trade to be logged"


def test_loser_mutates_into_a_distinct_new_generation():
    eng = _fresh_engine()
    a, b = _seed(eng)
    # capture loser config BEFORE the round
    configs_before = {
        r["id"]: r["strategy_config"]
        for r in eng.conn.execute("SELECT id, strategy_config FROM agents")
    }
    feed = SyntheticFeed(eng.config.universe, bars=150, seed=7)
    out = eng.run_round(a, b, feed=feed)

    loser_lineage = eng.conn.execute(
        "SELECT lineage_id FROM agents WHERE id=?", (out.loser_agent_id,)
    ).fetchone()["lineage_id"]
    new_agent = eng.conn.execute(
        """SELECT * FROM agents WHERE lineage_id=? ORDER BY generation DESC LIMIT 1""",
        (loser_lineage,),
    ).fetchone()

    assert new_agent["generation"] == 2, "loser should spawn a gen-2 agent"
    assert new_agent["seeded_from_trade_agent_id"] == out.winner_agent_id
    # mutated, not cloned: config differs from the loser's previous config
    assert new_agent["strategy_config"] != configs_before[out.loser_agent_id]
    # and it must be valid JSON with the expected shape
    cfg = json.loads(new_agent["strategy_config"])
    assert "heartbeat_minutes" in cfg


def test_stakes_shift_after_a_round():
    eng = _fresh_engine()
    a, b = _seed(eng)
    feed = SyntheticFeed(eng.config.universe, bars=150, seed=7)
    out = eng.run_round(a, b, feed=feed)
    stakes = {
        r["name"]: r["current_stake"]
        for r in eng.conn.execute("SELECT name, current_stake FROM lineages")
    }
    base = eng.config.base_capital
    # one lineage should be up ~20%, the other down ~20%
    assert any(v > base for v in stakes.values())
    assert any(v < base for v in stakes.values())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

"""The stop-loss and take-profit hard constraints actually fire during a live
session — both in the LLM-decision tick and in the cheap between-decisions
mark-to-market — and freeze the right status ('liquidated' vs 'goal_hit').

No network/LLM calls needed: with empty holdings, return_pct is driven purely
by round_states.current_capital, so we can put an agent past either
threshold without touching market data or the decision loop at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3  # noqa: E402

from pit import db as dbm, forward, live  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    return conn


def _stub_decide(monkeypatch):
    """Whichever agent stays active this tick must not reach the real LLM —
    stub decide() to a no-op hold so these tests are fast, offline, and
    deterministic; only the hard-stop plumbing itself is under test here."""
    monkeypatch.setattr(live.autonomous, "decide",
                        lambda *a, **k: ([], "", "", []))


def _round_with_capital(conn, config, ronin_capital, viper_capital):
    """A fresh 7-day round with RONIN/VIPER's cash set directly (no holdings),
    so return_pct is exactly known without any price lookups."""
    rid = forward.start_round(conn, config, days=7)
    for name, cap in (("RONIN", ronin_capital), ("VIPER", viper_capital)):
        aid = conn.execute(
            "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
            "WHERE l.name=?", (name,)).fetchone()["id"]
        conn.execute("UPDATE round_states SET current_capital=? "
                     "WHERE round_id=? AND agent_id=?", (cap, rid, aid))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    return rnd


def test_tick_stops_out_an_agent_past_the_stop_loss(monkeypatch):
    _stub_decide(monkeypatch)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd = _round_with_capital(conn, config, ronin_capital=100_000 * 0.85,  # -15%
                              viper_capital=100_000)
    live._tick(conn, config, rnd, tick=1, verbose=False)
    st = dict(conn.execute(
        "SELECT rs.* FROM round_states rs JOIN agents a ON a.id=rs.agent_id "
        "JOIN lineages l ON l.id=a.lineage_id WHERE l.name='RONIN' AND rs.round_id=?",
        (rnd["id"],)).fetchone())
    assert st["status"] == "liquidated"


def test_tick_books_profit_for_an_agent_past_the_goal(monkeypatch):
    _stub_decide(monkeypatch)
    conn = _fresh_conn()
    config = ArenaConfig()
    goal = config.goal_pct_for(7)
    over_goal_capital = 100_000 * (1 + (goal + 5) / 100)
    rnd = _round_with_capital(conn, config, ronin_capital=over_goal_capital,
                              viper_capital=100_000)
    live._tick(conn, config, rnd, tick=1, verbose=False)
    st = dict(conn.execute(
        "SELECT rs.* FROM round_states rs JOIN agents a ON a.id=rs.agent_id "
        "JOIN lineages l ON l.id=a.lineage_id WHERE l.name='RONIN' AND rs.round_id=?",
        (rnd["id"],)).fetchone())
    assert st["status"] == "goal_hit"


def test_refresh_marks_also_enforces_both_bands(monkeypatch):
    _stub_decide(monkeypatch)
    conn = _fresh_conn()
    config = ArenaConfig()
    goal = config.goal_pct_for(7)
    rnd = _round_with_capital(
        conn, config,
        ronin_capital=100_000 * 0.80,                       # well past stop-loss
        viper_capital=100_000 * (1 + (goal + 10) / 100))    # well past goal
    live._refresh_marks(conn, config, rnd, verbose=False)
    statuses = {r["name"]: r["status"] for r in conn.execute(
        "SELECT l.name, rs.status FROM round_states rs "
        "JOIN agents a ON a.id=rs.agent_id JOIN lineages l ON l.id=a.lineage_id "
        "WHERE rs.round_id=?", (rnd["id"],))}
    assert statuses["RONIN"] == "liquidated"
    assert statuses["VIPER"] == "goal_hit"


def test_liquidation_freezes_the_agent_from_further_trading(monkeypatch):
    """Once stopped/booked, the tick loop must skip the agent entirely (no
    further decide() calls) — status != 'active' short-circuits the tick."""
    conn = _fresh_conn()
    config = ArenaConfig()
    # both past a hard band, so this test never reaches the real decide()
    # call (which would need network/LLM access) on either agent.
    goal = config.goal_pct_for(7)
    rnd = _round_with_capital(conn, config, ronin_capital=100_000 * 0.85,
                              viper_capital=100_000 * (1 + (goal + 10) / 100))
    live._tick(conn, config, rnd, tick=1, verbose=False)
    # a second tick must not error or change the frozen agent's capital
    live._tick(conn, config, rnd, tick=2, verbose=False)
    st = dict(conn.execute(
        "SELECT rs.* FROM round_states rs JOIN agents a ON a.id=rs.agent_id "
        "JOIN lineages l ON l.id=a.lineage_id WHERE l.name='RONIN' AND rs.round_id=?",
        (rnd["id"],)).fetchone())
    assert st["status"] == "liquidated"
    assert st["current_capital"] == 100_000 * 0.85


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

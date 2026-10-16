"""A "win" earned with ZERO trades — whether a deliberate all-cash hold or a
decision loop that silently failed (LLM/network error) — didn't beat the
rival's trading, it just avoided their loss. The stake bonus is capped well
below the normal win bonus, and the reflection note says so plainly instead
of fabricating a "what worked" story.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3  # noqa: E402

from pit import db as dbm, forward  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    return conn


def _agent_id(conn, name):
    return conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name=?", (name,)).fetchone()["id"]






def test_passive_win_note_flags_it_instead_of_fabricating_a_story(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id, viper_id = _agent_id(conn, "RONIN"), _agent_id(conn, "VIPER")
    conn.execute("UPDATE round_states SET current_capital=100000, trade_count=0 "
                 "WHERE round_id=? AND agent_id=?", (rid, ronin_id))
    conn.execute("UPDATE round_states SET current_capital=88000, status='liquidated' "
                 "WHERE round_id=? AND agent_id=?", (rid, viper_id))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    outcome = forward._resolve(conn, rnd, config, lambda t: 100.0)

    assert "zero trades" in outcome["winner_note"].lower()
    assert "wasn't earned" in outcome["winner_note"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

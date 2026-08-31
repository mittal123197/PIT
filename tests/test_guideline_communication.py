"""Guidelines now form from agents actually communicating about their own
experience (self-reflection notes shared as messages), not just win/loss
trade-count/drawdown statistics — and every guideline is now typed as a GOOD
practice (do this) or a BAD practice (avoid this), visible to every agent's
decision every turn.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3  # noqa: E402

from pit import db as dbm, forward, guidelines as gmod  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    return conn


def test_migration_adds_new_columns_to_a_preexisting_db(tmp_path):
    """A DB created before this feature (no `practice`/`kind` columns) must
    still work after init_db runs again — the point of the ALTER TABLE
    best-effort migration in db.py."""
    path = str(tmp_path / "old.db")
    # simulate a pre-existing DB using an OLDER schema (no new columns)
    old_conn = sqlite3.connect(path)
    old_conn.executescript("""
        CREATE TABLE guidelines (id INTEGER PRIMARY KEY, text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, retired_at TEXT);
        CREATE TABLE agent_messages (id INTEGER PRIMARY KEY, round_id INTEGER,
            agent_id INTEGER, ts TEXT NOT NULL, message TEXT NOT NULL);
    """)
    old_conn.commit()
    old_conn.close()

    conn = dbm.connect(path)
    dbm.init_db(conn)  # must not raise, and must add the missing columns
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(guidelines)")}
    assert "practice" in cols
    cols2 = {r["name"] for r in conn.execute("PRAGMA table_info(agent_messages)")}
    assert "kind" in cols2


def test_active_texts_are_labeled_do_or_avoid():
    conn = _fresh_conn()
    conn.execute("INSERT INTO guidelines (text, practice, status, version, created_at) "
                 "VALUES ('cut losers fast', 'good', 'active', 1, 'now')")
    conn.execute("INSERT INTO guidelines (text, practice, status, version, created_at) "
                 "VALUES ('chasing hype names', 'bad', 'active', 2, 'now')")
    conn.commit()
    texts = gmod.active_texts(conn)
    assert "DO: cut losers fast" in texts
    assert "AVOID: chasing hype names" in texts


def test_self_reflection_note_is_shared_as_a_lesson_message(monkeypatch):
    """The "communication" piece: after a round resolves, both agents' own
    self-reflection notes get posted as messages, so the rival can actually
    see what the other one learned — not just silently self-improve."""
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    for name, cap in (("RONIN", 110_000), ("VIPER", 90_000)):
        aid = conn.execute(
            "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
            "WHERE l.name=?", (name,)).fetchone()["id"]
        conn.execute("UPDATE round_states SET current_capital=? "
                     "WHERE round_id=? AND agent_id=?", (cap, rid, aid))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    lessons = conn.execute(
        "SELECT COUNT(*) c FROM agent_messages WHERE round_id=? AND kind='lesson'",
        (rid,)).fetchone()["c"]
    # every current lineage (now 3 by default) trades the same round and
    # posts a lesson — one reinforcement (winner) + N-1 self-critiques
    assert lessons == 3


def test_rival_sees_last_round_lesson_at_the_start_of_the_next_round(monkeypatch):
    """Lessons are posted scoped to the round that just ENDED. Without the
    cross-round fallback in _recent_messages, the very next round would never
    surface them to the rival."""
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id = conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name='RONIN'").fetchone()["id"]
    viper_id = conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name='VIPER'").fetchone()["id"]
    conn.execute("UPDATE round_states SET current_capital=110000 "
                 "WHERE round_id=? AND agent_id=?", (rid, ronin_id))
    conn.execute("UPDATE round_states SET current_capital=90000 "
                 "WHERE round_id=? AND agent_id=?", (rid, viper_id))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    # a brand-new round starts; VIPER (now a new agent id, generation 2)
    # should still see RONIN's lesson from the round that just ended
    new_rid = forward.start_round(conn, config, days=7)
    new_viper_id = conn.execute(
        "SELECT current_agent_id FROM lineages WHERE name='VIPER'").fetchone()["current_agent_id"]
    seen = forward._recent_messages(conn, new_rid, new_viper_id)
    assert any("Lesson from this round" in m for m in seen)
    assert any("RONIN" in m for m in seen)


def test_experience_draft_is_grounded_in_both_lineages_notes():
    """_recent_experience pulls each lineage's own self-reflection notes
    (not the opponent's) — the raw material an LLM-based drafting pass would
    "discuss" to find a shared lesson."""
    conn = _fresh_conn()
    config = ArenaConfig()
    forward.ensure_autonomous_lineages(conn, config)
    ronin_id = conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name='RONIN'").fetchone()["id"]
    import json
    conn.execute("UPDATE agents SET strategy_config=? WHERE id=?",
                 (json.dumps({"mode": "autonomous", "notes": "cut losers fast next time"}),
                  ronin_id))
    conn.commit()

    exp = gmod._recent_experience(conn, per_lineage=3)
    lineages_seen = {e["lineage"] for e in exp}
    assert "RONIN" in lineages_seen
    ronin_notes = next(e["notes"] for e in exp if e["lineage"] == "RONIN")
    assert any("cut losers fast" in n["note"] for n in ronin_notes)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

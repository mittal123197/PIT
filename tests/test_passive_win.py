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


def test_zero_trade_winner_gets_the_capped_bonus_not_the_full_one(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id, viper_id = _agent_id(conn, "RONIN"), _agent_id(conn, "VIPER")
    # RONIN never trades (stays at starting capital); VIPER trades and loses
    conn.execute("UPDATE round_states SET current_capital=100000, trade_count=0 "
                 "WHERE round_id=? AND agent_id=?", (rid, ronin_id))
    conn.execute("UPDATE round_states SET current_capital=88000, trade_count=3, "
                 "status='liquidated' WHERE round_id=? AND agent_id=?", (rid, viper_id))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    outcome = forward._resolve(conn, rnd, config, lambda t: 100.0)

    assert outcome["winner"] == "RONIN"
    assert outcome["passive_win"] is True
    stake = conn.execute("SELECT current_stake FROM lineages WHERE name='RONIN'").fetchone()["current_stake"]
    full_bonus_stake = 100000 * (1 + config.win_stake_bonus_pct / 100)
    capped_stake = 100000 * (1 + config.passive_win_stake_bonus_pct / 100)
    assert stake == round(capped_stake, 2)
    assert stake < full_bonus_stake
    assert config.passive_win_stake_bonus_pct < 10.0  # the explicit ask: cap under 10%


def test_genuine_active_winner_still_gets_the_full_bonus(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id, viper_id = _agent_id(conn, "RONIN"), _agent_id(conn, "VIPER")
    conn.execute("UPDATE round_states SET current_capital=105000, trade_count=2 "
                 "WHERE round_id=? AND agent_id=?", (rid, ronin_id))
    conn.execute("UPDATE round_states SET current_capital=95000, trade_count=1 "
                 "WHERE round_id=? AND agent_id=?", (rid, viper_id))
    conn.execute(
        "INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
        "capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)",
        (rid, ronin_id, forward._now(), "AAA", "buy", 10, 100.0, 105000, "t"))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    outcome = forward._resolve(conn, rnd, config, lambda t: 100.0)

    assert outcome["passive_win"] is False
    stake = conn.execute("SELECT current_stake FROM lineages WHERE name='RONIN'").fetchone()["current_stake"]
    # RONIN's return here is +5% (105000 vs 100000 starting) — genuinely
    # positive, not just "lost the least" — so the extra positive-delta
    # bonus is in play on top of the base win bonus (see forward._resolve).
    assert stake == round(
        100000 * (1 + (config.win_stake_bonus_pct + config.win_positive_delta_bonus_pct) / 100), 2)


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

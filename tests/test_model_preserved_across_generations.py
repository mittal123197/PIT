"""A real, long-standing bug: `_reflect_loser`'s new generation config dropped
the "model" field entirely. Since `decide()` falls back to Groq's
DEFAULT_MODEL whenever `model` is None, EVERY agent that had ever lost even
once — i.e. almost every agent, almost immediately — silently stopped using
its assigned model and started running on Groq instead, invalidating any
"different model per agent" comparison from generation 2 onward.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import sqlite3  # noqa: E402

from pit import db as dbm, forward  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    return conn


def _current_agent_id(conn, name):
    """The lineage's CURRENT agent — unlike a plain agents/lineages join,
    this follows lineages.current_agent_id specifically, so it stays correct
    across generations (a plain join can silently return gen-1's id forever)."""
    return conn.execute(
        "SELECT current_agent_id FROM lineages WHERE name=?",
        (name,)).fetchone()["current_agent_id"]


def _agent_id(conn, name):
    return conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name=?", (name,)).fetchone()["id"]


def test_losers_new_generation_keeps_its_assigned_model(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id, viper_id, lynx_id = (_agent_id(conn, n) for n in ("RONIN", "VIPER", "LYNX"))
    # RONIN wins clearly; VIPER and LYNX lose
    conn.execute("UPDATE round_states SET current_capital=110000, trade_count=1 "
                 "WHERE round_id=? AND agent_id=?", (rid, ronin_id))
    conn.execute("UPDATE round_states SET current_capital=90000 "
                 "WHERE round_id=? AND agent_id=?", (rid, viper_id))
    conn.execute("UPDATE round_states SET current_capital=85000 "
                 "WHERE round_id=? AND agent_id=?", (rid, lynx_id))
    conn.execute(
        "INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
        "capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)",
        (rid, ronin_id, forward._now(), "AAA", "buy", 10, 100.0, 110000, "t"))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    for name, expected_model in (("VIPER", forward._VIPER_MODEL),
                                 ("LYNX", forward._LYNX_MODEL)):
        lineage_id = conn.execute(
            "SELECT id FROM lineages WHERE name=?", (name,)).fetchone()["id"]
        newest = conn.execute(
            "SELECT strategy_config FROM agents WHERE lineage_id=? "
            "ORDER BY generation DESC LIMIT 1", (lineage_id,)).fetchone()
        cfg = json.loads(newest["strategy_config"])
        assert cfg.get("model") == expected_model, (
            f"{name}'s new generation lost its model — got {cfg.get('model')!r}, "
            f"would silently fall back to Groq's DEFAULT_MODEL in decide()")


def test_model_survives_multiple_consecutive_losses(monkeypatch):
    """The bug compounds: even if fixed for one generation, it must not
    regress across several evolutions in a row."""
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()

    for round_num in range(3):
        rid = forward.start_round(conn, config, days=7)
        ronin_id, viper_id, lynx_id = (_current_agent_id(conn, n) for n in ("RONIN", "VIPER", "LYNX"))
        # capital must be set RELATIVE to each round's own starting_capital
        # (which shifts every round as stakes grow/shrink) — not a fixed
        # absolute number, or "winner" flips unpredictably round to round
        for aid, ret in ((ronin_id, 0.10), (viper_id, -0.10), (lynx_id, -0.15)):
            starting = conn.execute(
                "SELECT starting_capital FROM round_states WHERE round_id=? AND agent_id=?",
                (rid, aid)).fetchone()["starting_capital"]
            conn.execute("UPDATE round_states SET current_capital=? "
                         "WHERE round_id=? AND agent_id=?",
                         (starting * (1 + ret), rid, aid))
        conn.execute("UPDATE round_states SET trade_count=1 WHERE round_id=? AND agent_id=?",
                     (rid, ronin_id))
        conn.execute(
            "INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
            "capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, ronin_id, forward._now(), "AAA", "buy", 10, 100.0, 110000, "t"))
        conn.commit()
        rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
        forward._resolve(conn, rnd, config, lambda t: 100.0)

    viper_lineage = conn.execute("SELECT id FROM lineages WHERE name='VIPER'").fetchone()["id"]
    newest = conn.execute(
        "SELECT generation, strategy_config FROM agents WHERE lineage_id=? "
        "ORDER BY generation DESC LIMIT 1", (viper_lineage,)).fetchone()
    assert newest["generation"] == 4  # gen 1 + 3 losses
    cfg = json.loads(newest["strategy_config"])
    assert cfg.get("model") == forward._VIPER_MODEL


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

"""Stake rules: no bonus unless the winner made money, and flat stakes when
stake_evolution is off."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from pit import db as dbm, forward
from pit.config import ArenaConfig


def _round(config, caps):
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON"); conn.executescript(dbm.SCHEMA_PATH.read_text())
    rid = forward.start_round(conn, config, days=7)
    for name, cap in caps.items():
        aid = conn.execute("SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id WHERE l.name=?", (name,)).fetchone()["id"]
        conn.execute("UPDATE round_states SET current_capital=?, trade_count=2 WHERE round_id=? AND agent_id=?", (cap, rid, aid))
    conn.commit()
    return conn, dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())


def _stake(conn, name):
    return conn.execute("SELECT current_stake FROM lineages WHERE name=?", (name,)).fetchone()["current_stake"]


def test_negative_winner_gets_no_bonus_and_nothing_is_redistributed(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    config = ArenaConfig()
    conn, rnd = _round(config, {"RONIN": 99000, "VIPER": 98000, "LYNX": 97000})
    forward._resolve(conn, rnd, config, lambda t: 100.0)
    assert _stake(conn, "RONIN") == 100000          # best of a losing bunch: no bonus
    # losers still take only their base penalties (no extra redistribution)
    assert _stake(conn, "VIPER") == round(100000 * (1 - config.middle_place_penalty_pct / 100), 2)
    assert _stake(conn, "LYNX") == round(100000 * (1 - config.loss_stake_penalty_pct / 100), 2)


def test_flat_mode_keeps_every_stake_at_base_capital(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    config = ArenaConfig(stake_evolution=0)
    conn, rnd = _round(config, {"RONIN": 105000, "VIPER": 99000, "LYNX": 95000})
    forward._resolve(conn, rnd, config, lambda t: 100.0)
    for n in ("RONIN", "VIPER", "LYNX"):
        assert _stake(conn, n) == config.base_capital
    # ...and the next round starts everyone flat too
    forward._active_round  # resolved round no longer active
    rid = forward.start_round(conn, config, days=7)
    caps = {r[0] for r in conn.execute("SELECT starting_capital FROM round_states WHERE round_id=?", (rid,))}
    assert caps == {config.base_capital}

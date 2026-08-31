"""Win/loss learning, redesigned: no more "recreation" (rebuilding the loser
out of the winner's trade log). Both agents study only their OWN trades —
the winner reinforces what worked, the loser self-critiques its mistakes —
and a separate, pool-wide guideline reflection pass now actually runs from
the forward/live path (it used to only be wired into the legacy engine).
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


def _resolvable_round(conn, config, winner_capital=110_000, loser_capital=90_000):
    """A round with clear cash-only returns and a couple of fake fills each,
    so reflection has something to read without any network/LLM access."""
    rid = forward.start_round(conn, config, days=7)
    ids = {}
    for name, cap in (("RONIN", winner_capital), ("VIPER", loser_capital)):
        aid = conn.execute(
            "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
            "WHERE l.name=?", (name,)).fetchone()["id"]
        ids[name] = aid
        conn.execute("UPDATE round_states SET current_capital=? "
                     "WHERE round_id=? AND agent_id=?", (cap, rid, aid))
        conn.execute(
            "INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, "
            "price, capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, aid, forward._now(), "AAA", "buy", 10, 100.0, cap, "test fill"))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    return rnd, ids


def test_winner_note_updates_same_agent_no_new_generation(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _resolvable_round(conn, config)
    price = lambda t: 100.0
    forward._resolve(conn, rnd, config, price)

    ronin = conn.execute("SELECT * FROM agents WHERE id=?", (ids["RONIN"],)).fetchone()
    assert ronin["generation"] == 1  # unchanged — no new agent for a win
    cfg = json.loads(ronin["strategy_config"])
    assert cfg["notes"]  # reinforcement note was written in place


def test_loser_note_is_seeded_from_its_own_agent_not_the_winners(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _resolvable_round(conn, config)
    price = lambda t: 100.0
    forward._resolve(conn, rnd, config, price)

    new_viper = conn.execute(
        "SELECT * FROM agents WHERE lineage_id=(SELECT lineage_id FROM agents WHERE id=?) "
        "ORDER BY generation DESC LIMIT 1", (ids["VIPER"],)).fetchone()
    assert new_viper["generation"] == 2
    assert new_viper["seeded_from_trade_agent_id"] == ids["VIPER"]  # self, not RONIN
    assert new_viper["seeded_from_trade_agent_id"] != ids["RONIN"]


def test_fallback_reflection_never_mentions_the_opponent(monkeypatch):
    """Without an LLM, the offline fallback summary must stay self-only —
    no leakage of the opponent's tickers into either note."""
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ids = {}
    for name, cap, sym in (("RONIN", 110_000, "WINSYM"), ("VIPER", 90_000, "LOSESYM")):
        aid = conn.execute(
            "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
            "WHERE l.name=?", (name,)).fetchone()["id"]
        ids[name] = aid
        conn.execute("UPDATE round_states SET current_capital=? "
                     "WHERE round_id=? AND agent_id=?", (cap, rid, aid))
        conn.execute(
            "INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, "
            "price, capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, aid, forward._now(), sym, "buy", 10, 100.0, cap, "test fill"))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    outcome = forward._resolve(conn, rnd, config, lambda t: 100.0)

    assert "LOSESYM" not in outcome["winner_note"]
    assert "WINSYM" not in outcome["mutation_note"]
    assert "WINSYM" in outcome["winner_note"]     # winner cites only its own symbol
    assert "LOSESYM" in outcome["mutation_note"]  # loser cites only its own symbol


def test_reflection_pass_wired_into_forward_resolve(monkeypatch):
    """The pool-wide guideline reflection used to only run from the legacy
    engine; it must now fire from forward._resolve() too, every
    reflection_interval rounds, once there's enough history to draw on."""
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig(reflection_interval=1, reflection_min_sample=1,
                         reflection_pattern_frac=0.5)
    # round 1: RONIN wins with fewer trades and lower drawdown than VIPER —
    # a pattern draft_proposal's heuristics can detect on the very first round.
    rid = forward.start_round(conn, config, days=7)
    ronin_id = conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name='RONIN'").fetchone()["id"]
    viper_id = conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name='VIPER'").fetchone()["id"]
    conn.execute("UPDATE round_states SET current_capital=110000, trade_count=1 "
                 "WHERE round_id=? AND agent_id=?", (rid, ronin_id))
    conn.execute("UPDATE round_states SET current_capital=90000, trade_count=9 "
                 "WHERE round_id=? AND agent_id=?", (rid, viper_id))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())

    outcome = forward._resolve(conn, rnd, config, lambda t: 100.0)
    # A pattern was detected and put to a vote — the reflection pass reached
    # forward._resolve() at all, which is the thing that used to be missing.
    # With only 2 lineages, a winning one heuristically resists a new
    # constraint (see guidelines._heuristic_vote), so a tie -> rejected is the
    # CORRECT outcome here, not a bug — unanimity is required with a pool of 2.
    assert outcome["reflection"] is not None
    assert outcome["reflection"]["kind"] == "add"
    proposals = conn.execute("SELECT COUNT(*) c FROM guideline_proposals").fetchone()["c"]
    assert proposals == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

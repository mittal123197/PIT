"""The actual feature: all agents in the pool battle simultaneously every
round, ranked 1..N with no ties. Only 1st truly wins; everyone else is a
loser and self-critiques into a new generation — but middle finishers take
a smaller stake penalty than dead last (explicit design choice).
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


def _three_way_round(conn, config, returns: dict):
    """returns: {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0}"""
    rid = forward.start_round(conn, config, days=7)
    ids = {}
    for name, ret in returns.items():
        aid = _agent_id(conn, name)
        ids[name] = aid
        cap = 100_000 * (1 + ret / 100)
        conn.execute("UPDATE round_states SET current_capital=?, trade_count=1 "
                     "WHERE round_id=? AND agent_id=?", (cap, rid, aid))
        conn.execute(
            "INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
            "capital_after, reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, aid, forward._now(), "AAA", "buy", 1, 100.0, cap, "t"))
    conn.commit()
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    return rnd, ids


def test_start_round_always_seats_all_three(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    n = conn.execute("SELECT COUNT(*) c FROM round_states WHERE round_id=?",
                     (rnd["id"],)).fetchone()["c"]
    assert n == 3


def test_ranking_is_return_order_best_to_worst(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    outcome = forward._resolve(conn, rnd, config, lambda t: 100.0)
    names_in_order = [r["name"] for r in outcome["ranking"]]
    assert names_in_order == ["RONIN", "VIPER", "LYNX"]
    assert outcome["winner"] == "RONIN"
    assert outcome["loser"] == "LYNX"


def test_only_first_place_reinforces_everyone_else_self_critiques(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    for name in ("VIPER", "LYNX"):
        lineage_id = conn.execute("SELECT lineage_id FROM agents WHERE id=?",
                                  (ids[name],)).fetchone()["lineage_id"]
        newest = conn.execute(
            "SELECT generation, seeded_from_trade_agent_id FROM agents "
            "WHERE lineage_id=? ORDER BY generation DESC LIMIT 1", (lineage_id,)).fetchone()
        assert newest["generation"] == 2  # both losers evolve
        assert newest["seeded_from_trade_agent_id"] == ids[name]  # self-seeded

    ronin_lineage = conn.execute("SELECT lineage_id FROM agents WHERE id=?",
                                 (ids["RONIN"],)).fetchone()["lineage_id"]
    ronin_current = conn.execute(
        "SELECT generation FROM agents WHERE lineage_id=? ORDER BY generation DESC LIMIT 1",
        (ronin_lineage,)).fetchone()
    assert ronin_current["generation"] == 1  # winner does NOT evolve


def test_middle_place_penalty_is_smaller_than_last_place(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    viper_stake = conn.execute("SELECT current_stake FROM lineages WHERE name='VIPER'").fetchone()["current_stake"]
    lynx_stake = conn.execute("SELECT current_stake FROM lineages WHERE name='LYNX'").fetchone()["current_stake"]
    viper_expected = 100_000 * (1 - config.middle_place_penalty_pct / 100)
    lynx_expected = 100_000 * (1 - config.loss_stake_penalty_pct / 100)
    assert viper_stake == round(viper_expected, 2)
    assert lynx_stake == round(lynx_expected, 2)
    assert viper_stake > lynx_stake  # middle punished less than last
    assert config.middle_place_penalty_pct < config.loss_stake_penalty_pct


def test_winner_gets_full_bonus_and_both_others_count_as_losses(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    ronin = conn.execute("SELECT wins, losses, current_stake FROM lineages WHERE name='RONIN'").fetchone()
    viper = conn.execute("SELECT wins, losses FROM lineages WHERE name='VIPER'").fetchone()
    lynx = conn.execute("SELECT wins, losses FROM lineages WHERE name='LYNX'").fetchone()
    assert ronin["wins"] == 1 and ronin["losses"] == 0
    assert viper["wins"] == 0 and viper["losses"] == 1   # middle IS a loss
    assert lynx["wins"] == 0 and lynx["losses"] == 1
    assert ronin["current_stake"] == round(100_000 * (1 + config.win_stake_bonus_pct / 100), 2)


def test_round_rankings_captures_every_participant(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    rows = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM round_rankings WHERE round_id=? ORDER BY rank", (rnd["id"],))}
    assert len(rows) == 3
    assert rows[ids["RONIN"]]["rank"] == 1
    assert rows[ids["VIPER"]]["rank"] == 2
    assert rows[ids["LYNX"]]["rank"] == 3
    assert rows[ids["RONIN"]]["note"]
    assert rows[ids["LYNX"]]["note"]


def test_elo_pairwise_zero_sum_across_the_pool(monkeypatch):
    """Every pairwise ELO update is zero-sum, so the NET change summed
    across all participants in a round should be (near) zero."""
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    forward.ensure_autonomous_lineages(conn, config)  # seed before snapshotting
    before = {r["name"]: r["rating"] for r in conn.execute(
        "SELECT l.name, a.rating FROM lineages l JOIN agents a ON a.id=l.current_agent_id")}
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    forward._resolve(conn, rnd, config, lambda t: 100.0)
    after = {r["name"]: r["rating"] for r in conn.execute(
        "SELECT l.name, a.rating FROM lineages l JOIN agents a ON a.id=l.current_agent_id")}
    total_delta = sum(after[n] - before.get(n, after[n]) for n in after)
    assert abs(total_delta) < 0.01
    assert after["RONIN"] > before["RONIN"]  # winner's rating went up
    assert after["LYNX"] < before["LYNX"]    # last place's rating went down


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

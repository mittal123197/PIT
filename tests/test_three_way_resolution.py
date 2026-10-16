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




def test_only_the_winner_counts_a_win_and_stakes_stay_flat(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
    forward._resolve(conn, rnd, config, lambda t: 100.0)

    ronin = conn.execute("SELECT wins, losses FROM lineages WHERE name='RONIN'").fetchone()
    viper = conn.execute("SELECT wins, losses FROM lineages WHERE name='VIPER'").fetchone()
    lynx = conn.execute("SELECT wins, losses FROM lineages WHERE name='LYNX'").fetchone()
    assert ronin["wins"] == 1 and ronin["losses"] == 0
    assert viper["wins"] == 0 and viper["losses"] == 1   # middle IS a loss
    assert lynx["wins"] == 0 and lynx["losses"] == 1
    # flat stakes: the next round starts everyone at base capital again
    rid2 = forward.start_round(conn, config, days=7)
    caps = {r[0] for r in conn.execute(
        "SELECT starting_capital FROM round_states WHERE round_id=?", (rid2,))}
    assert caps == {config.base_capital}


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


def test_short_rounds_move_ratings_less_than_full_sessions(monkeypatch):
    gains = {}
    for minutes in (30, 390):
        conn = _fresh_conn()
        config = ArenaConfig()
        rnd, ids = _three_way_round(conn, config, {"RONIN": 5.0, "VIPER": 1.0, "LYNX": -3.0})
        conn.execute("UPDATE rounds SET length_min=? WHERE id=?", (minutes, rnd["id"]))
        rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rnd["id"],)).fetchone())
        forward._resolve(conn, rnd, config, lambda t: 100.0)
        gains[minutes] = conn.execute(
            "SELECT rating_delta FROM round_rankings WHERE round_id=? AND rank=1",
            (rnd["id"],)).fetchone()[0]
    assert abs(gains[30] / gains[390] - (30 / 390) ** 0.5) < 0.02


def test_drawdown_tiebreak_uses_peak_to_trough_from_marks(monkeypatch):
    """Two agents tied on return: the one whose account fell further from its
    peak during the round ranks lower."""
    conn = _fresh_conn()
    config = ArenaConfig()
    rnd, ids = _three_way_round(conn, config, {"RONIN": 1.0, "VIPER": 1.0, "LYNX": -3.0})
    for aid, path in ((ids["RONIN"], [0, 2.0, 1.0]), (ids["VIPER"], [0, 0.5, 1.0])):
        for r in path:
            conn.execute("INSERT INTO round_marks (round_id, agent_id, ts, return_pct) "
                         "VALUES (?,?,?,?)", (rnd["id"], aid, "t", r))
    conn.commit()
    dd = forward.max_drawdowns(conn, rnd["id"])
    assert dd[ids["RONIN"]] > 0.9 and dd[ids["VIPER"]] == 0
    out = forward._resolve(conn, rnd, config, lambda t: 100.0)
    assert [r["name"] for r in out["ranking"]][:2] == ["VIPER", "RONIN"]

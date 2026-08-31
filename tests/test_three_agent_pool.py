"""A pool of 3+ lineages, ALL trading simultaneously every round — the
explicit design ask: "3 agents should fight every battle," not a rotating
1v1 where someone sits out. Ranking is a full N-way order (never a tie),
with only 1st place truly "winning."
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


def test_seeds_all_three_lineages():
    conn = _fresh_conn()
    ids = forward.ensure_autonomous_lineages(conn, ArenaConfig())
    assert len(ids) == 3
    names = {r["name"] for r in conn.execute("SELECT name FROM lineages")}
    assert names == {"RONIN", "VIPER", "LYNX"}


def test_lynx_uses_plain_groq_no_openrouter_prefix():
    """A real bug this needed to avoid: a stray 'groq:' prefix would be
    passed straight to Groq's API as a literal (nonexistent) model name."""
    assert not forward._LYNX_MODEL.startswith("groq:")
    assert not forward._LYNX_MODEL.startswith("openrouter:")


def test_existing_two_lineage_db_gets_lynx_added_not_duplicated():
    """A DB that already has RONIN+VIPER from before this feature must gain
    LYNX on the next call, without re-creating RONIN/VIPER."""
    conn = _fresh_conn()
    config = ArenaConfig()
    forward._create_lineage(conn, "RONIN", {"mode": "autonomous", "notes": "",
                                            "model": forward._RONIN_MODEL}, config)
    forward._create_lineage(conn, "VIPER", {"mode": "autonomous", "notes": "",
                                            "model": forward._VIPER_MODEL}, config)
    ids = forward.ensure_autonomous_lineages(conn, config)
    assert len(ids) == 3
    names = [r["name"] for r in conn.execute("SELECT name FROM lineages ORDER BY id")]
    assert names == ["RONIN", "VIPER", "LYNX"]


def test_start_round_seats_every_lineage_in_the_same_round():
    """The actual ask: all 3 fight every battle, not a rotating 2-at-a-time."""
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    participant_lineages = {r["lineage_id"] for r in conn.execute(
        "SELECT a.lineage_id FROM round_states rs JOIN agents a ON a.id=rs.agent_id "
        "WHERE rs.round_id=?", (rid,))}
    all_lineages = {r["id"] for r in conn.execute("SELECT id FROM lineages")}
    assert participant_lineages == all_lineages
    assert len(participant_lineages) == 3


def test_rank_participants_orders_by_return_then_trades_then_drawdown():
    from pit.resolve import AgentSummary
    eps = 0.01
    summaries = [
        AgentSummary(1, 5.0, 3, 2.0),
        AgentSummary(2, 5.0, 1, 2.0),   # same return, fewer trades -> ranks above 1
        AgentSummary(3, -2.0, 0, 0.0),  # worst return -> last
    ]
    ranking = forward._rank_participants(summaries, eps)
    assert ranking == [2, 1, 3]


def test_ranking_never_produces_a_tie_even_on_identical_stats():
    """Agent_id is the deterministic final tiebreak — with N>2 there's no
    sudden-death mini-round to fall back on, so ties must still resolve."""
    from pit.resolve import AgentSummary
    summaries = [AgentSummary(3, 0.0, 0, 0.0), AgentSummary(1, 0.0, 0, 0.0),
                AgentSummary(2, 0.0, 0, 0.0)]
    ranking = forward._rank_participants(summaries, 0.01)
    assert ranking == [1, 2, 3]  # fully ordered, smallest agent_id wins ties


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

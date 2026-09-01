"""The bug this covers: a live session used to just mark the round 'ended'
and stop — it never resolved a winner/loser, never touched `lineages`
(wins/losses/stake/cumulative_return), and never mutated the loser into its
next generation. That made the leaderboard look frozen across live sessions
and made the "recreation" mechanic untestable outside forward/replay mode.

run_live() now calls the same forward._resolve() cascade + mutation used by
the day-based engine when the session's wall-clock time runs out. This test
drives run_live() end-to-end (real loop, real sleep-based timing) but stubs
out the LLM decision and market data so it's fast, offline, and deterministic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import sqlite3  # noqa: E402

from pit import db as dbm, forward, live  # noqa: E402
from pit.config import ArenaConfig  # noqa: E402


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(dbm.SCHEMA_PATH.read_text())
    return conn


def _wire_deterministic_winner(monkeypatch):
    """The very FIRST decide() call ever (RONIN's, since agents are processed
    in id order and RONIN is id 1) buys AAA; every other call — VIPER, LYNX,
    and RONIN's own later turns — always holds. AAA's price rises every time
    it's quoted, so RONIN's mark-to-market return climbs above the other two
    over the session — a clean, deterministic winner with no network/LLM
    calls anywhere in the loop.

    Deliberately NOT keyed off `model`: RONIN and VIPER can share the same
    default model (both currently default to the cheap DeepSeek tier), so
    model identity alone can't distinguish which agent is deciding.

    VIPER and LYNX tie exactly (both zero trades, zero return) — the
    ranking's deterministic tiebreak (agent_id ascending) always puts VIPER
    (created first, lower id) ahead of LYNX, so VIPER lands in the MIDDLE
    and LYNX is dead last. That's real, intended behavior with 3 agents
    always battling together now (not a rotating 1v1), not an artifact of
    this test."""
    state = {"price": 100.0}
    calls = {"n": 0}

    def fake_quote(ticker):
        state["price"] += 0.5
        return {"price": state["price"]}

    def fake_decide(view, tick, total_days, goal_pct, guidelines, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([{"ticker": "AAA", "side": "buy", "qty": 500,
                      "reason": "test buy"}], "", "", [])
        return ([], "", "", [])

    monkeypatch.setattr(live.market, "quote", fake_quote)
    monkeypatch.setattr(live.autonomous, "decide", fake_decide)
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)  # no LLM mutation call
    monkeypatch.setattr(live.time, "sleep", lambda s: None)       # busy-loop, no real wait


def test_run_live_resolves_and_returns_an_outcome(monkeypatch):
    _wire_deterministic_winner(monkeypatch)
    conn = _fresh_conn()
    config = ArenaConfig()
    result = live.run_live(conn, config=config, minutes=0.02, interval=900,
                           refresh=1, verbose=False)
    assert result["outcome"] is not None
    assert result["outcome"]["winner"] == "RONIN"
    assert result["outcome"]["loser"] == "LYNX"  # see _wire_deterministic_winner


def test_round_status_ends_as_resolved_not_just_ended(monkeypatch):
    _wire_deterministic_winner(monkeypatch)
    conn = _fresh_conn()
    result = live.run_live(conn, minutes=0.02, interval=900, refresh=1, verbose=False)
    rid = result["round_id"]
    status = conn.execute("SELECT status FROM rounds WHERE id=?", (rid,)).fetchone()["status"]
    assert status == "resolved"


def test_lineage_standings_persist_after_a_live_session(monkeypatch):
    """The reported bug, directly: wins/losses/cumulative_return_pct on
    `lineages` — what the leaderboard reads — must move after a live
    session, not stay frozen at their pre-session values."""
    _wire_deterministic_winner(monkeypatch)
    conn = _fresh_conn()
    before = {r["name"]: dict(r) for r in conn.execute(
        "SELECT name, wins, losses, cumulative_return_pct FROM lineages")}
    # lineages don't exist yet before the first round; force-create them so
    # "before" is a real, comparable snapshot
    forward.ensure_autonomous_lineages(conn, ArenaConfig())
    before = {r["name"]: dict(r) for r in conn.execute(
        "SELECT name, wins, losses, cumulative_return_pct FROM lineages")}
    assert before["RONIN"]["wins"] == 0 and before["VIPER"]["losses"] == 0

    live.run_live(conn, minutes=0.02, interval=900, refresh=1, verbose=False)

    after = {r["name"]: dict(r) for r in conn.execute(
        "SELECT name, wins, losses, cumulative_return_pct FROM lineages")}
    assert after["RONIN"]["wins"] == 1
    assert after["VIPER"]["losses"] == 1
    assert after["RONIN"]["cumulative_return_pct"] != before["RONIN"]["cumulative_return_pct"]


def test_loser_self_reflects_into_a_new_generation_after_a_live_session(monkeypatch):
    """The redesigned mechanic: VIPER (loser) gets a new generation seeded
    from its OWN prior trade log — never the winner's — since a loss is
    self-critique, not recreation from whoever beat it."""
    _wire_deterministic_winner(monkeypatch)
    conn = _fresh_conn()
    forward.ensure_autonomous_lineages(conn, ArenaConfig())
    viper_lineage = conn.execute(
        "SELECT id, current_agent_id FROM lineages WHERE name='VIPER'").fetchone()
    old_agent_id = viper_lineage["current_agent_id"]
    old_cfg = conn.execute("SELECT strategy_config FROM agents WHERE id=?",
                           (old_agent_id,)).fetchone()["strategy_config"]

    result = live.run_live(conn, minutes=0.02, interval=900, refresh=1, verbose=False)

    new_agent = conn.execute(
        "SELECT * FROM agents WHERE lineage_id=? ORDER BY generation DESC LIMIT 1",
        (viper_lineage["id"],)).fetchone()
    assert new_agent["generation"] == 2
    # seeded from itself (its own prior attempt), never the winner
    assert new_agent["seeded_from_trade_agent_id"] == old_agent_id
    assert new_agent["strategy_config"] != old_cfg
    # the lineage must now point at the new generation, not the old loser
    current = conn.execute("SELECT current_agent_id FROM lineages WHERE id=?",
                           (viper_lineage["id"],)).fetchone()["current_agent_id"]
    assert current == new_agent["id"]
    assert result["outcome"]["mutation_note"]


def test_winner_reinforces_in_place_without_a_new_generation(monkeypatch):
    """A win doesn't spawn a new agent/generation — the SAME agent's notes
    are updated with reinforcement from its own trades."""
    _wire_deterministic_winner(monkeypatch)
    conn = _fresh_conn()
    forward.ensure_autonomous_lineages(conn, ArenaConfig())
    ronin_lineage = conn.execute(
        "SELECT id, current_agent_id FROM lineages WHERE name='RONIN'").fetchone()
    winner_agent_id = ronin_lineage["current_agent_id"]

    result = live.run_live(conn, minutes=0.02, interval=900, refresh=1, verbose=False)

    still_same_agent = conn.execute(
        "SELECT id, generation FROM agents WHERE lineage_id=? ORDER BY generation DESC LIMIT 1",
        (ronin_lineage["id"],)).fetchone()
    assert still_same_agent["id"] == winner_agent_id  # no new agent row
    assert still_same_agent["generation"] == 1
    assert result["outcome"]["winner_note"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

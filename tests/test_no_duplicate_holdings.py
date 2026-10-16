"""Agents see rivals' returns only — not their holdings or trades — and
may buy any name — the old "no buying
what a rival holds" block was removed: first mover took the obvious names and
pushed everyone else into thin ones, which is a bias, not a choice."""
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


def _agent_id(conn, name):
    return conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name=?", (name,)).fetchone()["id"]






def test_tick_allows_a_buy_on_a_ticker_a_rival_already_holds(monkeypatch):
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id, viper_id = _agent_id(conn, "RONIN"), _agent_id(conn, "VIPER")
    # VIPER already holds AAA going into this tick
    conn.execute("UPDATE round_states SET holdings=? WHERE round_id=? AND agent_id=?",
                 (json.dumps({"AAA": 10.0}), rid, viper_id))
    conn.commit()

    def fake_decide(view, tick, total_days, goal_pct, guidelines, model=None):
        # identify "not VIPER" by an empty book rather than by model string —
        # RONIN and VIPER can share the same model (both default to the cheap
        # DeepSeek tier), so model identity alone can't distinguish agents
        if not view["positions"]:
            assert "rival_held_tickers" not in view   # rivals' holdings are private
            return ([{"ticker": "AAA", "side": "buy", "qty": 5, "reason": "t"}], "", "", [])
        return ([], "", "", [])

    monkeypatch.setattr(live.autonomous, "decide", fake_decide)
    monkeypatch.setattr(live.market, "quote", lambda t: {"price": 100.0})
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    live._tick(conn, config, rnd, tick=1, verbose=False)

    ronin_holdings = json.loads(conn.execute(
        "SELECT holdings FROM round_states WHERE round_id=? AND agent_id=?",
        (rid, ronin_id)).fetchone()["holdings"])
    assert ronin_holdings.get("AAA") == 5  # the buy went through


def test_tick_allows_a_buy_on_a_ticker_nobody_else_holds(monkeypatch):
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id = _agent_id(conn, "RONIN")

    def fake_decide(view, tick, total_days, goal_pct, guidelines, model=None):
        if model == forward._RONIN_MODEL:
            return ([{"ticker": "FREESTOCK", "side": "buy", "qty": 5, "reason": "t"}], "", "", [])
        return ([], "", "", [])

    monkeypatch.setattr(live.autonomous, "decide", fake_decide)
    monkeypatch.setattr(live.market, "quote", lambda t: {"price": 100.0})
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    live._tick(conn, config, rnd, tick=1, verbose=False)

    ronin_holdings = json.loads(conn.execute(
        "SELECT holdings FROM round_states WHERE round_id=? AND agent_id=?",
        (rid, ronin_id)).fetchone()["holdings"])
    assert "FREESTOCK" in ronin_holdings


def test_snapshot_is_taken_before_this_ticks_own_trades_avoiding_order_bias(monkeypatch):
    """If nobody held a ticker at the start of the tick, two agents buying it
    in the SAME tick is allowed for both — the block only stops buying what a
    rival held BEFORE this decision cycle, not a same-moment coincidence."""
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)

    def fake_decide(view, tick, total_days, goal_pct, guidelines, model=None):
        assert "rival_held_tickers" not in view and "rival_recent_moves" not in view
        return ([{"ticker": "SHARED", "side": "buy", "qty": 1, "reason": "t"}], "", "", [])

    monkeypatch.setattr(live.autonomous, "decide", fake_decide)
    monkeypatch.setattr(live.market, "quote", lambda t: {"price": 100.0})
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    live._tick(conn, config, rnd, tick=1, verbose=False)

    for name in ("RONIN", "VIPER", "LYNX"):
        aid = _agent_id(conn, name)
        holdings = json.loads(conn.execute(
            "SELECT holdings FROM round_states WHERE round_id=? AND agent_id=?",
            (rid, aid)).fetchone()["holdings"])
        assert "SHARED" in holdings  # every agent's buy went through


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

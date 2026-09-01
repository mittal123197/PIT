"""Agents can't buy a stock a rival already holds. The reveal is minimal
(ticker symbols only, this round, no size/price/reasoning) and the block is
enforced server-side (not just requested of the LLM), using a snapshot taken
once at the start of each decision cycle — so processing order within a tick
can't hand one agent a race-condition advantage over another.
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


def _agent_id(conn, name):
    return conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name=?", (name,)).fetchone()["id"]


def test_held_by_rivals_excludes_self_and_unions_everyone_else():
    snapshot = {1: {"OWN"}, 2: {"BBB", "CCC"}, 3: {"EEE", "DDD"}}
    assert forward._held_by_rivals(snapshot, 1) == ["BBB", "CCC", "DDD", "EEE"]
    assert forward._held_by_rivals(snapshot, 2) == ["DDD", "EEE", "OWN"]


def test_drop_blocked_buys_only_strips_matching_buys():
    orders = [
        {"side": "buy", "ticker": "AAA", "reason": "t"},
        {"side": "buy", "ticker": "ZZZ", "reason": "t"},
        {"side": "sell", "ticker": "AAA", "reason": "t"},  # sells never blocked
    ]
    clean, rejected = forward._drop_blocked_buys(orders, {"AAA"})
    assert rejected == ["AAA"]
    assert clean == [orders[1], orders[2]]


def test_tick_rejects_a_buy_on_a_ticker_a_rival_already_holds(monkeypatch):
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
            assert "AAA" in view["rival_held_tickers"]
            return ([{"ticker": "AAA", "side": "buy", "qty": 5, "reason": "t"}], "", "", [])
        return ([], "", "", [])

    monkeypatch.setattr(live.autonomous, "decide", fake_decide)
    monkeypatch.setattr(live.market, "quote", lambda t: {"price": 100.0})
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    live._tick(conn, config, rnd, tick=1, verbose=False)

    ronin_holdings = json.loads(conn.execute(
        "SELECT holdings FROM round_states WHERE round_id=? AND agent_id=?",
        (rid, ronin_id)).fetchone()["holdings"])
    assert "AAA" not in ronin_holdings  # the buy was rejected


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
        assert view["rival_held_tickers"] == []  # nobody held anything at tick start
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

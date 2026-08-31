"""Two related hard-constraint fixes:

1. A per-position stop-loss, separate from the portfolio-level one — a single
   collapsing holding gets force-sold on its own, before the AGGREGATE book
   has to fall far enough to trigger the portfolio stop.
2. Every open position gets force-closed for REAL at round-end resolution —
   final_return_pct is a realized number, not a paper mark on stock nobody
   actually sold.
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


def _persist(conn, round_id, agent_id, st):
    """Mirror what the real callers (step_round/_tick) do after _execute:
    _execute only persists trade_count/cost_basis itself; cash/holdings are
    the caller's job."""
    conn.execute("UPDATE round_states SET current_capital=?, holdings=? "
                 "WHERE round_id=? AND agent_id=?",
                 (st["current_capital"], st["holdings"], round_id, agent_id))


def _agent_id(conn, name):
    return conn.execute(
        "SELECT a.id FROM agents a JOIN lineages l ON l.id=a.lineage_id "
        "WHERE l.name=?", (name,)).fetchone()["id"]


def test_execute_tracks_weighted_average_cost_basis():
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    aid = _agent_id(conn, "RONIN")
    st = dict(conn.execute("SELECT * FROM round_states WHERE round_id=? AND agent_id=?",
                          (rid, aid)).fetchone())
    price = lambda t: {"AAA": 100.0}[t]
    forward._execute(conn, rid, aid, st, [
        {"ticker": "AAA", "side": "buy", "qty": 10, "reason": "t"}], price, "now")
    _persist(conn, rid, aid, st)
    price2 = lambda t: {"AAA": 200.0}[t]
    forward._execute(conn, rid, aid, st, [
        {"ticker": "AAA", "side": "buy", "qty": 10, "reason": "t"}], price2, "now")
    _persist(conn, rid, aid, st)
    cost_basis = json.loads(st["cost_basis"])
    assert cost_basis["AAA"] == 150.0  # (10*100 + 10*200) / 20


def test_position_stop_sells_only_the_collapsing_ticker():
    """A single holding down past the per-position band gets force-sold on
    its own, even though the OTHER holding (and the aggregate book) is fine."""
    conn = _fresh_conn()
    config = ArenaConfig(position_stop_loss_pct=8.0)
    rid = forward.start_round(conn, config, days=7)
    aid = _agent_id(conn, "RONIN")
    st = dict(conn.execute("SELECT * FROM round_states WHERE round_id=? AND agent_id=?",
                          (rid, aid)).fetchone())
    buy_price = lambda t: 100.0
    forward._execute(conn, rid, aid, st, [
        {"ticker": "BAD", "side": "buy", "qty": 10, "reason": "t"},
        {"ticker": "GOOD", "side": "buy", "qty": 10, "reason": "t"},
    ], buy_price, "now")
    _persist(conn, rid, aid, st)

    # BAD collapses 10% (past the 8% band), GOOD is up 5% (fine)
    def live_price(t):
        return {"BAD": 90.0, "GOOD": 105.0}[t]
    sold = forward.check_position_stops(conn, rid, aid, st, live_price, "now", config)
    assert sold == 1
    holdings = json.loads(st["holdings"])
    assert "BAD" not in holdings
    assert "GOOD" in holdings  # untouched
    reason = conn.execute(
        "SELECT reason FROM trades WHERE round_id=? AND symbol='BAD' AND side='sell'",
        (rid,)).fetchone()["reason"]
    assert "position stop-loss" in reason


def test_position_within_band_is_left_alone():
    conn = _fresh_conn()
    config = ArenaConfig(position_stop_loss_pct=8.0)
    rid = forward.start_round(conn, config, days=7)
    aid = _agent_id(conn, "RONIN")
    st = dict(conn.execute("SELECT * FROM round_states WHERE round_id=? AND agent_id=?",
                          (rid, aid)).fetchone())
    forward._execute(conn, rid, aid, st, [
        {"ticker": "OK", "side": "buy", "qty": 10, "reason": "t"}], lambda t: 100.0, "now")
    _persist(conn, rid, aid, st)
    sold = forward.check_position_stops(conn, rid, aid, st, lambda t: 95.0, "now", config)  # -5%, within band
    assert sold == 0
    assert "OK" in json.loads(st["holdings"])


def test_close_out_all_positions_sells_everything_and_clears_cost_basis():
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    aid = _agent_id(conn, "RONIN")
    st = dict(conn.execute("SELECT * FROM round_states WHERE round_id=? AND agent_id=?",
                          (rid, aid)).fetchone())
    forward._execute(conn, rid, aid, st, [
        {"ticker": "AAA", "side": "buy", "qty": 10, "reason": "t"}], lambda t: 100.0, "now")
    _persist(conn, rid, aid, st)
    before_cash = st["current_capital"]
    forward.close_out_all_positions(conn, rid, aid, st, lambda t: 120.0, "now")
    assert json.loads(st["holdings"]) == {}
    assert json.loads(st["cost_basis"]) == {}
    assert st["current_capital"] == before_cash + 10 * 120.0
    sell_row = conn.execute(
        "SELECT reason FROM trades WHERE round_id=? AND symbol='AAA' AND side='sell'",
        (rid,)).fetchone()
    assert sell_row["reason"] == "round-end close"


def test_resolve_realizes_open_positions_instead_of_leaving_a_paper_mark(monkeypatch):
    monkeypatch.setattr("pit.llm.groq_available", lambda: False)
    conn = _fresh_conn()
    config = ArenaConfig()
    rid = forward.start_round(conn, config, days=7)
    ronin_id = _agent_id(conn, "RONIN")
    st = dict(conn.execute("SELECT * FROM round_states WHERE round_id=? AND agent_id=?",
                          (rid, ronin_id)).fetchone())
    forward._execute(conn, rid, ronin_id, st, [
        {"ticker": "AAA", "side": "buy", "qty": 10, "reason": "t"}], lambda t: 100.0, "now")
    _persist(conn, rid, ronin_id, st)
    rnd = dict(conn.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone())
    forward._resolve(conn, rnd, config, lambda t: 110.0)

    final_state = dict(conn.execute(
        "SELECT * FROM round_states WHERE round_id=? AND agent_id=?",
        (rid, ronin_id)).fetchone())
    assert json.loads(final_state["holdings"]) == {}
    # a real sell trade exists at the resolution price
    sell = conn.execute(
        "SELECT price FROM trades WHERE round_id=? AND symbol='AAA' AND side='sell' "
        "AND reason='round-end close'", (rid,)).fetchone()
    assert sell["price"] == 110.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

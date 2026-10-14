"""Fills pay half the spread; the agent's own sell record is a factual mirror
(arena-forced exits excluded) including what holding instead would be worth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, sqlite3
from pit import db as dbm, forward
from pit.config import ArenaConfig, DEFAULT


def _conn():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.executescript(dbm.SCHEMA_PATH.read_text())
    return c


def test_fill_pays_half_spread(monkeypatch):
    object.__setattr__(DEFAULT, "spread_bps", 10.0)
    try:
        assert abs(forward._fill(100.0, "buy") - 100.10) < 1e-9
        assert abs(forward._fill(100.0, "sell") - 99.90) < 1e-9
    finally:
        object.__setattr__(DEFAULT, "spread_bps", 0.0)


def test_sell_record_counts_only_own_sells_and_shows_holding_instead():
    c = _conn()
    rid = forward.start_round(c, ArenaConfig(), days=0)
    aid = c.execute("SELECT agent_id FROM round_states WHERE round_id=?", (rid,)).fetchone()[0]
    ins = ("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
           "capital_after, reason) VALUES (?,?,?,?,?,?,?,0,?)")
    c.execute(ins, (rid, aid, "t", "AAA", "buy", 10, 100, ""))
    c.execute(ins, (rid, aid, "t", "AAA", "sell", 10, 99, "flat, rotating"))     # -$10
    c.execute(ins, (rid, aid, "t", "BBB", "buy", 5, 50, ""))
    c.execute(ins, (rid, aid, "t", "BBB", "sell", 5, 52, "round-end close"))    # forced
    rec = forward.sell_record(c, rid, aid, lambda t: {"AAA": 103.0}.get(t))
    assert rec["sells_you_chose"] == 1 and rec["profitable"] == 0
    assert rec["net_from_those_sells"] == -10.0
    assert rec["if_you_had_held_those_shares_instead"] == 30.0


def test_charter_record_splits_on_and_off_charter_buys():
    c = _conn()
    rid = forward.start_round(c, ArenaConfig(), days=0)
    aid = c.execute("SELECT agent_id FROM round_states WHERE round_id=?", (rid,)).fetchone()[0]
    ins = ("INSERT INTO trades (round_id, agent_id, ts, symbol, side, qty, price, "
           "capital_after, reason) VALUES (?,?,?,?,?,?,?,0,?)")
    c.execute(ins, (rid, aid, "t", "AAA", "buy", 10, 100, "fits: dip to VWAP"))
    c.execute(ins, (rid, aid, "t", "BBB", "buy", 10, 50, "⚠ off-charter: no setup | chase"))
    c.execute(ins, (rid, aid, "t", "BBB", "sell", 10, 48, "fading"))
    rec = forward.charter_record(c, rid, aid, lambda t: {"AAA": 101.0}.get(t))
    assert rec["on_charter"] == {"buys": 1, "net_usd": 10.0}
    assert rec["off_charter"] == {"buys": 1, "net_usd": -20.0}

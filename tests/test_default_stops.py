"""Every buy carries a stop by default; an agent may set its own (bounded);
the per-position stop uses that distance."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from pit import db as dbm, forward, autonomous
from pit.config import ArenaConfig


def _setup():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON"); c.executescript(dbm.SCHEMA_PATH.read_text())
    cfg = ArenaConfig()
    rid = forward.start_round(c, cfg, days=0)
    aid = c.execute("SELECT agent_id FROM round_states WHERE round_id=? LIMIT 1", (rid,)).fetchone()[0]
    st = dict(c.execute("SELECT * FROM round_states WHERE round_id=? AND agent_id=?", (rid, aid)).fetchone())
    return c, cfg, rid, aid, st


def test_default_and_custom_stop_and_bounds():
    c, cfg, rid, aid, st = _setup()
    orders = autonomous._clean_orders([
        {"ticker": "AAA", "side": "buy", "amount": 100},                        # default
        {"ticker": "BBB", "side": "buy", "amount": 100, "stop_loss_pct": 2},    # own
        {"ticker": "CCC", "side": "buy", "amount": 100, "stop_loss_pct": 99}])  # clamped
    forward._execute(c, rid, aid, st, orders, lambda t: 10.0, "t")
    sp = json.loads(st["stop_pcts"])
    assert sp == {"AAA": cfg.position_stop_loss_pct, "BBB": 2.0, "CCC": forward.STOP_PCT_MAX}


def test_custom_stop_triggers_at_its_own_distance():
    c, cfg, rid, aid, st = _setup()
    forward._execute(c, rid, aid, st,
                     [{"ticker": "BBB", "side": "buy", "amount": 100, "stop_loss_pct": 2}],
                     lambda t: 10.0, "t")
    assert forward.check_position_stops(c, rid, aid, st, lambda t: 9.9, "t", cfg) == 0   # -1%
    assert forward.check_position_stops(c, rid, aid, st, lambda t: 9.79, "t", cfg) == 1  # -2.1%
    assert "BBB" not in json.loads(st["holdings"])


def test_oversized_buys_are_scaled_pro_rata_not_first_come():
    c, cfg, rid, aid, st = _setup()
    st["current_capital"] = 1000.0
    n = forward._execute(c, rid, aid, st, [
        {"ticker": "AAA", "side": "buy", "amount": 1000},
        {"ticker": "BBB", "side": "buy", "amount": 500},
        {"ticker": "CCC", "side": "buy", "amount": 500}], lambda t: 10.0, "t")
    h = json.loads(st["holdings"])
    assert n == 3 and set(h) == {"AAA", "BBB", "CCC"}
    assert abs(h["AAA"] * 10 - 500) < 0.01 and abs(h["BBB"] * 10 - 250) < 0.01

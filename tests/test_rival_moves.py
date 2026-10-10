"""rival_recent_moves: rivals' latest buys/sells (who/side/ticker/when) with no
price/size/reasoning, excluding forced closes; and the neutral scan shape."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from pit import db as dbm, forward, market


def _setup():
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON"); conn.executescript(dbm.SCHEMA_PATH.read_text())
    rid = forward.start_round(conn, forward.DEFAULT, days=0)
    ids = {r["name"]: r["current_agent_id"] for r in conn.execute("SELECT name,current_agent_id FROM lineages")}
    return conn, rid, ids


def _trade(conn, rid, aid, side, sym, reason):
    conn.execute("INSERT INTO trades (round_id,agent_id,ts,symbol,side,qty,price,capital_after,reason) "
                 "VALUES (?,?,?,?,?,1,10,100,?)", (rid, aid, "2026-01-01 10:00:00", sym, side, reason))


def test_moves_show_rivals_only_without_price_size_or_reason_and_skip_forced_closes():
    conn, rid, ids = _setup()
    _trade(conn, rid, ids["VIPER"], "buy", "AAA", "my secret thesis")
    _trade(conn, rid, ids["LYNX"], "sell", "BBB", "taking profit")
    _trade(conn, rid, ids["LYNX"], "sell", "CCC", "position stop-loss (entry 10.00, -8%)")
    _trade(conn, rid, ids["VIPER"], "sell", "DDD", "round-end close")
    _trade(conn, rid, ids["RONIN"], "buy", "EEE", "mine")           # self: excluded
    conn.commit()
    moves = forward._rival_recent_moves(conn, rid, ids["RONIN"])
    assert {(m["rival"], m["side"], m["ticker"]) for m in moves} == {("VIPER", "buy", "AAA"), ("LYNX", "sell", "BBB")}
    for m in moves:
        assert set(m) == {"rival", "side", "ticker", "at"}           # no price/qty/reason leaked


def test_scan_is_an_unranked_alphabetical_sample(monkeypatch):
    monkeypatch.setattr(market, "_movers_download", lambda *a, **k: object())
    monkeypatch.setattr(market, "_extract_closes", lambda df, ts: {t: (10.0 + i, 10.0) for i, t in enumerate(sorted(ts, reverse=True))})
    monkeypatch.setattr("pit.full_market.random_sample", lambda n, seed=None, mode="full": ["ZZZ", "AAA", "MMM"])
    monkeypatch.setattr("pit.full_market.sector_map", lambda mode="full": {"AAA": ("a", "Tech")})
    monkeypatch.setattr("pit.full_market.universe_size", lambda mode="full": 3)
    r = market.scan_full_market(n=3)
    assert "gainers" not in r and "losers" not in r
    assert [x["ticker"] for x in r["sample"]] == ["AAA", "MMM", "ZZZ"]   # alphabetical, not by change_pct

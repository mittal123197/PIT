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




def test_scan_is_an_unranked_alphabetical_sample(monkeypatch):
    monkeypatch.setattr(market, "_movers_download", lambda *a, **k: object())
    monkeypatch.setattr(market, "_extract_closes", lambda df, ts: {t: (10.0 + i, 10.0) for i, t in enumerate(sorted(ts, reverse=True))})
    monkeypatch.setattr("pit.full_market.random_sample", lambda n, seed=None, mode="full": ["ZZZ", "AAA", "MMM"])
    monkeypatch.setattr("pit.full_market.sector_map", lambda mode="full": {"AAA": ("a", "Tech")})
    monkeypatch.setattr("pit.full_market.universe_size", lambda mode="full": 3)
    r = market.scan_full_market(n=3)
    assert "gainers" not in r and "losers" not in r
    assert [x["ticker"] for x in r["sample"]] == ["AAA", "MMM", "ZZZ"]   # alphabetical, not by change_pct


def test_rival_lessons_pass_on_the_principle_not_the_ticker(monkeypatch):
    from pit import forward, guidelines
    monkeypatch.setattr(guidelines, "_universe_symbols", lambda: {"META", "CRM", "XOM"})
    out = forward._redact_tickers("CRM/META were green; keep RSI<65, skip XOM.")
    assert "META" not in out and "CRM" not in out and "XOM" not in out
    assert "RSI<65" in out


def test_moving_average_ma_is_not_redacted_as_mastercard(monkeypatch):
    from pit import forward, guidelines
    monkeypatch.setattr(guidelines, "_universe_symbols", lambda: {"MA", "META"})
    out = forward._redact_tickers("above its 20-day MA and 50d MA; bought MA and META")
    assert out == "above its 20-day MA and 50d MA; bought [a stock] and [a stock]"

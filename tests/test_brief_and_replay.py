"""The live table's intraday columns, and a replay served from the market
history (same table format, prices at the replayed bar, no lookahead)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import pandas as pd
from pit import brief, replay, market


def test_intraday_columns_use_todays_open_and_vwap():
    idx = pd.date_range("2026-10-09 09:30", periods=8, freq="5min")
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107.0], index=idx)
    vol = pd.Series([1000.0] * 8, index=idx)
    m = brief._intraday(close, close + 0.5, close - 0.5, vol, avg_daily_vol=39000,
                        minutes_open=40, open_=pd.Series([99.0] * 8, index=idx))
    assert round(m["o"], 2) == round((107 / 99 - 1) * 100, 2)      # vs the real open, not bar 1's close
    assert m["vwap"] > 0 and m["m15"] > 0 and m["rng"] >= 90 and m["pace"] > 0
    assert "SPY +1.00% since open" in brief._market_line({"SPY": {"o": 1.0, "vwap": 0.2}, "AAA": {"o": 2.0}}, "top100")


def _panel():
    day = datetime.date(2026, 10, 9)
    rows = []
    for i, ts in enumerate(pd.date_range("2026-10-09 09:30", periods=3, freq="5min", tz="America/New_York")):
        for sym, px in (("AAA", 10 + i), ("SPY", 500 + i)):
            rows.append({"ts": ts, "sym": sym, "day": day, "bar": i, "px": float(px),
                         "o": float(i), "vwap": 0.1, "m15": None, "m30": None, "rng": 50.0,
                         "pace": 1.0, "d1": 1.0, "d5": 2.0, "d20": 3.0, "v20": 1.0, "v50": 1.0,
                         "rsi": 55.0, "p52": 80.0, "volx": 1.0, "dvol": 1.5, "f30": 0.0, "f60": 0.0,
                         "fpe": 20.0, "margin": 10.0, "revg": 5.0, "epsg": 5.0, "roe": 15.0,
                         "de": 50.0, "tgtup": 10.0})
    return pd.DataFrame(rows)


def test_replay_serves_the_live_table_format_from_history(monkeypatch):
    monkeypatch.setattr("pit.history.panel", lambda *a, **k: _panel())
    monkeypatch.setattr("pit.full_market.sp500_universe", lambda: [])
    monkeypatch.setattr(brief, "_load_fund", lambda: {})
    replay.start(day=datetime.date(2026, 10, 9), compress_minutes=10)
    try:
        b = replay.brief()
        assert b["n"] == 1 and b["text"].splitlines()[0] == brief._HEADER   # SPY is not tradeable
        assert "SPY" in b["market"] and "AAA" in b["prices"]
        assert market.quote("AAA")["price"] == 10.0                        # first bar at the start
        assert replay.vols() == {"AAA": 1.5}
    finally:
        replay.stop()
    assert replay._STATE == {}

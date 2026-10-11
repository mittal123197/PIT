"""Risk bands scale with round length (sqrt-time) and each asset's own
volatility; without volatility data the flat config bands stay in force."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pit import risk


def test_bands_scale_with_sqrt_time_and_volatility():
    vols = {"BTC-USD": 2.0, "ALT-USD": 6.0, "MID-USD": 4.0}
    b30 = risk.bands(vols, 30, "crypto", 1.5)
    b120 = risk.bands(vols, 120, "crypto", 1.5)
    # 4x longer round -> ~2x wider bands (until a clamp kicks in)
    assert abs(b120["sigma"] / b30["sigma"] - 2.0) < 0.05
    assert b30["auto_stops"]["ALT-USD"] > b30["auto_stops"]["BTC-USD"]
    assert b30["goal"] == round(b30["stop"] * 1.5, 2)
    # 30 min of a 4%/day coin: 4*sqrt(30/1440)=0.58% -> 3 sigma stop ~1.73%
    assert 1.6 < b30["stop"] < 1.9


def test_us_day_is_390_minutes_and_clamps_apply():
    b = risk.bands({"A": 1.5}, 390, "us", 1.5)
    assert b["sigma"] == 1.5                          # one full session = one daily sigma
    assert risk.bands({"A": 0.1}, 5, "us", 1.5)["auto_stops"]["A"] == risk.POS_MIN
    assert risk.bands({}, 30, "us", 1.5) is None


def test_auto_stop_falls_back_without_data():
    risk.clear()
    assert risk.auto_stop("BTC-USD", 8.0) == 8.0
    risk.set_round(30, "crypto", {"BTC-USD": 2.0})
    assert 0.5 <= risk.auto_stop("BTC-USD", 8.0) < 1.0
    risk.clear()


def test_replay_fill_times_are_shown_in_market_time():
    import time
    from datetime import datetime, timezone
    from pit import replay
    replay._STATE.update(sim_start=datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc),
                         sim_end=datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc),
                         start_wall=time.time() - 60, speed=39.0)
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        assert replay.agent_ts(ts) in ("2026-10-09 10:08 ET", "2026-10-09 10:09 ET")  # 60s x 39 = 39 market min
    finally:
        replay._STATE.clear()
    assert replay.agent_ts("2026-10-11 15:49:00") == "2026-10-11 15:49:00"

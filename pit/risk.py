"""Risk bands sized to the round: its length AND the market's real volatility.

The old bands were flat numbers tuned for 7-day rounds (portfolio stop -10%,
take-profit +15%, position stop -8%). In a 30-minute crypto round those can
never fire — a typical coin moves ~0.6% in 30 minutes — so the "stop-loss"
was decoration. Here every band is a multiple of the move you'd expect over
the round's own horizon:

    sigma_horizon = daily_vol * sqrt(horizon_minutes / minutes_per_trading_day)

(the standard square-root-of-time scaling). daily_vol is each asset's own
20-day standard deviation of daily returns, so a 2% stop means "2 sigma" for
BTC and an alt-coin gets proportionally more room instead of being stopped
out by ordinary noise.

  * position stop (default for every buy) = STOP_SIGMA x that asset's sigma
  * portfolio stop = PORT_SIGMA x the universe's median sigma
  * take-profit    = portfolio stop x risk_reward_ratio
"""
from __future__ import annotations

import math
import os
import statistics

STOP_SIGMA = float(os.getenv("PIT_STOP_SIGMA", "2.0"))
PORT_SIGMA = float(os.getenv("PIT_PORTFOLIO_STOP_SIGMA", "3.0"))
POS_MIN, POS_MAX = 0.5, 20.0
PORT_MIN, PORT_MAX = 1.0, 15.0

# set by live.run_live at session start; empty = no volatility data, callers
# fall back to the flat config bands
_STATE: dict = {"horizon_min": None, "day_min": None, "vols": {}}


def minutes_per_day(market: str) -> int:
    """Trading minutes behind one daily bar: crypto never closes."""
    return 1440 if market == "crypto" else 390


def horizon_sigma(daily_vol_pct: float, horizon_min: float, day_min: float) -> float:
    return daily_vol_pct * math.sqrt(max(horizon_min, 1) / day_min)


def _clamp(v: float, lo: float, hi: float) -> float:
    return round(min(hi, max(lo, v)), 2)


def bands(vols: dict[str, float], horizon_min: float, market: str,
          risk_reward: float) -> dict | None:
    """{"stop", "goal", "sigma", "auto_stops": {sym: pct}} or None without data."""
    vols = {s: v for s, v in vols.items() if v and v > 0}
    if not vols:
        return None
    day = minutes_per_day(market)
    med = statistics.median(vols.values())
    sig = horizon_sigma(med, horizon_min, day)
    stop = _clamp(PORT_SIGMA * sig, PORT_MIN, PORT_MAX)
    return {"stop": stop, "goal": round(stop * risk_reward, 2),
            "sigma": round(sig, 3),
            "auto_stops": {s: _clamp(STOP_SIGMA * horizon_sigma(v, horizon_min, day),
                                     POS_MIN, POS_MAX) for s, v in vols.items()}}


def set_round(horizon_min: float, market: str, vols: dict[str, float]) -> None:
    _STATE.update(horizon_min=horizon_min, day_min=minutes_per_day(market),
                  vols=dict(vols))


def clear() -> None:
    _STATE.update(horizon_min=None, day_min=None, vols={})


def auto_stop(symbol: str, fallback: float) -> float:
    """Default stop % for a buy of `symbol` this round."""
    v = _STATE["vols"].get(symbol)
    if not v or not _STATE["horizon_min"]:
        return fallback
    return _clamp(STOP_SIGMA * horizon_sigma(v, _STATE["horizon_min"], _STATE["day_min"]),
                  POS_MIN, POS_MAX)


def active() -> bool:
    return bool(_STATE["horizon_min"] and _STATE["vols"])

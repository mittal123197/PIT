"""On-demand real market data — no fixed universe.

The autonomous agents don't get a pool; they name any ticker and this fetches
its real data (yfinance, NSE `.NS`). `movers()` is a discovery aid — "what's
moving today" — ranked over a broad reference set so an agent has somewhere to
start looking, exactly like a human glancing at a top-gainers screen. Agents may
still research and trade ANY ticker, not just the movers.

Everything is cached per calendar day so a day's repeated calls hit the network
once.
"""
from __future__ import annotations

import datetime
import functools

from .config import _BUILTIN_UNIVERSE  # reference for the movers feed ONLY


def _yf():
    import yfinance as yf
    return yf


def _today_key() -> str:
    return datetime.date.today().isoformat()


@functools.lru_cache(maxsize=1024)
def _history_cached(ticker: str, period: str, day_key: str) -> tuple:
    yf = _yf()
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return ()
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        return tuple((idx.date().isoformat(), round(float(v), 2))
                     for idx, v in close.items())
    except Exception:
        return ()


def history(ticker: str, days: int = 30) -> list[tuple[str, float]]:
    """Daily (date, close) for any ticker. Empty list if it doesn't resolve."""
    period = f"{max(7, days + 6)}d"
    return list(_history_cached(ticker.upper().strip(), period, _today_key()))


def quote(ticker: str) -> dict | None:
    """Latest close + 1-day change for any ticker, or None if unknown."""
    h = history(ticker, days=7)
    if not h:
        return None
    price = h[-1][1]
    prev = h[-2][1] if len(h) > 1 else price
    return {
        "ticker": ticker.upper().strip(),
        "price": price,
        "prev_close": prev,
        "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
        "asof": h[-1][0],
    }


@functools.lru_cache(maxsize=8)
def _movers_cached(day_key: str, ref: tuple[str, ...], n: int) -> tuple:
    yf = _yf()
    rows = []
    try:
        df = yf.download(list(ref), period="7d", interval="1d",
                         progress=False, auto_adjust=True)
        close = df["Close"]
        for t in ref:
            try:
                s = close[t].dropna() if hasattr(close, "columns") else close.dropna()
                if len(s) >= 2:
                    price, prev = float(s.iloc[-1]), float(s.iloc[-2])
                    rows.append({"ticker": t, "price": round(price, 2),
                                 "change_pct": round((price / prev - 1) * 100, 2)})
            except Exception:
                continue
    except Exception:
        return ((), ())
    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return (tuple(rows[:n]), tuple(reversed(rows[-n:])))


def movers(n: int = 12, reference: list[str] | None = None) -> dict:
    """Discovery aid: top gainers/losers over a broad reference set (NOT a
    constraint — agents may trade any ticker). Cached per day."""
    ref = tuple(reference or _BUILTIN_UNIVERSE)
    gainers, losers = _movers_cached(_today_key(), ref, n)
    return {"gainers": [dict(r) for r in gainers],
            "losers": [dict(r) for r in losers]}


def is_trading_day(date: datetime.date | None = None) -> bool:
    """Rough NSE trading-day check: weekday and (best-effort) has data today.
    Holidays aren't enumerated here; a no-data day is simply skipped upstream."""
    date = date or datetime.date.today()
    return date.weekday() < 5

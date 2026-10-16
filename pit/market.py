"""Live market data: quotes and fundamentals for any ticker.

Two backends behind one interface — yfinance (default, free, no key) and
Alpaca (PIT_DATA_SOURCE=alpaca + ALPACA_API_KEY/SECRET, real-time IEX quotes);
Alpaca falls back to yfinance on error. During a replay, pit/replay.py swaps
`quote()` for the replayed day's prices.
"""
from __future__ import annotations

import datetime
import functools
import json
import os
import urllib.request


DATA_SOURCE = os.getenv("PIT_DATA_SOURCE", "yfinance").lower()
# the tradeable universe: 'top100' (US) or 'crypto' — see full_market.py
TRADE_UNIVERSE_MODE = os.getenv("PIT_TRADE_UNIVERSE", "top100").lower()
_ALPACA_DATA = "https://data.alpaca.markets"


def _use_alpaca() -> bool:
    return (DATA_SOURCE == "alpaca"
            and bool(os.getenv("ALPACA_API_KEY"))
            and bool(os.getenv("ALPACA_SECRET_KEY")))


def _bucket() -> str:
    """A cache key that rolls over every N seconds, so intraday prices refresh
    (a day-long key would freeze them). Floor of 15s to avoid hammering Yahoo."""
    secs = max(15, int(os.getenv("PIT_QUOTE_REFRESH_SEC", "120")))
    return str(int(datetime.datetime.now().timestamp()) // secs)


# last successfully-seen price per ticker, so a transient fetch failure returns
# the previous price instead of collapsing a position's value to zero.
_LAST_PRICE: dict[str, float] = {}


# ---- public interface (dispatches to a backend) -----------------------

def history(ticker: str, days: int = 30) -> list[tuple[str, float]]:
    if _use_alpaca():
        h = _alpaca_history(ticker, days)
        if h:
            return h
    return _yf_history(ticker, days)


def quote(ticker: str) -> dict | None:
    t = ticker.upper().strip()
    q = _alpaca_quote(t) if _use_alpaca() else None
    if not q:
        q = _yf_quote(t)
    if q and q.get("price"):
        _LAST_PRICE[t] = q["price"]
        return q
    if t in _LAST_PRICE:  # transient failure — reuse the last good price
        p = _LAST_PRICE[t]
        return {"ticker": t, "price": p, "prev_close": p, "change_pct": 0.0,
                "asof": "cached"}
    return None


_FUNDAMENTALS_TTL = 6 * 3600  # company fundamentals barely move intraday
_FUNDAMENTALS_CACHE: dict[str, tuple[float, dict | None]] = {}


def fundamentals(ticker: str) -> dict | None:
    """Key valuation/growth metrics for a ticker (free, via yfinance). Not a
    per-tick quote — cached for hours since P/E, margins etc. don't move
    intraday. Returns None if the ticker has no fundamentals (e.g. some
    micro-caps, crypto-proxies with no earnings)."""
    import time
    t = ticker.upper().strip()
    cached = _FUNDAMENTALS_CACHE.get(t)
    if cached and time.time() - cached[0] < _FUNDAMENTALS_TTL:
        return cached[1]
    try:
        import yfinance as yf
        info = yf.Ticker(t).info
        if not info or not info.get("regularMarketPrice") and not info.get("currentPrice"):
            # yfinance still returns a near-empty dict for delisted/invalid symbols
            if len(info) < 5:
                _FUNDAMENTALS_CACHE[t] = (time.time(), None)
                return None
        result = {
            "ticker": t,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "profit_margin_pct": _pct(info.get("profitMargins")),
            "revenue_growth_pct": _pct(info.get("revenueGrowth")),
            "earnings_growth_pct": _pct(info.get("earningsGrowth")),
            "return_on_equity_pct": _pct(info.get("returnOnEquity")),
            "debt_to_equity": info.get("debtToEquity"),
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
        }
        _FUNDAMENTALS_CACHE[t] = (time.time(), result)
        return result
    except Exception:
        _FUNDAMENTALS_CACHE[t] = (time.time(), None)
        return None


def _pct(v) -> float | None:
    return round(v * 100, 2) if isinstance(v, (int, float)) else None


def _round_price(v) -> float:
    """Round an asset PRICE (not a percentage, not an aggregate dollar
    value) to 8 decimals, not 2. 2 decimals is fine for equities but rounds
    a sub-cent crypto token (SHIB-USD is ~$0.000009) straight down to a flat
    0.0 — which then makes every downstream use of that price (scan/history
    display, and critically the actual fill price in forward._execute) treat
    the asset as worthless, silently turning every buy/sell of it into a
    no-op. 8 decimals is the standard crypto convention (satoshi-level) and
    doesn't change anything for equities — display code that wants exactly
    2 decimals still gets that at print time (e.g. cli.py's `:.2f`)."""
    return round(float(v), 8)


def _alpaca_get(path: str):
    req = urllib.request.Request(_ALPACA_DATA + path, headers={
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    })
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _alpaca_quote(ticker: str) -> dict | None:
    try:
        d = _alpaca_get(f"/v2/stocks/{ticker.upper().strip()}/snapshot")
        lt = d.get("latestTrade") or {}
        price = lt.get("p")
        prev = (d.get("prevDailyBar") or {}).get("c") or (d.get("dailyBar") or {}).get("o")
        if not price:
            return None
        return {"ticker": ticker.upper().strip(), "price": round(price, 2),
                "prev_close": round(prev, 2) if prev else round(price, 2),
                "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
                "asof": (lt.get("t") or "")[:10]}
    except Exception:
        return None


def _alpaca_history(ticker: str, days: int) -> list[tuple[str, float]]:
    try:
        d = _alpaca_get(f"/v2/stocks/{ticker.upper().strip()}/bars"
                        f"?timeframe=1Day&limit={max(7, days + 5)}")
        return [(b["t"][:10], round(b["c"], 2)) for b in d.get("bars", [])]
    except Exception:
        return []


def _yf():
    import yfinance as yf
    return yf


@functools.lru_cache(maxsize=1024)
def _history_cached(ticker: str, period: str, day_key: str) -> tuple:
    yf = _yf()
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True, timeout=15)
        if df is None or df.empty:
            return ()
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        return tuple((idx.date().isoformat(), _round_price(float(v)))
                     for idx, v in close.items())
    except Exception:
        return ()


def _yf_history(ticker: str, days: int = 30) -> list[tuple[str, float]]:
    period = f"{max(7, days + 6)}d"
    return list(_history_cached(ticker.upper().strip(), period, _bucket()))


def _yf_quote(ticker: str) -> dict | None:
    h = _yf_history(ticker, days=7)
    if not h:
        return None
    price = h[-1][1]
    prev = h[-2][1] if len(h) > 1 else price
    return {"ticker": ticker.upper().strip(), "price": price, "prev_close": prev,
            "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
            "asof": h[-1][0]}

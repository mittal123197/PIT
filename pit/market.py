"""On-demand real market data — no fixed universe.

Two backends behind one interface:
  * yfinance (default) — free, no key, but delayed ~15 min / end-of-day.
  * Alpaca (PIT_DATA_SOURCE=alpaca + ALPACA_API_KEY/SECRET) — free real-time US
    (IEX) quotes, daily bars, AND a real market-wide movers screener.

Agents name any ticker; this fetches its real data. `movers()` is a discovery
aid ("what's moving today"); agents may still trade ANY ticker. Each backend
falls back to yfinance on error so a missing key or a hiccup never stops a run.
"""
from __future__ import annotations

import datetime
import functools
import json
import os
import urllib.request

from .config import _BUILTIN_UNIVERSE  # reference for the yfinance movers feed

DATA_SOURCE = os.getenv("PIT_DATA_SOURCE", "yfinance").lower()
_ALPACA_DATA = "https://data.alpaca.markets"


def _use_alpaca() -> bool:
    return (DATA_SOURCE == "alpaca"
            and bool(os.getenv("ALPACA_API_KEY"))
            and bool(os.getenv("ALPACA_SECRET_KEY")))


def _today_key() -> str:
    return datetime.date.today().isoformat()


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


def movers(n: int = 12, reference: list[str] | None = None) -> dict:
    if _use_alpaca():
        m = _alpaca_movers(n)
        if m and (m["gainers"] or m["losers"]):
            return m
    return _yf_movers(n, reference)


def is_trading_day(date: datetime.date | None = None) -> bool:
    date = date or datetime.date.today()
    return date.weekday() < 5


# ---- Alpaca backend (real-time US via IEX) ----------------------------

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


def _alpaca_movers(n: int) -> dict:
    try:
        d = _alpaca_get(f"/v1beta1/screener/stocks/movers?top={n}")

        def conv(rows):
            return [{"ticker": m["symbol"], "price": round(m.get("price", 0), 2),
                     "change_pct": round(m.get("percent_change", 0), 2)}
                    for m in rows]
        return {"gainers": conv(d.get("gainers", [])),
                "losers": conv(d.get("losers", []))}
    except Exception:
        return {"gainers": [], "losers": []}


# ---- yfinance backend (free, delayed) ---------------------------------

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
        return tuple((idx.date().isoformat(), round(float(v), 2))
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


@functools.lru_cache(maxsize=8)
def _movers_cached(day_key: str, ref: tuple[str, ...], n: int) -> tuple:
    yf = _yf()
    rows = []
    try:
        df = yf.download(list(ref), period="7d", interval="1d",
                         progress=False, auto_adjust=True, timeout=15)
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


def _yf_movers(n: int = 12, reference: list[str] | None = None) -> dict:
    ref = tuple(reference or _BUILTIN_UNIVERSE)
    gainers, losers = _movers_cached(_bucket(), ref, n)
    return {"gainers": [dict(r) for r in gainers],
            "losers": [dict(r) for r in losers]}

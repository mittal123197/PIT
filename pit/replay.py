"""Replay a past trading day at compressed speed.

Monkey-patches `market.quote()` so every price fetch returns the value at the
current *replay cursor* — which advances on wall-clock time, compressing the
~6.5h US session into e.g. 15 minutes.

There is deliberately no default watchlist. Agents name tickers themselves; the
replay lazily loads 5-minute bars for each named ticker on first quote.

Everything else — agents, decisions, dashboard — is unchanged. To an agent the
replay looks like a live-moving market.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, time as dt_time, timedelta

from . import market

_ORIGINAL_QUOTE = None
_STATE: dict = {}


def _last_trading_day(before: date | None = None) -> date:
    """Most recent COMPLETED trading day (i.e. never today, since today's session
    may still be open / have no data yet)."""
    d = (before or date.today()) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def load_day(tickers: list[str] | None = None,
             day: date | None = None) -> dict:
    """Fetch 5-min bars for the target trading day. Returns {ticker: [(dt, price), ...]}."""
    tickers = [t.upper().strip() for t in (tickers or []) if t.strip()]
    if not tickers:
        return {}
    return _load_tickers(tickers, day or _last_trading_day())


def _load_tickers(tickers: list[str], day: date) -> dict:
    import yfinance as yf
    # yfinance intraday needs a small window; request a couple days and filter.
    start = day.isoformat()
    end = (day + timedelta(days=1)).isoformat()
    print(f"[replay] loading {len(tickers)} tickers for {day} (5-min bars)...",
          flush=True)
    df = yf.download(tickers, start=start, end=end, interval="5m",
                     progress=False, auto_adjust=True, timeout=30)
    if df is None or df.empty:
        raise RuntimeError(f"no intraday data available for {day}")
    close = df["Close"]
    series = {}
    for t in tickers:
        try:
            s = close[t].dropna() if hasattr(close, "columns") else close.dropna()
            if len(s) > 0:
                series[t] = [(idx.to_pydatetime(), round(float(v), 2))
                             for idx, v in s.items()]
        except (KeyError, AttributeError):
            continue
    if series:
        print(f"[replay] loaded {len(series)} ticker(s), "
              f"{len(next(iter(series.values())))} bars each", flush=True)
    return series


def start(tickers: list[str] | None = None, day: date | None = None,
          compress_minutes: int | None = None) -> None:
    """Begin replaying `day`, compressing the session into `compress_minutes`
    of wall clock. Monkey-patches market.quote so all callers see replay prices."""
    global _ORIGINAL_QUOTE
    day = day or _last_trading_day()
    series = load_day(tickers, day)
    compress = compress_minutes or int(os.getenv("PIT_REPLAY_MINUTES", "20"))
    if series:
        first = next(iter(series.values()))
        sim_start, sim_end = first[0][0], first[-1][0]
    else:
        sim_start = datetime.combine(day, dt_time(9, 30))
        sim_end = datetime.combine(day, dt_time(16, 0))
    real_span = (sim_end - sim_start).total_seconds()
    speed = real_span / max(60, compress * 60)  # e.g. speed=20 means 1 wall-sec = 20 sim-sec

    _STATE.clear()
    _STATE.update({"series": series, "start_wall": time.time(),
                   "sim_start": sim_start, "sim_end": sim_end, "day": day,
                   "missing": set(), "speed": speed,
                   "compress_minutes": compress})
    if _ORIGINAL_QUOTE is None:
        _ORIGINAL_QUOTE = market.quote
    market.quote = _replay_quote  # type: ignore
    print(f"[replay] unbiased ticker discovery for {day}: agents name symbols; "
          f"bars load on demand at {speed:.0f}x — "
          f"the session will play out over ~{compress} wall-clock minutes.",
          flush=True)


def stop() -> None:
    global _ORIGINAL_QUOTE
    if _ORIGINAL_QUOTE is not None:
        market.quote = _ORIGINAL_QUOTE
        _ORIGINAL_QUOTE = None
    _STATE.clear()


def sim_now() -> datetime | None:
    if not _STATE:
        return None
    elapsed = (time.time() - _STATE["start_wall"]) * _STATE["speed"]
    return _STATE["sim_start"] + timedelta(seconds=elapsed)


def is_finished() -> bool:
    now = sim_now()
    return bool(now and _STATE and now >= _STATE["sim_end"])


def _replay_quote(ticker: str) -> dict | None:
    t = ticker.upper().strip()
    now = sim_now()
    series = _STATE.get("series", {}).get(t)
    if not series and t not in _STATE.get("missing", set()) and _STATE.get("day"):
        got = _load_tickers([t], _STATE["day"])
        if got.get(t):
            _STATE["series"][t] = got[t]
            series = got[t]
        else:
            _STATE["missing"].add(t)
    if not series or not now:
        return _ORIGINAL_QUOTE(ticker) if _ORIGINAL_QUOTE else None
    # find the latest bar with time <= sim_now
    price = None
    for dt, p in series:
        if dt.replace(tzinfo=None) <= now.replace(tzinfo=None):
            price = p
        else:
            break
    if price is None:
        price = series[0][1]
    prev = series[0][1]
    return {"ticker": t, "price": price, "prev_close": prev,
            "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
            "asof": now.strftime("%H:%M")}

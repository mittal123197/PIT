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
from datetime import date, datetime, timedelta

from . import market

_ORIGINAL_QUOTE = None
_ORIGINAL_SCAN = None
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


def _download(tickers: list[str], day: date):
    import yfinance as yf
    start = day.isoformat()
    end = (day + timedelta(days=1)).isoformat()
    # threads=False: yfinance's default is to fire one thread per ticker,
    # which for a 150-ticker scan means ~150 near-simultaneous requests to
    # Yahoo's API — exactly the burst pattern that gets a chunk of them
    # throttled back with no data (yfinance's own error message for that is
    # the misleading "possibly delisted", even for large, very-much-listed
    # names like HON or SLB). Sequential requests are slower wall-clock, but
    # that's no longer a real cost here — the replay clock is paused for the
    # whole tick regardless (see pause()/resume()), so slower fetching
    # doesn't burn simulated market time either.
    return yf.download(tickers, start=start, end=end, interval="5m",
                       progress=False, auto_adjust=True, timeout=30,
                       threads=False)


def _extract_series(df, tickers: list[str]) -> dict:
    close = df["Close"]
    series = {}
    for t in tickers:
        try:
            s = close[t].dropna() if hasattr(close, "columns") else close.dropna()
            if len(s) > 0:
                # 8 decimals, not 2 — see market._round_price: 2 rounds a
                # sub-cent crypto price (e.g. SHIB-USD) straight to 0.0.
                series[t] = [(idx.to_pydatetime(), market._round_price(v))
                             for idx, v in s.items()]
        except (KeyError, AttributeError):
            continue
    return series


def _load_tickers(tickers: list[str], day: date) -> dict:
    print(f"[replay] loading {len(tickers)} tickers for {day} (5-min bars)...",
          flush=True)
    df = _download(tickers, day)
    if df is None or df.empty:
        raise RuntimeError(f"no intraday data available for {day}")
    series = _extract_series(df, tickers)

    # A request this size routinely drops a handful of tickers to Yahoo's own
    # throttling, not real data gaps — retry just the misses once, after a
    # short pause, rather than silently losing real, tradeable names every
    # single call.
    missing = [t for t in tickers if t not in series]
    if missing and len(missing) < len(tickers):
        time.sleep(2.0)
        retry_df = _download(missing, day)
        if retry_df is not None and not retry_df.empty:
            series.update(_extract_series(retry_df, missing))

    if series:
        print(f"[replay] loaded {len(series)} ticker(s), "
              f"{len(next(iter(series.values())))} bars each", flush=True)
    return series


# A reliably-liquid ticker used ONLY to anchor the sim clock to the real full
# trading session — never as a trade suggestion. An arbitrary randomly-sampled
# ticker is a bad anchor: some stocks trade in a narrow window or have sparse
# 5-min bars (a handful of bars covering a couple hours, not the full ~6.5h
# session), which silently truncates and skews the whole day's clock.
_CLOCK_ANCHOR_TICKER = "SPY"


def start(tickers: list[str] | None = None, day: date | None = None,
          compress_minutes: int | None = None) -> None:
    """Begin replaying `day`, compressing the session into `compress_minutes`
    of wall clock. Monkey-patches market.quote so all callers see replay prices.

    The sim clock is anchored to SPY's own bar range for the day (always a
    full, clean session) rather than whatever ticker an agent happens to look
    up first — that used to occasionally pin the whole session's clock to a
    thinly-traded stock's narrow, sparse data window."""
    global _ORIGINAL_QUOTE, _ORIGINAL_SCAN
    day = day or _last_trading_day()
    series = load_day(tickers, day)
    compress = compress_minutes or int(os.getenv("PIT_REPLAY_MINUTES", "20"))

    _STATE.clear()
    _STATE.update({"series": series, "day": day, "missing": set(),
                   "compress_minutes": compress,
                   "sim_start": None, "sim_end": None,
                   "start_wall": None, "speed": None})

    anchor = _load_tickers([_CLOCK_ANCHOR_TICKER], day).get(_CLOCK_ANCHOR_TICKER)
    if anchor and len(anchor) >= 10:  # sanity check: a real full session
        _STATE["series"][_CLOCK_ANCHOR_TICKER] = anchor
        _anchor_clock(anchor[0][0], anchor[-1][0])
    elif series:  # SPY somehow failed; fall back to whatever was given upfront
        first = next(iter(series.values()))
        _anchor_clock(first[0][0], first[-1][0])

    if _ORIGINAL_QUOTE is None:
        _ORIGINAL_QUOTE = market.quote
        _ORIGINAL_SCAN = market.scan_full_market
    market.quote = _replay_quote  # type: ignore
    market.scan_full_market = _replay_scan  # type: ignore
    print(f"[replay] unbiased ticker discovery for {day}: agents name symbols; "
          f"bars load on demand, session compressed to ~{compress} wall-clock "
          f"minutes (clock starts on the first ticker looked up).", flush=True)


def _anchor_clock(sim_start: datetime, sim_end: datetime) -> None:
    """Set the sim clock's start/end/speed from a real bar range, and start the
    wall-clock timer now. Called once, the first time any ticker is loaded."""
    compress = _STATE["compress_minutes"]
    real_span = (sim_end - sim_start).total_seconds()
    speed = real_span / max(60, compress * 60)
    _STATE.update({"sim_start": sim_start, "sim_end": sim_end,
                   "start_wall": time.time(), "speed": speed})
    print(f"[replay] clock anchored to {sim_start.strftime('%H:%M')}–"
          f"{sim_end.strftime('%H:%M')} (data tz) at {speed:.0f}x", flush=True)


def stop() -> None:
    global _ORIGINAL_QUOTE, _ORIGINAL_SCAN
    if _ORIGINAL_QUOTE is not None:
        market.quote = _ORIGINAL_QUOTE
        market.scan_full_market = _ORIGINAL_SCAN
        _ORIGINAL_QUOTE = None
        _ORIGINAL_SCAN = None
    _STATE.clear()


def sim_now() -> datetime | None:
    if not _STATE or _STATE.get("start_wall") is None:
        return None  # clock not anchored yet — nobody has looked up a ticker
    wall_now = _STATE.get("paused_at") or time.time()
    elapsed = (wall_now - _STATE["start_wall"]) * _STATE["speed"]
    return _STATE["sim_start"] + timedelta(seconds=elapsed)


def pause() -> None:
    """Freeze the sim clock's advance. Call before any real-time-expensive,
    non-market work (an agent's multi-turn LLM decision loop, in practice) so
    that latency there doesn't silently burn simulated market time.

    Without this, the clock keeps advancing at `speed`x for however long the
    LLM calls actually take in the real world — at 90x+ speeds, a single
    slow multi-agent tick can burn hours of simulated time before the tick
    even finishes, blowing straight past end-of-day and leaving every
    subsequent price lookup (including mark-to-market refreshes) frozen on
    the day's last bar for the rest of the session. It also means, within
    one nominal tick, agents queried later would see a different simulated
    instant than agents queried earlier — silently breaking "every agent
    trades the same round, at the same time." Pausing keeps the whole tick
    pinned to one consistent instant. No-op if the clock isn't anchored yet
    or is already paused.
    """
    if not _STATE or _STATE.get("start_wall") is None or _STATE.get("paused_at") is not None:
        return
    _STATE["paused_at"] = time.time()


def resume() -> None:
    """Undo pause(): shift the clock's origin forward by however long it was
    paused, so no simulated time is counted as having passed while paused."""
    if not _STATE or _STATE.get("paused_at") is None:
        return
    paused_for = time.time() - _STATE["paused_at"]
    _STATE["start_wall"] += paused_for
    _STATE["paused_at"] = None


def is_finished() -> bool:
    now = sim_now()
    return bool(now and _STATE and now >= _STATE["sim_end"])


def _replay_quote(ticker: str) -> dict | None:
    t = ticker.upper().strip()
    series = _STATE.get("series", {}).get(t)
    if not series and t not in _STATE.get("missing", set()) and _STATE.get("day"):
        got = _load_tickers([t], _STATE["day"])
        if got.get(t):
            _STATE["series"][t] = got[t]
            series = got[t]
            if _STATE.get("start_wall") is None:  # first ticker ever -> anchor now
                _anchor_clock(series[0][0], series[-1][0])
        else:
            _STATE["missing"].add(t)
    now = sim_now()
    if not series or not now:
        return _ORIGINAL_QUOTE(ticker) if _ORIGINAL_QUOTE else None
    price = _price_at(series, now)
    prev = series[0][1]
    return {"ticker": t, "price": price, "prev_close": prev,
            "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
            "asof": now.strftime("%H:%M")}


def _price_at(series: list[tuple[datetime, float]], now: datetime) -> float:
    """The latest bar price at or before `now`; falls back to the first bar."""
    price = None
    for dt, p in series:
        if dt.replace(tzinfo=None) <= now.replace(tzinfo=None):
            price = p
        else:
            break
    return price if price is not None else series[0][1]


def _replay_scan(n: int = 12, sample_size: int = 150,
                 seed: int | None = None) -> dict:
    """scan_full_market, but sourced from the REPLAY day, not today — so an
    agent's discovery stays inside the timeline it's actually trading in.
    A random sample each call, same as the live version."""
    from . import full_market
    from .market import TRADE_UNIVERSE_MODE
    day = _STATE.get("day")
    if not day:
        return {"gainers": [], "losers": [], "note": "replay not active"}
    sample = full_market.random_sample(sample_size, seed=seed,
                                       mode=TRADE_UNIVERSE_MODE)
    if not sample:
        return {"gainers": [], "losers": [],
                "note": "full-market universe unreachable this call"}
    cache = _STATE.setdefault("series", {})
    # Every agent scans its own independent random sample each tick, and
    # those samples overlap heavily by chance (150 of ~500 names, drawn 3x a
    # tick) — re-downloading a ticker another agent already fetched THIS
    # replay just adds pointless load that's part of what was tripping
    # Yahoo's throttling. A ticker's whole day of bars is already final and
    # cached the first time it's loaded, so reusing it is exact, not stale.
    to_fetch = [t for t in sample if t not in cache and t not in _STATE.get("missing", set())]
    fetched = _load_tickers(to_fetch, day) if to_fetch else {}
    for t, series in fetched.items():
        cache[t] = series
    _STATE.setdefault("missing", set()).update(t for t in to_fetch if t not in fetched)
    got = {t: cache[t] for t in sample if t in cache}
    if _STATE.get("start_wall") is None and got:
        first = next(iter(got.values()))
        _anchor_clock(first[0][0], first[-1][0])

    now = sim_now()
    rows = []
    for t, series in got.items():
        if len(series) < 2 or not now:
            continue
        price = _price_at(series, now)
        first_price = series[0][1]
        if first_price:
            rows.append({"ticker": t, "price": market._round_price(price),
                        "change_pct": round((price / first_price - 1) * 100, 2)})
    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return {"gainers": rows[:n], "losers": list(reversed(rows[-n:])),
            "universe_size": full_market.universe_size(TRADE_UNIVERSE_MODE),
            "sampled": len(sample)}

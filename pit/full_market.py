"""A genuine full-market ticker universe, for real discovery instead of memory.

Without this, an agent's only way to "find" a stock is to name one from its own
training memory — which reliably converges on the handful of most-famous names
(mega-cap tech, recent AI/crypto hype) regardless of what's actually moving.
That's not our bias, it's the underlying LLM's: those names dominate its
training data, so they're what it reaches for by default.

The fix is to give it something to genuinely SCAN: the real, comprehensive list
of every NASDAQ/NYSE/AMEX-listed common stock (thousands, not a hand-picked
25-50), then compute REAL current price movement over a random sample of it via
Yahoo at call time. Discovery becomes data-driven, not memory-driven.

The symbol list itself is pulled from a public GitHub mirror of NASDAQ's own
market-activity listings (nasdaqtrader.com's own servers are unreachable from
here) — only ticker/name/sector are used from it; price data always comes
fresh from Yahoo, never from this mirror (its snapshot recency is unverified).
"""
from __future__ import annotations

import csv
import io
import json
import random
import time
import urllib.request

_MIRROR_BASE = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"
_EXCHANGES = ["nasdaq", "nyse", "amex"]

_UNIVERSE_CACHE: dict = {"tickers": None, "loaded_at": 0.0}
_CACHE_TTL_SECONDS = 24 * 3600  # the listing itself barely changes day to day

# A real, published index (not hand-picked by us) — used as an optional
# QUALITY filter on top of the full universe: cuts illiquid/delisted penny
# names while keeping genuine sector diversity across ~500 companies, not
# just the 10 most-famous mega-caps. See config.trade_universe_mode.
_SP500_URL = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")
_SP500_CACHE: dict = {"tickers": None, "loaded_at": 0.0}

# Symbols that are technically "listed" but aren't ordinary common stock —
# warrants, rights, units, preferred shares, test issues. Filtered by name
# keyword since that's what the mirror gives us.
_EXCLUDE_KEYWORDS = ("warrant", "right", " unit", "preferred", "depositary",
                    "acquisition corp", "test stock", " notes", "trust pfd")


def _fetch_exchange(exchange: str) -> list[dict]:
    url = f"{_MIRROR_BASE}/{exchange}/{exchange}_full_tickers.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def full_universe(force_refresh: bool = False) -> list[dict]:
    """The full real ticker universe: [{"symbol", "name", "sector"}, ...].

    Cached in-process for a day (this doesn't change intraday). Returns
    whatever exchanges are reachable — if the mirror is down, returns [].
    """
    now = time.time()
    if (not force_refresh and _UNIVERSE_CACHE["tickers"] is not None
            and now - _UNIVERSE_CACHE["loaded_at"] < _CACHE_TTL_SECONDS):
        return _UNIVERSE_CACHE["tickers"]

    combined: dict[str, dict] = {}
    for ex in _EXCHANGES:
        try:
            rows = _fetch_exchange(ex)
        except Exception:
            continue
        for row in rows:
            sym = str(row.get("symbol", "")).strip().upper()
            name = str(row.get("name", ""))
            if not sym or any(c in sym for c in "./^~"):
                continue  # skip odd share-class / warrant-style symbols
            if any(kw in name.lower() for kw in _EXCLUDE_KEYWORDS):
                continue
            combined[sym] = {"symbol": sym, "name": name,
                             "sector": row.get("sector") or ""}

    tickers = list(combined.values())
    if tickers:  # only cache a non-empty result
        _UNIVERSE_CACHE["tickers"] = tickers
        _UNIVERSE_CACHE["loaded_at"] = now
    return tickers or (_UNIVERSE_CACHE["tickers"] or [])


def sp500_universe(force_refresh: bool = False) -> list[dict]:
    """The S&P 500 constituent list: [{"symbol", "name", "sector"}, ...].
    A real, publicly-maintained index — not us curating a shortlist — so it
    stays a genuine ~500-company discovery pool, just a quality-filtered one."""
    now = time.time()
    if (not force_refresh and _SP500_CACHE["tickers"] is not None
            and now - _SP500_CACHE["loaded_at"] < _CACHE_TTL_SECONDS):
        return _SP500_CACHE["tickers"]
    try:
        req = urllib.request.Request(_SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode()
    except Exception:
        return _SP500_CACHE["tickers"] or []

    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        sym = (row.get("Symbol") or "").strip().upper().replace(".", "-")
        if not sym:
            continue
        rows.append({"symbol": sym, "name": row.get("Security") or "",
                     "sector": row.get("GICS Sector") or ""})
    if rows:
        _SP500_CACHE["tickers"] = rows
        _SP500_CACHE["loaded_at"] = now
    return rows or (_SP500_CACHE["tickers"] or [])


def universe_for(mode: str) -> list[dict]:
    """mode: 'full' (every NASDAQ/NYSE/AMEX common stock, thousands of
    tickers, includes illiquid/delisted noise) or 'top500' (S&P 500
    constituents only — real and published, quality-filtered but still
    genuinely diverse). Falls back to 'full' if the S&P mirror is down."""
    if mode == "top500":
        pool = sp500_universe()
        return pool or full_universe()
    return full_universe()


def random_sample(n: int = 150, seed: int | None = None,
                  mode: str = "full") -> list[str]:
    """N random real ticker symbols from the chosen universe. A different
    sample each call (unless seeded) so discovery isn't stuck on one subset."""
    universe = universe_for(mode)
    if not universe:
        return []
    rng = random.Random(seed)
    n = min(n, len(universe))
    return [row["symbol"] for row in rng.sample(universe, n)]


def universe_size(mode: str = "full") -> int:
    return len(universe_for(mode))

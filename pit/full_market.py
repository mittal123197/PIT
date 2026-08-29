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

import json
import random
import time
import urllib.request

_MIRROR_BASE = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"
_EXCHANGES = ["nasdaq", "nyse", "amex"]

_UNIVERSE_CACHE: dict = {"tickers": None, "loaded_at": 0.0}
_CACHE_TTL_SECONDS = 24 * 3600  # the listing itself barely changes day to day

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


def random_sample(n: int = 150, seed: int | None = None) -> list[str]:
    """N random real ticker symbols from the full universe. A different sample
    each call (unless seeded) so discovery isn't stuck on the same subset."""
    universe = full_universe()
    if not universe:
        return []
    rng = random.Random(seed)
    n = min(n, len(universe))
    return [row["symbol"] for row in rng.sample(universe, n)]


def universe_size() -> int:
    return len(full_universe())

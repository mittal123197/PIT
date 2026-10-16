"""The tradeable universes: 'top100' (large US stocks) and 'crypto'.

top100 is a curated list of large, liquid US names (pit/top100.py); company
names and GICS sectors come from the published S&P 500 constituent list
(fetched, cached for a day). crypto is a fixed list of liquid pairs in Yahoo's
"-USD" form. Every agent sees the whole universe in its data table each
decision, so there is no sampling or scanning here.
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request

_SP500_URL = ("https://raw.githubusercontent.com/datasets/"
              "s-and-p-500-companies/main/data/constituents.csv")
_SP500_CACHE: dict = {"tickers": None, "loaded_at": 0.0}
_CACHE_TTL_SECONDS = 24 * 3600

_CRYPTO_UNIVERSE: list[dict] = [
    {"symbol": s, "name": s.replace("-USD", ""), "sector": "Crypto"} for s in (
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD",
        "ATOM-USD", "XLM-USD", "ETC-USD", "FIL-USD", "ARB-USD",
        "OP-USD", "NEAR-USD", "ICP-USD", "HBAR-USD", "VET-USD", "ALGO-USD",
        "AAVE-USD", "MKR-USD", "SAND-USD", "MANA-USD", "EOS-USD", "XTZ-USD",
        "TRX-USD", "SHIB-USD",
    )
]


def sp500_universe(force_refresh: bool = False) -> list[dict]:
    """S&P 500 constituents [{"symbol", "name", "sector"}] — names and sectors
    for the top-100 list. Cached for a day; [] if unreachable."""
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
        if sym:
            rows.append({"symbol": sym, "name": row.get("Security") or "",
                         "sector": row.get("GICS Sector") or ""})
    if rows:
        _SP500_CACHE.update(tickers=rows, loaded_at=now)
    return rows or (_SP500_CACHE["tickers"] or [])


def universe_for(mode: str) -> list[dict]:
    """[{"symbol", "name", "sector"}] for 'top100' or 'crypto'."""
    if mode == "crypto":
        return _CRYPTO_UNIVERSE
    from .top100 import TOP100
    sp = {r["symbol"]: r for r in sp500_universe()}
    return [{"symbol": t, "name": sp.get(t, {}).get("name", ""),
             "sector": sp.get(t, {}).get("sector", "")} for t in TOP100]

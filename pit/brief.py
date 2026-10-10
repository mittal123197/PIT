"""The market brief: ONE complete, pre-computed data table (fundamentals +
technicals) for the whole tradeable universe, built once per tick and shared
by every agent.

This replaces the multi-turn "research" loop (scan -> history -> fundamentals,
one tool call at a time) for small universes. A trader looking at a screen of
100 names doesn't query them one by one — they see the whole table and make a
call. One LLM call per decision, every agent sees the identical data.

Used when the universe is `top100` (a curated list of ~100 large US names) or
`crypto` (fundamentals don't exist there — technicals only). Data is real
yfinance: daily bars for technicals, `market.fundamentals` for fundamentals
(cached on disk for a day — they barely move intraday).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import full_market, market, risk

_FUND_PATH = Path(__file__).resolve().parent.parent / "data" / "fundamentals_cache.json"
_FUND_TTL = 24 * 3600
_BRIEF = {"at": 0.0, "mode": None, "value": None}
BRIEF_MODES = ("top100", "crypto")


def brief_mode(mode: str) -> bool:
    return mode in BRIEF_MODES and os.getenv("PIT_BRIEF", "1") != "0"


def symbols_for(mode: str) -> list[str]:
    return [r["symbol"] for r in full_market.universe_for(mode)]


# ---- fundamentals (disk-cached) ----------------------------------------

def _load_fund() -> dict:
    try:
        return json.loads(_FUND_PATH.read_text())
    except Exception:
        return {}


def _save_fund(d: dict) -> None:
    try:
        _FUND_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FUND_PATH.write_text(json.dumps(d))
    except Exception:
        pass


def prefetch_fundamentals(symbols: list[str], verbose: bool = True) -> None:
    """Fill the on-disk fundamentals cache for anything missing/stale. Run once
    at session start so the first tick isn't slowed by ~100 slow info calls."""
    cache = _load_fund()
    now = time.time()
    todo = [s for s in symbols if now - cache.get(s, {}).get("at", 0) > _FUND_TTL]
    if not todo:
        return
    if verbose:
        print(f"[brief] fetching fundamentals for {len(todo)} tickers "
              f"(cached for 24h)...", flush=True)
    # few workers: yfinance bulk/threaded bursts get throttled (see replay.py)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for s, f in zip(todo, ex.map(market.fundamentals, todo)):
            cache[s] = {"at": now, "f": f}
    _save_fund(cache)


# ---- technicals ----------------------------------------------------------

def _pct(a, b):
    return None if not b else round((a / b - 1) * 100, 2)


def _rsi14(close) -> float | None:
    d = close.diff().dropna()
    if len(d) < 15:
        return None
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    return 100.0 if dn == 0 else round(100 - 100 / (1 + up / dn), 1)


def _technicals(close, volume) -> dict | None:
    close = close.dropna()
    if len(close) < 22:
        return None
    px = float(close.iloc[-1])
    out = {"px": px, "d1": _pct(px, float(close.iloc[-2])),
           "d5": _pct(px, float(close.iloc[-6])) if len(close) > 6 else None,
           "d20": _pct(px, float(close.iloc[-21]))}
    sma20 = float(close.iloc[-20:].mean())
    out["v20"] = _pct(px, sma20)
    out["v50"] = _pct(px, float(close.iloc[-50:].mean())) if len(close) >= 50 else None
    out["rsi"] = _rsi14(close)
    # 20-day std of daily returns (%) — drives the volatility-scaled stops
    rets = close.pct_change().dropna().iloc[-20:]
    out["dvol"] = round(float(rets.std()) * 100, 2) if len(rets) >= 10 else None
    hi, lo = float(close.iloc[-252:].max()), float(close.iloc[-252:].min())
    out["p52"] = round((px - lo) / (hi - lo) * 100) if hi > lo else None
    try:
        vol = volume.dropna()
        out["volx"] = round(float(vol.iloc[-1]) / float(vol.iloc[-21:-1].mean()), 2)
    except Exception:
        out["volx"] = None
    return out


def _download_daily(symbols: list[str]):
    import yfinance as yf
    frames = []
    for i in range(0, len(symbols), 50):        # chunks, sequential: gentler on Yahoo
        chunk = symbols[i:i + 50]
        df = yf.download(chunk, period="1y", interval="1d", progress=False,
                         auto_adjust=True, timeout=30, threads=False)
        if df is not None and not df.empty:
            frames.append(df)
    return frames


# ---- the table -----------------------------------------------------------

_HEADER = ("TICKER|sector|price|1d%|5d%|20d%|vs20dMA%|vs50dMA%|RSI14|52wk-pos%|"
           "vol-x|dayVol%|autoStop%|fwdPE|margin%|revGr%|epsGr%|ROE%|D/E|analyst|tgt-upside%")

# the data-table columns an agent-written rule may test, -> technicals key.
# Shared with guidelines.py so a rule's condition is checkable against the
# exact numbers the agent saw when it traded.
RULE_FIELDS = {"1d%": "d1", "5d%": "d5", "20d%": "d20", "vs20dMA%": "v20",
               "vs50dMA%": "v50", "RSI14": "rsi", "52wk-pos%": "p52",
               "vol-x": "volx", "dayVol%": "dvol"}


def _f(v, nd=1):
    if v is None:
        return "."
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _row(sym: str, sector: str, t: dict, f: dict | None) -> str:
    f = f or {}
    tgt = f.get("analyst_target_price")
    up = _pct(tgt, t["px"]) if tgt else None
    px = t["px"]
    price = f"{px:,.2f}" if px >= 1 else f"{px:.6g}"
    return "|".join([
        sym, (sector or ".")[:11], price, _f(t["d1"], 2), _f(t["d5"]), _f(t["d20"]),
        _f(t["v20"]), _f(t["v50"]), _f(t["rsi"]), _f(t["p52"], 0), _f(t["volx"], 2),
        _f(t.get("dvol"), 2),
        _f(risk.auto_stop(sym, 0.0) or None, 2) if risk.active() else ".",
        _f(f.get("forward_pe")), _f(f.get("profit_margin_pct"), 0),
        _f(f.get("revenue_growth_pct"), 0), _f(f.get("earnings_growth_pct"), 0),
        _f(f.get("return_on_equity_pct"), 0), _f(f.get("debt_to_equity"), 0),
        (f.get("analyst_recommendation") or ".")[:10], _f(up, 0)])


def get_brief(mode: str, max_age: float = 60.0) -> dict | None:
    """{"text": table, "prices": {sym: px}, "features": {sym: technicals},
    "n": count, "asof": ts} — cached for `max_age` seconds, so every agent in
    a tick shares one build. The text is re-rendered on every call (cheap) so
    the autoStop% column follows the current round's risk settings."""
    now = time.time()
    if not (_BRIEF["value"] and _BRIEF["mode"] == mode and now - _BRIEF["at"] < max_age):
        built = _build(mode)
        if not built:
            return None
        _BRIEF.update(at=now, mode=mode, value=built)
    v = _BRIEF["value"]
    rows = [_row(s, sec, t, f) for s, sec, t, f in v["rows"]]
    return {"text": _HEADER + "\n" + "\n".join(rows), "prices": v["prices"],
            "features": v["features"], "n": len(rows), "asof": v["asof"]}


def _build(mode: str) -> dict | None:
    symbols = symbols_for(mode)
    sectors = {r["symbol"]: r.get("sector", "") for r in full_market.universe_for(mode)}
    fund = _load_fund()
    rows, prices, features = [], {}, {}
    for df in _download_daily(symbols):
        close, vol = df["Close"], df["Volume"]
        for s in symbols:
            if s in prices or s not in close.columns:
                continue
            t = _technicals(close[s], vol[s] if s in vol.columns else close[s] * 0)
            if not t:
                continue
            prices[s] = t["px"]
            features[s] = {k: t.get(k) for k in RULE_FIELDS.values()}
            rows.append((s, sectors.get(s, ""), t, (fund.get(s) or {}).get("f")))
    if not rows:
        return None
    return {"rows": rows, "prices": prices, "features": features,
            "asof": time.strftime("%Y-%m-%d %H:%M:%S")}

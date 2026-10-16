"""Replay a past trading day at compressed speed, from the market history.

The day comes from pit/history.py's panel — the same data-table columns the
live brief shows, for every top-100 stock at every 5-minute bar of the last
~60 trading days (daily columns use only closes BEFORE that day, so there is
no lookahead). A wall-clock timer advances a simulated clock through the
session; at any moment the replay serves:

  * prices  — market.quote() is patched to the latest bar at the sim clock,
              so fills, marks and stops all see the replayed day;
  * brief() — the agents' data table as it stood at that bar;
  * vols()  — each name's daily volatility, for the round's risk bands.

Agents therefore trade a replay exactly as they trade live: one table, one
decision per wake-up, charters checkable, rulebook evidence recorded.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

from . import market

_ORIGINAL_QUOTE = None
_STATE: dict = {}
INDEXES = ("SPY", "QQQ")




def start(day: date | None = None, compress_minutes: int | None = None) -> None:
    """Begin replaying `day` (default: the last completed session in the
    history), compressing the session into `compress_minutes` of wall clock."""
    global _ORIGINAL_QUOTE
    from . import history
    df = history.panel()
    if df is None:
        raise RuntimeError("replay needs the US top-100 market history "
                           "(PIT_MARKET=us, PIT_TRADE_UNIVERSE=top100)")
    days = sorted(df["day"].unique())
    if day is None:
        today = datetime.now().date()
        day = next(d for d in reversed(days) if d < today)
    if day not in days:
        raise RuntimeError(f"{day} isn't in the market history "
                           f"({days[0]} … {days[-1]}, the last ~60 sessions)")
    d = df[df["day"] == day]
    bars = sorted(d["ts"].unique())
    compress = compress_minutes or 20
    sim_start, sim_end = bars[0], bars[-1] + timedelta(minutes=5)
    _STATE.clear()
    _STATE.update({
        "day": day, "bars": bars, "rows": d.set_index(["ts", "sym"]).sort_index(),
        "sim_start": sim_start, "sim_end": sim_end, "start_wall": time.time(),
        "speed": (sim_end - sim_start).total_seconds() / max(60, compress * 60),
        "brief_cache": {}})
    if _ORIGINAL_QUOTE is None:
        _ORIGINAL_QUOTE = market.quote
    market.quote = _replay_quote  # type: ignore
    print(f"[replay] {day}: {d['sym'].nunique()} names, {len(bars)} bars, "
          f"session compressed to ~{compress} wall-clock minutes "
          f"({_STATE['speed']:.0f}x)", flush=True)


def stop() -> None:
    global _ORIGINAL_QUOTE
    if _ORIGINAL_QUOTE is not None:
        market.quote = _ORIGINAL_QUOTE
        _ORIGINAL_QUOTE = None
    _STATE.clear()


def sim_now() -> datetime | None:
    if not _STATE:
        return None
    t = _STATE["sim_start"] + timedelta(
        seconds=(time.time() - _STATE["start_wall"]) * _STATE["speed"])
    return min(t, _STATE["sim_end"])


def _bar() -> datetime | None:
    """The latest completed bar at the sim clock."""
    now = sim_now()
    if now is None:
        return None
    done = [b for b in _STATE["bars"] if b <= now]
    return done[-1] if done else _STATE["bars"][0]


def _row(sym: str):
    try:
        return _STATE["rows"].loc[(_bar(), sym)]
    except KeyError:
        return None


def _replay_quote(ticker: str) -> dict | None:
    r = _row(ticker.upper().strip())
    if r is None:
        return None
    px = float(r["px"])
    prev = px / (1 + r["d1"] / 100) if r["d1"] == r["d1"] else px
    return {"ticker": ticker.upper(), "price": px, "prev_close": prev,
            "change_pct": round(float(r["d1"]), 2) if r["d1"] == r["d1"] else 0.0,
            "asof": _bar().strftime("%H:%M")}


def vols() -> dict[str, float]:
    """Each name's daily volatility (%) going into the replayed day."""
    first = _STATE["rows"].loc[_STATE["bars"][0]]
    return {s: float(v) for s, v in first["dvol"].items() if v == v and s not in INDEXES}


def brief() -> dict | None:
    """The agents' data table as it stood at the current bar (cached per bar)."""
    from . import brief as brief_mod, full_market
    bar = _bar()
    if bar is None:
        return None
    cached = _STATE["brief_cache"].get(bar)
    if cached:
        return cached
    at = _STATE["rows"].loc[bar]
    fund = brief_mod._load_fund()
    sectors = {r["symbol"]: r.get("sector", "") for r in full_market.universe_for("top100")}
    rows, prices, features, intra = [], {}, {}, {}
    for sym, r in at.iterrows():
        t = {k: (None if v is None or v != v else float(v)) for k, v in r.items()
             if k not in ("sym", "day", "bar")}
        intra[sym] = t
        if sym in INDEXES:
            continue
        f = (fund.get(sym) or {}).get("f")
        prices[sym] = t["px"]
        features[sym] = {k: t.get(k) for k in brief_mod.RULE_FIELDS.values()}
        rows.append(brief_mod._row(sym, sectors.get(sym, ""), t, f))
    value = {"text": brief_mod._HEADER + "\n" + "\n".join(rows), "prices": prices,
             "features": features, "n": len(rows),
             "asof": f"{_STATE['day']} {bar.strftime('%H:%M')} ET (replay)",
             "market": brief_mod._market_line(intra, "top100")}
    _STATE["brief_cache"][bar] = value
    return value


# ---- the replay clock, as agents and the dashboard should see it ----------

def market_ts(wall_ts: str | None) -> str | None:
    """A stored wall-clock stamp ("YYYY-mm-dd HH:MM:SS", local) as the replayed
    session's own time ("2026-10-09 09:49 ET"). Fills are stamped in wall
    time; in a 10-minute replay an opening buy otherwise read as "15:49" and an
    agent concluded it had bought "before the close". None outside a replay."""
    if not wall_ts or not _STATE:
        return None
    try:
        wall = time.mktime(datetime.strptime(wall_ts[:19], "%Y-%m-%d %H:%M:%S").timetuple())
    except ValueError:
        return None
    sim = _STATE["sim_start"] + timedelta(seconds=max(0.0, wall - _STATE["start_wall"])
                                          * _STATE["speed"])
    return _et(min(sim, _STATE["sim_end"])).strftime("%Y-%m-%d %H:%M ET")


def _et(d: datetime) -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return d.astimezone(ZoneInfo("America/New_York")) if d.tzinfo else d
    except Exception:
        return d


def anchor() -> dict | None:
    """The replay clock as plain data (ET, naive) for the dashboard."""
    if not _STATE:
        return None
    return {"start_wall": _STATE["start_wall"], "speed": _STATE["speed"],
            "sim_start": _et(_STATE["sim_start"]).strftime("%Y-%m-%d %H:%M:%S"),
            "sim_end": _et(_STATE["sim_end"]).strftime("%Y-%m-%d %H:%M:%S")}


def agent_ts(wall_ts: str | None) -> str | None:
    """The time an agent should see: market time in a replay, else as stored."""
    return market_ts(wall_ts) or wall_ts


def sim_minutes_left() -> float | None:
    now = sim_now()
    if now is None:
        return None
    return max(0.0, round((_STATE["sim_end"] - now).total_seconds() / 60, 1))

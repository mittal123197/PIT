"""Market history for research: the agents' data table, rebuilt for every
stock at every 5-minute bar of the last ~60 trading days.

Rules and charters used to be judged only on the pool's own trades (~50
fills over a week) — almost everything came back "untested". Here the arena
rebuilds the SAME columns the agents read (open%, vsVWAP%, 15m%, 30m%,
dayRange%, volPace + the daily view: 1d/5d/20d%, vs20/50dMA%, RSI14,
52wk-pos%, vol-x, dayVol%) as of each bar, plus what the price actually did
over the next 30 / 60 minutes. ~450k snapshots, so a rule or a charter can be
tested on thousands of real market moments before anyone trades on it.

Cached on disk for the day (Yahoo serves 60 days of 5-minute bars). Daily
features use only closes BEFORE the bar's day plus the bar's own price; RSI14
uses prior closes only (an approximation of the live column).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import brief

_DIR = Path(__file__).resolve().parent.parent / "data" / "history"
_PANEL: dict = {"day": None, "df": None}
SPREAD_BPS = 3.0
MIN_DAYS = 10            # days with data on both sides before a verdict
T_SUPPORT, T_CONTRA = 2.0, 1.0


# ---- building the panel --------------------------------------------------

def _download(symbols: list[str], **kw):
    import yfinance as yf
    frames = []
    for i in range(0, len(symbols), 25):
        df = yf.download(symbols[i:i + 25], progress=False, auto_adjust=True,
                         threads=False, timeout=60, **kw)
        if df is not None and not df.empty:
            frames.append(df)
    return frames


def _col(frames, field, sym):
    for df in frames:
        try:
            if sym in df[field].columns:
                return df[field][sym]
        except KeyError:
            continue
    return None


def _rsi_series(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _symbol_panel(sym: str, intra: pd.DataFrame, daily_close: pd.Series,
                  daily_vol: pd.Series) -> pd.DataFrame | None:
    df = intra.dropna(subset=["Close"]).copy()
    if df.empty or daily_close is None or daily_close.dropna().empty:
        return None
    df.index = df.index.tz_convert("America/New_York")
    df = df.between_time("09:30", "15:55")
    if df.empty:
        return None
    day = pd.Series(df.index.date, index=df.index)
    g = df.groupby(day.values)
    C = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["sym"] = sym
    out["day"] = day.values
    out["bar"] = g.cumcount().values
    out["px"] = C.values
    # today's session
    day_open = g["Open"].transform("first")
    out["o"] = (C / day_open - 1) * 100
    tp = (df["High"] + df["Low"] + C) / 3
    cum_pv = (tp * df["Volume"]).groupby(day.values).cumsum()
    cum_v = df["Volume"].groupby(day.values).cumsum()
    out["vwap"] = (C / (cum_pv / cum_v.replace(0, np.nan)) - 1) * 100
    out["m15"] = (C / g["Close"].shift(3) - 1) * 100
    out["m30"] = (C / g["Close"].shift(6) - 1) * 100
    hi = g["High"].cummax()
    lo = g["Low"].cummin()
    out["rng"] = ((C - lo) / (hi - lo).replace(0, np.nan) * 100).round()
    # forward returns (same session only)
    out["f30"] = (g["Close"].shift(-6) / C - 1) * 100
    out["f60"] = (g["Close"].shift(-12) / C - 1) * 100

    # the daily view, from closes BEFORE each day + the bar's own price
    dc = daily_close.dropna()
    dc.index = pd.to_datetime(dc.index).date
    dv = daily_vol.reindex(daily_close.index).fillna(0)
    dv.index = pd.to_datetime(dv.index).date
    prior = pd.DataFrame({
        "prev": dc.shift(1), "c5": dc.shift(5), "c20": dc.shift(20),
        "s19": dc.shift(1).rolling(19).sum(), "s49": dc.shift(1).rolling(49).sum(),
        "hi251": dc.shift(1).rolling(251, min_periods=60).max(),
        "lo251": dc.shift(1).rolling(251, min_periods=60).min(),
        "rsi": _rsi_series(dc).shift(1),
        "dvol": dc.pct_change().shift(1).rolling(20).std() * 100,
        "avgvol": dv.shift(1).rolling(20).mean(),
    })
    pr = prior.reindex(out["day"].values)
    pr.index = out.index
    px = out["px"]
    out["d1"] = (px / pr["prev"] - 1) * 100
    out["d5"] = (px / pr["c5"] - 1) * 100
    out["d20"] = (px / pr["c20"] - 1) * 100
    out["v20"] = (px / ((pr["s19"] + px) / 20) - 1) * 100
    out["v50"] = (px / ((pr["s49"] + px) / 50) - 1) * 100
    out["rsi"] = pr["rsi"]
    hi52 = np.maximum(pr["hi251"], px)
    lo52 = np.minimum(pr["lo251"], px)
    out["p52"] = ((px - lo52) / (hi52 - lo52).replace(0, np.nan) * 100).round()
    out["dvol"] = pr["dvol"]
    elapsed = (out["bar"] + 1) * 5
    out["volx"] = cum_v / pr["avgvol"].replace(0, np.nan)
    out["pace"] = cum_v / (pr["avgvol"] * np.minimum(1.0, elapsed / 390)).replace(0, np.nan)
    return out


def build_panel(mode: str = "top100", verbose: bool = True) -> pd.DataFrame | None:
    symbols = brief.symbols_for(mode)
    t = time.time()
    intra = _download(symbols, period="60d", interval="5m")
    daily = _download(symbols, period="2y", interval="1d")
    parts = []
    for s in symbols:
        fields = {f: _col(intra, f, s) for f in ("Open", "High", "Low", "Close", "Volume")}
        if any(v is None for v in fields.values()):
            continue
        p = _symbol_panel(s, pd.DataFrame(fields), _col(daily, "Close", s),
                          _col(daily, "Volume", s))
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    df = pd.concat(parts).reset_index(names="ts")
    if verbose:
        print(f"[history] built {len(df):,} snapshots for {df['sym'].nunique()} names "
              f"over {df['day'].nunique()} days in {time.time() - t:.0f}s", flush=True)
    return df


def panel(mode: str = "top100", verbose: bool = True) -> pd.DataFrame | None:
    """Today's cached panel (built once a day)."""
    import os
    if mode != "top100" or os.getenv("PIT_HISTORY", "1") == "0":
        return None              # only the US top-100 table has a matching history
    today = time.strftime("%Y-%m-%d")
    if _PANEL["day"] == today and _PANEL["df"] is not None:
        return _PANEL["df"]
    path = _DIR / f"panel_{mode}_{today}.pkl"
    df = None
    if path.exists():
        try:
            df = pd.read_pickle(path)
        except Exception:
            df = None
    if df is None:
        try:
            df = build_panel(mode, verbose)
        except Exception as exc:
            if verbose:
                print(f"[history] build failed: {exc!r}", flush=True)
            return None
        if df is None:
            return None
        _DIR.mkdir(parents=True, exist_ok=True)
        for old in _DIR.glob(f"panel_{mode}_*.pkl"):
            old.unlink(missing_ok=True)
        df.to_pickle(path)
    _PANEL.update(day=today, df=df)
    return df


# ---- conditions ------------------------------------------------------------

def _fields():
    return brief.RULE_FIELDS


def _mask(df: pd.DataFrame, when: list) -> pd.Series | None:
    """Boolean mask for [[field, op, value], ...] over table columns; None if a
    condition uses something the history can't see (e.g. pos_pnl%)."""
    f = _fields()
    m = pd.Series(True, index=df.index)
    for name, op, v in when:
        key = f.get(name)
        if key is None or key not in df.columns:
            return None
        col = df[key]
        m &= {"<": col < v, "<=": col <= v, ">": col > v, ">=": col >= v}[op] & col.notna()
    return m


# ---- rule research ---------------------------------------------------------

def research_rule(spec: dict, practice: str, df: pd.DataFrame | None = None) -> dict | None:
    """Test a rulebook rule on the market history. Entry rules compare the
    next-30-minute return of snapshots that match the condition vs. those that
    don't; exit rules ('sell when …') score a sell as avoiding the next 30
    minutes. Day-clustered: one edge per trading day, t across days (all
    stocks on a given day move together, so bars aren't independent)."""
    df = panel() if df is None else df
    if df is None:
        return None
    sample = df[(df["bar"] % 6 == 0) & df["f30"].notna()]          # non-overlapping 30-min steps
    m = _mask(sample, spec["when"])
    if m is None:
        return None
    x = sample["f30"] if spec.get("action", "buy") == "buy" else -sample["f30"]
    sign = 1 if practice == "good" else -1
    daily = []
    for d, idx in sample.groupby("day").groups.items():
        mm = m.loc[idx]
        if mm.sum() >= 3 and (~mm).sum() >= 3:
            xs = x.loc[idx]
            daily.append(sign * (xs[mm].mean() - xs[~mm].mean()))
    xm, xo = x[m], x[~m]
    ev = {"source": "history", "unit": "market snapshots",
          "n_match": int(m.sum()), "n_other": int((~m).sum()), "days": len(daily),
          "avg_match": round(float(xm.mean()), 3) if len(xm) else None,
          "avg_other": round(float(xo.mean()), 3) if len(xo) else None,
          "win_rate_match": round(float((xm > 0).mean() * 100)) if len(xm) else None}
    if len(daily) < MIN_DAYS:
        ev.update(t=None, verdict="untested")
        return ev
    arr = np.array(daily)
    se = arr.std(ddof=1) / np.sqrt(len(arr)) or 1e-9
    t = round(float(arr.mean() / se), 2)
    ev.update(t=t, edge_per_day=round(float(arr.mean()), 3),
              verdict="supported" if t >= T_SUPPORT else "contradicted" if t <= -T_CONTRA
              else "inconclusive")
    return ev


# ---- charter backtest ------------------------------------------------------

def split(df: pd.DataFrame, train_frac: float = 2 / 3):
    """(train, test) by trading day — the charter search only ever sees the
    earlier days; the final charter is scored on the later, unseen ones so
    repeated refinement can't just curve-fit the history."""
    days = sorted(df["day"].unique())
    cut = days[int(len(days) * train_frac)]
    return df[df["day"] < cut], df[df["day"] >= cut]


def backtest_charter(spec: dict, df: pd.DataFrame | None = None) -> dict | None:
    """Replay a charter's entry/exit conditions over the history: enter at a
    bar where every entry condition holds, exit at the first later bar of the
    same session where any exit condition holds (exits may use pos_pnl% and
    held_min), after max_hold_min, or at the session's last bar. Each round
    trip pays the spread on both sides. One position per name at a time."""
    df = panel() if df is None else df
    if df is None or not spec or not spec.get("entry"):
        return None
    entry = _mask(df, spec["entry"])
    if entry is None:
        return None
    f = _fields()
    exits = spec.get("exit") or []
    max_hold = float(spec.get("max_hold_min") or 60)
    cost = 2 * SPREAD_BPS / 100                                # % round trip
    cols = {k: df[k].to_numpy() for k in set(f.values()) if k in df.columns}
    px = df["px"].to_numpy()
    ent = entry.to_numpy()
    trades = []                                                # (day, ret%, hold_min)
    for (_, d), idx in df.groupby(["sym", "day"], sort=False).indices.items():
        i, n = 0, len(idx)
        while i < n - 1:
            r = idx[i]
            if not ent[r]:
                i += 1
                continue
            p0 = px[r]
            j = i + 1
            while j < n:
                rj = idx[j]
                held = (j - i) * 5
                pnl = (px[rj] / p0 - 1) * 100
                hit = held >= max_hold or j == n - 1
                for name, op, v in exits:
                    if name == "pos_pnl%":
                        val = pnl
                    elif name == "held_min":
                        val = held
                    else:
                        k = f.get(name)
                        val = cols[k][rj] if k in cols else np.nan
                    if val == val and {"<": val < v, "<=": val <= v,
                                       ">": val > v, ">=": val >= v}[op]:
                        hit = True
                        break
                if hit:
                    trades.append((d, pnl - cost, held))
                    break
                j += 1
            i = j + 1
    days = df["day"].nunique()
    if not trades:
        return {"days": days, "trades": 0, "setups_per_day": 0.0,
                "verdict": "never triggers", "t": None}
    t_df = pd.DataFrame(trades, columns=["day", "ret", "hold"])
    per_day = t_df.groupby("day")["ret"].mean()
    t = None
    if len(per_day) >= 5:
        se = per_day.std(ddof=1) / np.sqrt(len(per_day)) or 1e-9
        t = round(float(per_day.mean() / se), 2)
    return {"days": days, "trades": len(t_df),
            "setups_per_day": round(len(t_df) / days, 1),
            "avg_trade_pct": round(float(t_df["ret"].mean()), 3),
            "win_rate": round(float((t_df["ret"] > 0).mean() * 100)),
            "avg_hold_min": round(float(t_df["hold"].mean())),
            "t": t, "verdict": ("edge" if t is not None and t >= T_SUPPORT else
                                "losing" if t is not None and t <= -T_SUPPORT else
                                "no clear edge")}


def card_text(card: dict | None) -> str:
    if not card:
        return "not backtested"
    if "train" in card:     # search card: in-sample + out-of-sample
        tr, te = card.get("train"), card.get("test")
        return (f"search days: {card_text(tr)} | unseen later days: {card_text(te)}")
    if not card.get("trades"):
        return f"never triggered in {card.get('days', '?')} days of history"
    return (f"{card['days']}-day backtest: {card['setups_per_day']} trades/day, avg "
            f"{card['avg_trade_pct']:+.2f}% per trade after costs, win {card['win_rate']}%, "
            f"avg hold {card['avg_hold_min']}m, t={card['t']:+.2f} → {card['verdict']}"
            if card.get("t") is not None else
            f"{card['days']}-day backtest: {card['trades']} trades, avg "
            f"{card['avg_trade_pct']:+.2f}% — too few trading days to judge")

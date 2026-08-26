"""Price feeds — the arena's clock and market.

A feed is a finite sequence of bars: (timestamp, {symbol: price}). The round
engine steps through it; each bar is one orchestrator tick. Timestamps are real
(so heartbeat cadence and watch thresholds mean something), while the whole
sequence can be replayed far faster than wall-clock ("a week in minutes").

Two implementations:
  * SyntheticFeed  — deterministic geometric-brownian paths, zero network.
                     The default: makes the whole arena runnable offline and
                     makes tests reproducible.
  * HistoricalFeed — real NSE bars via yfinance, cached to CSV. Opt-in, for
                     realistic replay. Falls back cleanly if the network or
                     yfinance is unavailable.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

from .config import DEFAULT_UNIVERSE


class PriceFeed(ABC):
    symbols: list[str]

    @abstractmethod
    def now(self) -> datetime: ...

    @abstractmethod
    def prices(self) -> dict[str, float]: ...

    @abstractmethod
    def advance(self) -> bool:
        """Step to the next bar. Return False when the feed is exhausted."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of bars (used for progress + goal scaling)."""


class SyntheticFeed(PriceFeed):
    """Deterministic GBM price paths. Same seed => identical run."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        bars: int = 120,
        start: datetime | None = None,
        bar_minutes: int = 15,
        seed: int = 42,
        annual_drift: float = 0.08,
        annual_vol: float = 0.30,
        start_price_range: tuple[float, float] = (500.0, 3500.0),
    ) -> None:
        self.symbols = list(symbols or DEFAULT_UNIVERSE)
        self.bar_minutes = bar_minutes
        rng = random.Random(seed)
        start = start or (datetime(2024, 1, 1, 9, 15))

        # per-bar drift/vol from annualised figures (~252*25 fifteen-min bars/yr)
        bars_per_year = 252 * (375 / bar_minutes)
        mu = annual_drift / bars_per_year
        sigma = annual_vol / (bars_per_year ** 0.5)

        self._timestamps: list[datetime] = [
            start + timedelta(minutes=bar_minutes * i) for i in range(bars)
        ]
        self._paths: dict[str, list[float]] = {}
        for sym in self.symbols:
            price = rng.uniform(*start_price_range)
            path = [round(price, 2)]
            for _ in range(bars - 1):
                shock = rng.gauss(mu, sigma)
                price = max(1.0, price * (1.0 + shock))
                path.append(round(price, 2))
            self._paths[sym] = path

        self._cursor = 0

    def now(self) -> datetime:
        return self._timestamps[self._cursor]

    def prices(self) -> dict[str, float]:
        return {sym: self._paths[sym][self._cursor] for sym in self.symbols}

    def advance(self) -> bool:
        if self._cursor >= len(self._timestamps) - 1:
            return False
        self._cursor += 1
        return True

    def __len__(self) -> int:
        return len(self._timestamps)


class HistoricalFeed(PriceFeed):
    """Real NSE bars via yfinance, cached to CSV under `data/`.

    Import of yfinance/pandas is deferred so the package works without them.
    If a download fails, raises RuntimeError — the CLI catches it and suggests
    the synthetic feed instead.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        period: str = "1mo",
        interval: str = "15m",
        cache_dir: str | None = None,
    ) -> None:
        self.symbols = list(symbols or DEFAULT_UNIVERSE)
        self.period = period
        self.interval = interval
        self.cache_dir = Path(
            cache_dir or (Path(__file__).resolve().parent.parent / "data" / "feed_cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._timestamps, self._paths = self._load()
        self._cursor = 0

    def _load(self):
        try:
            import pandas as pd
            import yfinance as yf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "HistoricalFeed needs pandas + yfinance. Use SyntheticFeed, or "
                "`pip install -r requirements.txt`."
            ) from exc

        frames: dict[str, "pd.Series"] = {}
        for sym in self.symbols:
            cache = self.cache_dir / f"{sym.replace('/', '_')}_{self.period}_{self.interval}.csv"
            series = None
            if cache.exists():
                try:
                    df = pd.read_csv(cache, index_col=0, parse_dates=True)
                    series = df["close"]
                except Exception:
                    series = None
            if series is None:
                df = yf.download(
                    sym, period=self.period, interval=self.interval,
                    progress=False, auto_adjust=True,
                )
                if df is None or df.empty:
                    raise RuntimeError(f"no data returned for {sym}")
                close = df["Close"]
                if hasattr(close, "columns"):  # multiindex when >1 symbol
                    close = close.iloc[:, 0]
                series = close.dropna()
                series.name = "close"
                series.to_frame().to_csv(cache)
            frames[sym] = series

        # align all symbols on their common timestamps
        common = None
        for series in frames.values():
            idx = set(series.index)
            common = idx if common is None else (common & idx)
        timestamps = sorted(common or [])
        if len(timestamps) < 2:
            raise RuntimeError("not enough overlapping bars across symbols")

        paths = {
            sym: [float(frames[sym].loc[ts]) for ts in timestamps]
            for sym in self.symbols
        }
        return list(timestamps), paths

    def now(self) -> datetime:
        return self._timestamps[self._cursor]

    def prices(self) -> dict[str, float]:
        return {sym: self._paths[sym][self._cursor] for sym in self.symbols}

    def advance(self) -> bool:
        if self._cursor >= len(self._timestamps) - 1:
            return False
        self._cursor += 1
        return True

    def __len__(self) -> int:
        return len(self._timestamps)

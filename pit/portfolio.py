"""In-memory per-agent state during a round.

The round engine holds one Portfolio per agent while a round runs, writing
trades to the DB as they happen and persisting the final state at resolution.
Keeping it in memory during the round keeps the tick loop cheap; the DB stays
the source of truth for anything outside the round.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    starting_capital: float
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> qty
    trade_count: int = 0
    status: str = "active"           # active | liquidated
    peak_value: float = 0.0          # high-water mark for drawdown
    max_drawdown_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_value == 0.0:
            self.peak_value = self.starting_capital

    @classmethod
    def open(cls, capital: float) -> "Portfolio":
        return cls(starting_capital=capital, cash=capital, peak_value=capital)

    def holdings_value(self, prices: dict[str, float]) -> float:
        return sum(qty * prices.get(sym, 0.0) for sym, qty in self.positions.items())

    def total_value(self, prices: dict[str, float]) -> float:
        return self.cash + self.holdings_value(prices)

    def return_pct(self, prices: dict[str, float]) -> float:
        return (self.total_value(prices) / self.starting_capital - 1.0) * 100.0

    def mark(self, prices: dict[str, float]) -> None:
        """Update the high-water mark and max drawdown at the current prices."""
        value = self.total_value(prices)
        if value > self.peak_value:
            self.peak_value = value
        if self.peak_value > 0:
            dd = (1.0 - value / self.peak_value) * 100.0
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

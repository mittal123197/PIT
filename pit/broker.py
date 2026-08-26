"""Broker interface + paper implementation.

The `Broker` boundary is the single seam between "the arena" and "where orders
go". Phase 1 uses `PaperBroker`, which fills instantly at the feed's current
price against an in-memory Portfolio. Phase 4 swaps in a `DhanBroker` behind the
same interface — the engine above never learns which one it's talking to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .portfolio import Portfolio


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    cash_after: float


class InsufficientFunds(Exception):
    pass


class InvalidOrder(Exception):
    pass


class Broker(ABC):
    """Executes orders against a portfolio. Stateless w.r.t. the arena."""

    @abstractmethod
    def execute(
        self, portfolio: Portfolio, symbol: str, side: str, qty: float, price: float
    ) -> Fill:
        ...

    def liquidate(self, portfolio: Portfolio, prices: dict[str, float]) -> list[Fill]:
        """Sell every open position at current prices (used for stop-loss)."""
        fills: list[Fill] = []
        for sym, qty in list(portfolio.positions.items()):
            if qty > 0 and sym in prices:
                fills.append(self.execute(portfolio, sym, "sell", qty, prices[sym]))
        return fills


class PaperBroker(Broker):
    """Instant fills at the quoted price. Long-only unless configured otherwise."""

    def __init__(self, per_order_fee: float = 0.0, allow_short: bool = False) -> None:
        self.per_order_fee = per_order_fee
        self.allow_short = allow_short

    def execute(
        self, portfolio: Portfolio, symbol: str, side: str, qty: float, price: float
    ) -> Fill:
        if qty <= 0:
            raise InvalidOrder(f"qty must be positive, got {qty}")
        if price <= 0:
            raise InvalidOrder(f"price must be positive, got {price}")
        side = side.lower()

        if side == "buy":
            cost = qty * price + self.per_order_fee
            if cost > portfolio.cash + 1e-9:
                raise InsufficientFunds(
                    f"buy {qty}x{symbol}@{price:.2f} costs {cost:.2f}, "
                    f"cash is {portfolio.cash:.2f}"
                )
            portfolio.cash -= cost
            portfolio.positions[symbol] = portfolio.position(symbol) + qty

        elif side == "sell":
            held = portfolio.position(symbol)
            if not self.allow_short and qty > held + 1e-9:
                raise InvalidOrder(
                    f"sell {qty}x{symbol} but only {held} held (short disabled)"
                )
            proceeds = qty * price - self.per_order_fee
            portfolio.cash += proceeds
            remaining = held - qty
            if abs(remaining) < 1e-9:
                portfolio.positions.pop(symbol, None)
            else:
                portfolio.positions[symbol] = remaining
        else:
            raise InvalidOrder(f"side must be buy/sell, got {side!r}")

        portfolio.trade_count += 1
        return Fill(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            fee=self.per_order_fee,
            cash_after=portfolio.cash,
        )

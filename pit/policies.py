"""Agent policies — the "brain" that decides actions on each wake.

An agent's context is scoped to only its own state (never the opponent's),
which is what enforces "blind during the round". A policy returns a list of
Actions; the engine executes them.

Phase 1 ships `SimplePolicy`: deterministic momentum / mean-reversion rules so
the whole arena runs and is testable with no LLM key. The `strategy_config`
dict it reads is exactly the artefact the mutation step rewrites between rounds,
and the LLM policy in Phase 2 slots in behind the same `decide()` signature.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from .actions import Action, Hold, PlaceOrder, SetHeartbeat, SetWatch


@dataclass
class AgentContext:
    """What an agent may see.

    Its own state in full, PLUS the opponent's live return % (the scoreboard) —
    but NOT the opponent's trades. Seeing the score creates competitive pressure
    and a survival instinct; hiding the trade log preserves the loser-copies-
    winner mechanic (you only get the winner's log *after* you lose).
    """
    now: datetime
    prices: dict[str, float]
    history: dict[str, list[float]]     # recent close prices per symbol
    cash: float
    positions: dict[str, float]
    return_pct: float
    universe: list[str]
    strategy_config: dict
    guidelines: list[str] = field(default_factory=list)
    goal_pct: float = 0.0
    bars_remaining: int = 0
    opponent_return_pct: float = 0.0    # the visible scoreboard
    opponent_liquidated: bool = False


class AgentPolicy(ABC):
    @abstractmethod
    def decide(self, ctx: AgentContext) -> list[Action]: ...


class SimplePolicy(AgentPolicy):
    """Deterministic rules driven entirely by `strategy_config`.

    Supported `type`s: "momentum", "mean_reversion". Every numeric field is a
    mutation target, so two lineages with different configs — or a mutated
    generation — behave visibly differently.
    """

    def decide(self, ctx: AgentContext) -> list[Action]:
        cfg = ctx.strategy_config
        kind = cfg.get("type", "momentum")
        actions: list[Action] = []

        # First wake of a round: declare a cadence, so the engine can honour a
        # per-agent heartbeat rather than a global tick.
        if cfg.get("_announced_heartbeat") is not True:
            actions.append(SetHeartbeat(minutes=int(cfg.get("heartbeat_minutes", 60))))
            cfg["_announced_heartbeat"] = True  # engine persists config; harmless if not

        # Survival instinct: if the visible scoreboard says we're being beaten,
        # get more aggressive — bigger size, easier entries — to claw back. Not
        # a copy of the opponent (we can't see their trades), just a response to
        # falling behind.
        work = dict(cfg)
        deficit = ctx.opponent_return_pct - ctx.return_pct
        survival = deficit > float(cfg.get("survival_margin_pct", 2.0))
        if survival and not ctx.opponent_liquidated:
            work["position_frac"] = min(0.9, float(cfg.get("position_frac", 0.25)) * 1.6)
            work["buy_threshold_pct"] = float(cfg.get("buy_threshold_pct", 1.0)) * 0.5
            work["band_pct"] = float(cfg.get("band_pct", 2.0)) * 0.6

        if kind == "mean_reversion":
            actions.extend(self._mean_reversion(ctx, work, survival))
        else:
            actions.extend(self._momentum(ctx, work, survival))

        return actions or [Hold(reason="no signal")]

    # -- strategies -------------------------------------------------------

    def _momentum(self, ctx: AgentContext, cfg: dict, survival: bool = False) -> list[Action]:
        lookback = int(cfg.get("lookback", 5))
        buy_thr = float(cfg.get("buy_threshold_pct", 1.0))
        sell_thr = float(cfg.get("sell_threshold_pct", -1.0))
        frac = float(cfg.get("position_frac", 0.25))
        tag = "survival " if survival else ""
        out: list[Action] = []

        for sym in ctx.universe:
            hist = ctx.history.get(sym, [])
            if len(hist) <= lookback:
                continue
            mom = (hist[-1] / hist[-1 - lookback] - 1.0) * 100.0
            price = ctx.prices.get(sym, hist[-1])
            held = ctx.positions.get(sym, 0.0)

            if mom >= buy_thr and held == 0.0:
                qty = self._qty_for(ctx.cash, price, frac)
                if qty > 0:
                    out.append(PlaceOrder(sym, "buy", qty,
                                          reason=f"{tag}momentum {mom:.2f}%>= {buy_thr:.2f}%"))
            elif mom <= sell_thr and held > 0.0:
                out.append(PlaceOrder(sym, "sell", held,
                                      reason=f"momentum {mom:.2f}%<= {sell_thr}%"))
        return out

    def _mean_reversion(self, ctx: AgentContext, cfg: dict, survival: bool = False) -> list[Action]:
        lookback = int(cfg.get("lookback", 10))
        band = float(cfg.get("band_pct", 2.0))
        frac = float(cfg.get("position_frac", 0.25))
        tag = "survival " if survival else ""
        out: list[Action] = []

        for sym in ctx.universe:
            hist = ctx.history.get(sym, [])
            if len(hist) < lookback:
                continue
            window = hist[-lookback:]
            mean = sum(window) / len(window)
            price = ctx.prices.get(sym, hist[-1])
            held = ctx.positions.get(sym, 0.0)
            dev = (price / mean - 1.0) * 100.0

            if dev <= -band and held == 0.0:
                qty = self._qty_for(ctx.cash, price, frac)
                if qty > 0:
                    out.append(PlaceOrder(sym, "buy", qty,
                                          reason=f"{tag}{dev:.2f}% below mean"))
                    # a mean-reversion agent naturally wants an alert if it dips
                    # further — exercises the watch path
                    out.append(SetWatch(sym, "price_below", price * (1 - band / 100),
                                        reason="add-on level"))
            elif dev >= band and held > 0.0:
                out.append(PlaceOrder(sym, "sell", held,
                                      reason=f"{dev:.2f}% above mean"))
        return out

    @staticmethod
    def _qty_for(cash: float, price: float, frac: float) -> float:
        if price <= 0:
            return 0.0
        budget = cash * frac
        return float(int(budget // price))  # whole shares only

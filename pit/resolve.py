"""The tie-break cascade — the rule that makes a draw mechanically impossible.

Pure functions over two agent summaries so they're trivially unit-testable
(the plan's verification section calls for exactly this). The engine handles
the sudden-death mini-round; here, an all-tie simply reports `sudden_death` so
the caller knows to run one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentSummary:
    agent_id: int
    return_pct: float
    trade_count: int
    max_drawdown_pct: float
    liquidated: bool = False


@dataclass
class Resolution:
    winner_id: int | None      # None only when reason == "sudden_death"
    loser_id: int | None
    reason: str                # return | trade_count | drawdown | sudden_death


def _tie(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps


def resolve(a: AgentSummary, b: AgentSummary, tie_epsilon_pct: float = 0.01) -> Resolution:
    """Return the winner via the cascade. Never returns a draw.

    Cascade, first rule that separates them wins:
      1. higher absolute return
      2. fewer trades (less noise, more signal)
      3. lower max drawdown (better risk control)
      4. still tied -> sudden death (caller runs a 1-day mini-round)
    """
    # 1. absolute return
    if not _tie(a.return_pct, b.return_pct, tie_epsilon_pct):
        winner, loser = (a, b) if a.return_pct > b.return_pct else (b, a)
        return Resolution(winner.agent_id, loser.agent_id, "return")

    # 2. fewer trades
    if a.trade_count != b.trade_count:
        winner, loser = (a, b) if a.trade_count < b.trade_count else (b, a)
        return Resolution(winner.agent_id, loser.agent_id, "trade_count")

    # 3. lower drawdown
    if not _tie(a.max_drawdown_pct, b.max_drawdown_pct, tie_epsilon_pct):
        winner, loser = (
            (a, b) if a.max_drawdown_pct < b.max_drawdown_pct else (b, a)
        )
        return Resolution(winner.agent_id, loser.agent_id, "drawdown")

    # 4. genuine dead heat -> sudden death
    return Resolution(None, None, "sudden_death")

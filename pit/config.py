"""Tunable knobs for the arena. Every value can be overridden from the
environment (a local `.env` is loaded by `pit.db`), so a run never needs a
code edit.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


# Which market the arena trades: "us" (US top-100 stocks) or "crypto".
MARKET: str = os.getenv("PIT_MARKET", "us").lower()
CURRENCY: str = "$"


@dataclass(frozen=True)
class ArenaConfig:
    # --- capital: every agent starts every round with the same paper stake ---
    base_capital: float = _env_float("PIT_BASE_CAPITAL", 100_000.0)

    # --- fallback risk bands (used only when no volatility data is available;
    # live rounds size their bands from volatility, see pit/risk.py) ---
    # Portfolio stop for a round of N days = daily_stop_loss_pct * sqrt(N)
    # (3.78 * sqrt(7) ≈ 10% for a week); take-profit = stop * risk_reward_ratio.
    daily_stop_loss_pct: float = _env_float("PIT_DAILY_STOP_LOSS_PCT", 3.78)
    risk_reward_ratio: float = _env_float("PIT_RISK_REWARD_RATIO", 1.5)
    # Default per-position stop (% below average entry) when no volatility-
    # based auto stop exists; an agent may set its own per buy.
    position_stop_loss_pct: float = _env_float("PIT_POSITION_STOP_LOSS_PCT", 8.0)
    # Day count used for the fallback bands when a round has no fixed length.
    start_round_days: int = _env_int("PIT_START_ROUND_DAYS", 7)

    # --- execution ---
    # Half the bid/ask spread, in basis points, paid on every fill (buys fill
    # above the quote, sells below). Marks stay at the quote. ~3 bps is
    # realistic for the largest US stocks; mid-cap crypto pairs are wider.
    spread_bps: float = _env_float("PIT_SPREAD_BPS", 8.0 if MARKET == "crypto" else 3.0)

    # --- ranking ---
    # Returns within this many percentage points tie, and the tie-break moves
    # on (fewer trades, then lower drawdown) — never a draw.
    tie_epsilon_pct: float = _env_float("PIT_TIE_EPSILON_PCT", 0.01)
    elo_k: float = _env_float("PIT_ELO_K", 32.0)
    elo_base: float = _env_float("PIT_ELO_BASE", 1000.0)

    # --- the rulebook (see pit/guidelines.py) ---
    reflection_enabled: bool = os.getenv("PIT_REFLECTION", "1") not in ("0", "false", "False")
    reflection_interval: int = _env_int("PIT_REFLECTION_INTERVAL", 1)   # every N rounds
    max_proposals_per_round: int = _env_int("PIT_MAX_PROPOSALS_PER_ROUND", 3)
    max_active_guidelines: int = _env_int("PIT_MAX_ACTIVE_GUIDELINES", 12)
    max_adoptions_per_round: int = _env_int("PIT_MAX_ADOPTIONS_PER_ROUND", 1)

    # --- the tradeable universe for live sessions ---
    # 'top100' (curated large US names) or 'crypto'. See pit/full_market.py.
    trade_universe_mode: str = os.getenv("PIT_TRADE_UNIVERSE", "top100").lower()

    def stop_loss_pct_for(self, round_days: int) -> float:
        """Fallback portfolio stop for a round of this length (sqrt-time);
        0 days (time-based rounds) uses start_round_days."""
        days = round_days if round_days and round_days > 0 else self.start_round_days
        return round(self.daily_stop_loss_pct * math.sqrt(days), 3)

    def goal_pct_for(self, round_days: int) -> float:
        """Fallback take-profit: the stop scaled by risk_reward_ratio."""
        return round(self.stop_loss_pct_for(round_days) * self.risk_reward_ratio, 3)


DEFAULT = ArenaConfig()

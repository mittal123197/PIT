"""Central, tunable knobs for the arena.

Nothing here is load-bearing on day one — every number is a starting point
chosen in the build plan, meant to be adjusted once the mechanic is running.
Values can be overridden via environment variables (loaded from a local
`.env` by `pit.db`/CLI) so deploys don't need code edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


# The watchlist the agents may trade in Phase 1. NSE tickers (yfinance uses the
# `.NS` suffix); the synthetic feed just treats them as opaque symbol names.
# The tradeable universe for the PAPER agents — liquid NSE large-caps across
# sectors, chosen for tight spreads / realistic fills, NOT as investment picks.
# (yfinance uses the `.NS` suffix for NSE.)
DEFAULT_UNIVERSE: list[str] = [
    "RELIANCE.NS",    # energy/conglomerate
    "TCS.NS",         # IT
    "HDFCBANK.NS",    # bank
    "INFY.NS",        # IT
    "ICICIBANK.NS",   # bank
    "SBIN.NS",        # bank (PSU)
    "BHARTIARTL.NS",  # telecom
    "ITC.NS",         # FMCG
    "LT.NS",          # infra/capital goods
    "HINDUNILVR.NS",  # FMCG
]


@dataclass(frozen=True)
class ArenaConfig:
    # --- capital & risk (set once at round start, never touched by the agent) ---
    base_capital: float = _env_float("PIT_BASE_CAPITAL", 100_000.0)  # ₹ paper
    stop_loss_pct: float = _env_float("PIT_STOP_LOSS_PCT", 10.0)      # hard constraint
    stake_delta_pct: float = _env_float("PIT_STAKE_DELTA_PCT", 20.0)  # winner +, loser -

    # --- round length (see build plan: 7-day rounds, shrink to a 4-day floor) ---
    start_round_days: int = _env_int("PIT_START_ROUND_DAYS", 7)
    round_shrink_days: int = _env_int("PIT_ROUND_SHRINK_DAYS", 1)
    min_round_days: int = _env_int("PIT_MIN_ROUND_DAYS", 4)

    # goal_pct is display-only narrative; winners are resolved by the return
    # cascade, not by the goal. Scales with round length: 0.714/day => ~5% over
    # a 7-day round (a stretch target to show, not a realistic expectation).
    goal_pct_per_day: float = _env_float("PIT_GOAL_PCT_PER_DAY", 0.714)

    # --- how a "day" maps to price bars (so round length is real) ---
    # NSE trades ~375 minutes/day; at 15-min bars that's 25 bars per day. A
    # round of `length_days` therefore replays length_days * bars_per_day bars,
    # so a 4-day round genuinely simulates less market than a 7-day one.
    bar_minutes: int = _env_int("PIT_BAR_MINUTES", 15)
    market_minutes_per_day: int = _env_int("PIT_MARKET_MINUTES", 375)

    def bars_per_day(self) -> int:
        return max(1, self.market_minutes_per_day // self.bar_minutes)

    def bars_for_days(self, length_days: int) -> int:
        return max(2, length_days * self.bars_per_day())

    # --- rating ---
    elo_k: float = _env_float("PIT_ELO_K", 32.0)
    elo_base: float = _env_float("PIT_ELO_BASE", 1000.0)

    # --- shared guidelines ("constitution") ---
    # A reflection pass runs every N rounds, drafting at most one proposal only
    # if a pattern recurs — "worked once" doesn't qualify. Every lineage then
    # votes; a strict majority is required (ties keep the status quo).
    reflection_enabled: bool = os.getenv("PIT_REFLECTION", "1") not in ("0", "false", "False")
    reflection_interval: int = _env_int("PIT_REFLECTION_INTERVAL", 5)
    reflection_min_sample: int = _env_int("PIT_REFLECTION_MIN_SAMPLE", 3)
    reflection_pattern_frac: float = _env_float("PIT_REFLECTION_PATTERN_FRAC", 0.6)

    # --- execution ---
    per_order_fee: float = _env_float("PIT_PER_ORDER_FEE", 0.0)  # Dhan: ₹0 delivery
    allow_short: bool = False  # Phase 1: long-only paper trading

    # Two returns within this many percentage points count as a tie, sending
    # resolution to the next cascade rule instead of splitting hairs on noise.
    tie_epsilon_pct: float = _env_float("PIT_TIE_EPSILON_PCT", 0.01)

    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    def goal_pct_for(self, round_days: int) -> float:
        return round(self.goal_pct_per_day * round_days, 3)

    def next_round_days(self, current_days: int) -> int:
        return max(self.min_round_days, current_days - self.round_shrink_days)


DEFAULT = ArenaConfig()

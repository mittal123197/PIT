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
# The candidate MARKET the agents research and pick from — deliberately spanning
# large, mid and small caps so there's NO cap bias baked in by us. The agents
# choose which of these to trade; nothing here is a recommendation.
#
# This built-in list is just a starting pool. To hand the agents the entire
# market instead, set PIT_UNIVERSE="TICK1.NS,TICK2.NS,..." or point
# PIT_UNIVERSE_FILE at a newline-separated list (e.g. the full NSE equity list).
# The historical feed drops any ticker that doesn't return data, so an
# over-broad list is safe.
_BUILTIN_UNIVERSE: list[str] = [
    # large cap
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "BAJFINANCE.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "ONGC.NS",
    # mid cap
    "DIXON.NS", "PERSISTENT.NS", "COFORGE.NS", "POLYCAB.NS", "ASTRAL.NS",
    "PAGEIND.NS", "MPHASIS.NS", "AUBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "INDHOTEL.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "CUMMINSIND.NS",
    "HAVELLS.NS", "GODREJCP.NS", "MARICO.NS", "PIDILITIND.NS", "TATAPOWER.NS",
    "GAIL.NS", "VEDL.NS", "SAIL.NS", "PFC.NS", "RECLTD.NS",
    "IRCTC.NS", "DMART.NS", "TRENT.NS", "LTIM.NS", "APOLLOHOSP.NS",
    # small / newer listings
    "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "IEX.NS", "CDSL.NS",
    "BSE.NS", "ANGELONE.NS", "KPITTECH.NS", "TATAELXSI.NS", "SUZLON.NS",
    "IRFC.NS", "YESBANK.NS", "IDEA.NS", "ZEEL.NS", "RVNL.NS",
]


def resolve_universe() -> list[str]:
    """The tradeable pool. Overridable so the list comes from the market/user,
    not from us: PIT_UNIVERSE (comma list) or PIT_UNIVERSE_FILE (a file path)."""
    raw = os.getenv("PIT_UNIVERSE")
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    path = os.getenv("PIT_UNIVERSE_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            return [ln.strip() for ln in f
                    if ln.strip() and not ln.startswith("#")]
    return list(_BUILTIN_UNIVERSE)


DEFAULT_UNIVERSE: list[str] = resolve_universe()


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

    universe: list[str] = field(default_factory=resolve_universe)

    def goal_pct_for(self, round_days: int) -> float:
        return round(self.goal_pct_per_day * round_days, 3)

    def next_round_days(self, current_days: int) -> int:
        return max(self.min_round_days, current_days - self.round_shrink_days)


DEFAULT = ArenaConfig()

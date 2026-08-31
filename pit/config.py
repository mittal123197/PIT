"""Central, tunable knobs for the arena.

Nothing here is load-bearing on day one — every number is a starting point
chosen in the build plan, meant to be adjusted once the mechanic is running.
Values can be overridden via environment variables (loaded from a local
`.env` by `pit.db`/CLI) so deploys don't need code edits.
"""
from __future__ import annotations

import math
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
# Which market the arena trades. The autonomous agents pick any ticker they
# want; the lists below are ONLY a "what's moving" discovery reference (spanning
# caps so no bias is baked in) and a default pool for the offline replay engine.
MARKET: str = os.getenv("PIT_MARKET", "us").lower()
CURRENCY: str = "$" if MARKET == "us" else "₹"

_US_UNIVERSE: list[str] = [
    # mega / large cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "ADBE", "CRM", "ORCL", "INTC", "QCOM", "CSCO", "MU", "TXN",
    "JPM", "BAC", "WMT", "DIS", "BA", "XOM", "PYPL", "COST", "PEP", "KO",
    # higher-volatility growth / momentum / meme
    "COIN", "PLTR", "MSTR", "SMCI", "ARM", "SOFI", "HOOD", "RIVN", "LCID",
    "MARA", "RIOT", "AFRM", "SNAP", "UBER", "ABNB", "SHOP", "ROKU", "DKNG",
    "GME", "AMC", "NIO", "F", "CCL", "CVNA", "DELL", "MRVL", "PANW", "NOW",
]

_NSE_UNIVERSE: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "BAJFINANCE.NS", "AXISBANK.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "COALINDIA.NS", "ONGC.NS", "DIXON.NS",
    "PERSISTENT.NS", "COFORGE.NS", "TATAPOWER.NS", "VEDL.NS", "SAIL.NS",
    "IRFC.NS", "SUZLON.NS", "RVNL.NS", "ETERNAL.NS", "TRENT.NS",
]


def market_reference() -> list[str]:
    return _US_UNIVERSE if MARKET == "us" else _NSE_UNIVERSE


# kept for imports; the movers feed and offline engine use this.
_BUILTIN_UNIVERSE: list[str] = market_reference()


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
    return list(market_reference())


DEFAULT_UNIVERSE: list[str] = resolve_universe()


@dataclass(frozen=True)
class ArenaConfig:
    # --- capital & risk (set once at round start, never touched by the agent) ---
    base_capital: float = _env_float("PIT_BASE_CAPITAL", 100_000.0)  # ₹ paper
    # Stop-loss and take-profit are both HARD constraints, enforced by the
    # arena, not the agent — when either fires the position is force-closed
    # and locked for the rest of the round. Stop-loss scales with round
    # length by sqrt(days) (standard volatility-over-time scaling: risk
    # grows with the square root of the holding period, not linearly) so a
    # 5-day round isn't punished by the same 10% band a 7-day round gets.
    # `daily_stop_loss_pct` is calibrated so a 7-day round still stops out
    # at 10% (3.78 * sqrt(7) ≈ 10.0) — same risk band as before, now tunable
    # by round length instead of flat.
    daily_stop_loss_pct: float = _env_float("PIT_DAILY_STOP_LOSS_PCT", 3.78)
    # Take-profit is the stop-loss scaled UP by the reward:risk ratio below —
    # not a separate flat number — so the two always move together when you
    # tune either the stop-loss or the ratio. See risk_reward_ratio.
    risk_reward_ratio: float = _env_float("PIT_RISK_REWARD_RATIO", 1.5)
    # Asymmetric on purpose: winning is rewarded slightly more than losing is
    # punished (25% up vs 20% down), so a lineage can claw back from one loss.
    win_stake_bonus_pct: float = _env_float("PIT_WIN_STAKE_BONUS_PCT", 25.0)
    loss_stake_penalty_pct: float = _env_float("PIT_LOSS_STAKE_PENALTY_PCT", 20.0)
    # A "win" earned by making ZERO trades (never entered the market at all —
    # whether a deliberate all-cash hold or a decision loop that errored out
    # and silently did nothing) didn't actually beat the rival's trading, it
    # just avoided the rival's own loss. Capped well below the normal bonus
    # so an inactive/failed round can't earn full credit for someone else's
    # stop-out.
    passive_win_stake_bonus_pct: float = _env_float("PIT_PASSIVE_WIN_STAKE_BONUS_PCT", 8.0)
    # With 3+ agents battling simultaneously, only 1st place truly "wins."
    # Last place gets the full loss_stake_penalty_pct above; anyone strictly
    # in between (2nd of 3, 2nd/3rd of 4, ...) is still a loser — it didn't
    # win — but punished less harshly than dead last.
    middle_place_penalty_pct: float = _env_float("PIT_MIDDLE_PLACE_PENALTY_PCT", 8.0)

    # Per-POSITION hard stop — separate from the portfolio-level one above.
    # Without this, one stock collapsing inside an otherwise-fine book just
    # sits there until the AGGREGATE return crosses the portfolio stop; this
    # force-sells that one position on its own, at its own entry price, the
    # moment it alone falls this far. Deliberately much wider than the
    # portfolio band (single-stock moves are noisier than a blended book).
    position_stop_loss_pct: float = _env_float("PIT_POSITION_STOP_LOSS_PCT", 8.0)

    # --- round length: fixed at 7 days, shrink mechanic OFF by default ---
    # (set PIT_ROUND_SHRINK_DAYS > 0 to bring back the shrinking-rounds idea)
    start_round_days: int = _env_int("PIT_START_ROUND_DAYS", 7)
    round_shrink_days: int = _env_int("PIT_ROUND_SHRINK_DAYS", 0)
    min_round_days: int = _env_int("PIT_MIN_ROUND_DAYS", 5)

    # goal_pct is now a REAL hard take-profit trigger (not display-only): once
    # an agent's return reaches it, the arena force-books the profit and
    # freezes the position for the rest of the round, same as a stop-loss.
    # Winners are still resolved by the return cascade at round end, not by
    # who hits goal first — this only stops a winner from giving a locked-in
    # gain back to the market on the way to the deadline.

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

    # --- discovery pool for scan_full_market() ---
    # 'top500' (default): S&P 500 constituents only — a real, published index,
    # not a hand-picked shortlist, but still ~500 liquid/listed companies, so
    # discovery stays genuinely diverse without the delisted/illiquid penny
    # names a fully random sample of all ~5,400 NASDAQ/NYSE/AMEX tickers pulls
    # in. 'full' restores the unrestricted universe. See full_market.py.
    trade_universe_mode: str = os.getenv("PIT_TRADE_UNIVERSE", "top500").lower()

    def stop_loss_pct_for(self, round_days: int) -> float:
        """Hard stop-loss for a round of this length, sqrt(days)-scaled.
        `round_days` may be 0 (time-based/live rounds carry no fixed day
        count) — fall back to the standard round length for scaling."""
        days = round_days if round_days and round_days > 0 else self.start_round_days
        return round(self.daily_stop_loss_pct * math.sqrt(days), 3)

    def goal_pct_for(self, round_days: int) -> float:
        """Hard take-profit for a round of this length: the stop-loss scaled
        up by risk_reward_ratio, so raising the ratio (or the daily stop-loss)
        moves both bands together instead of drifting out of sync."""
        return round(self.stop_loss_pct_for(round_days) * self.risk_reward_ratio, 3)

    def next_round_days(self, current_days: int) -> int:
        return max(self.min_round_days, current_days - self.round_shrink_days)


DEFAULT = ArenaConfig()

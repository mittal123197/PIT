"""Stop-loss / take-profit scaling: sqrt(days)-scaled risk band, and the
take-profit derived from it via a tunable reward:risk ratio.

Covers the two hard constraints introduced alongside the audit-log work:
  - stop_loss_pct_for(days) scales with sqrt(days), not flat or linear.
  - goal_pct_for(days) is always risk_reward_ratio * stop_loss_pct_for(days),
    so the two never drift out of sync when either knob is tuned.
  - length_days == 0 (time-based/live rounds carry no fixed day count) falls
    back to start_round_days for the scaling, instead of dividing by zero.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit.config import ArenaConfig  # noqa: E402


def test_stop_loss_scales_with_sqrt_days():
    c = ArenaConfig()
    s5 = c.stop_loss_pct_for(5)
    s7 = c.stop_loss_pct_for(7)
    s14 = c.stop_loss_pct_for(14)
    # sqrt-scaling: doubling days does NOT double the stop-loss band
    assert s5 < s7 < s14
    assert s14 < s7 * 2


def test_seven_day_default_matches_legacy_ten_percent():
    # daily_stop_loss_pct is calibrated so a 7-day round keeps the old flat
    # 10% stop-loss band — a deliberate continuity choice, not a coincidence.
    c = ArenaConfig()
    assert abs(c.stop_loss_pct_for(7) - 10.0) < 0.05


def test_goal_is_ratio_times_stop_loss_at_every_length():
    c = ArenaConfig()
    for days in (3, 5, 7, 10, 14, 21):
        stop = c.stop_loss_pct_for(days)
        goal = c.goal_pct_for(days)
        # both values are independently rounded to 3dp, so compare loosely
        assert math.isclose(goal, stop * c.risk_reward_ratio, abs_tol=0.01)


def test_zero_days_falls_back_to_start_round_days():
    c = ArenaConfig()
    assert c.stop_loss_pct_for(0) == c.stop_loss_pct_for(c.start_round_days)
    assert c.goal_pct_for(0) == c.goal_pct_for(c.start_round_days)


def test_raising_ratio_only_moves_goal_not_stop_loss():
    base = ArenaConfig()
    wider = ArenaConfig(risk_reward_ratio=base.risk_reward_ratio * 2)
    assert wider.stop_loss_pct_for(7) == base.stop_loss_pct_for(7)
    assert wider.goal_pct_for(7) > base.goal_pct_for(7)


def test_constructor_overrides_move_both_bands_together():
    # (dataclass field defaults read PIT_* env vars once at import time, so
    # this exercises the override path the way `.env` actually does — via
    # the constructor — rather than monkeypatching env vars post-import.)
    c = ArenaConfig(daily_stop_loss_pct=1.0, risk_reward_ratio=2.0)
    assert math.isclose(c.stop_loss_pct_for(9), 1.0 * 3.0)  # sqrt(9) == 3
    assert math.isclose(c.goal_pct_for(9), 6.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

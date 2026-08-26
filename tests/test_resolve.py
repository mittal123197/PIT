"""The resolution cascade must ALWAYS produce a winner — never a draw.

This is the plan's key correctness guarantee, so it gets the most direct test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit.resolve import AgentSummary, resolve  # noqa: E402


def s(agent_id, ret, trades, dd, liq=False):
    return AgentSummary(agent_id, ret, trades, dd, liq)


def test_higher_return_wins():
    r = resolve(s(1, 5.0, 10, 3.0), s(2, 2.0, 4, 1.0))
    assert r.winner_id == 1 and r.reason == "return"


def test_negative_less_bad_wins():
    r = resolve(s(1, -8.0, 5, 9.0), s(2, -2.0, 5, 4.0))
    assert r.winner_id == 2 and r.reason == "return"


def test_tie_return_breaks_on_fewer_trades():
    r = resolve(s(1, 3.0, 12, 5.0), s(2, 3.0, 3, 5.0))
    assert r.winner_id == 2 and r.reason == "trade_count"


def test_tie_return_and_trades_breaks_on_drawdown():
    r = resolve(s(1, 3.0, 7, 9.0), s(2, 3.0, 7, 2.0))
    assert r.winner_id == 2 and r.reason == "drawdown"


def test_total_tie_goes_to_sudden_death():
    r = resolve(s(1, 3.0, 7, 5.0), s(2, 3.0, 7, 5.0))
    assert r.winner_id is None and r.reason == "sudden_death"


def test_epsilon_treats_tiny_diff_as_tie():
    # returns differ by less than epsilon -> should fall through to trade count
    r = resolve(s(1, 3.000, 9, 5.0), s(2, 3.005, 2, 5.0), tie_epsilon_pct=0.01)
    assert r.reason == "trade_count" and r.winner_id == 2


def test_never_a_draw_across_random_cases():
    import random
    rng = random.Random(0)
    for _ in range(2000):
        a = s(1, round(rng.uniform(-20, 20), 3), rng.randint(0, 20),
              round(rng.uniform(0, 20), 3))
        b = s(2, round(rng.uniform(-20, 20), 3), rng.randint(0, 20),
              round(rng.uniform(0, 20), 3))
        r = resolve(a, b)
        # either a clear winner, or an explicit sudden-death signal — never a
        # silent draw with both ids populated equally / both None on a non-SD reason
        if r.reason == "sudden_death":
            assert r.winner_id is None
        else:
            assert r.winner_id in (1, 2) and r.loser_id in (1, 2)
            assert r.winner_id != r.loser_id


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

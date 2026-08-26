"""ELO rating updates.

Standard ELO: beating a strong opponent moves you more than beating a weak one,
so the rating reflects opponent strength, not just a win count.
"""
from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update(
    winner_rating: float, loser_rating: float, k: float = 32.0
) -> tuple[float, float, float]:
    """Return (new_winner, new_loser, delta_applied_to_winner)."""
    exp_w = expected_score(winner_rating, loser_rating)
    delta = k * (1.0 - exp_w)          # winner scored 1
    new_winner = winner_rating + delta
    new_loser = loser_rating - delta   # zero-sum
    return round(new_winner, 2), round(new_loser, 2), round(delta, 2)

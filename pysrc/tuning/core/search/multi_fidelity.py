"""Multi-fidelity search: successive halving and Hyperband bracket logic."""

from __future__ import annotations


def successive_halving_brackets(
    max_trials: int,
    min_budget: int,
    max_budget: int,
    eta: int = 3,
) -> list[tuple[int, int]]:
    """Return (n_trials, budget) brackets for successive halving.

    Each bracket halves the number of trials while multiplying budget by eta.
    """
    brackets: list[tuple[int, int]] = []
    n = max_trials
    b = min_budget
    while b <= max_budget and n >= 1:
        brackets.append((n, b))
        n = max(1, n // eta)
        b = min(b * eta, max_budget)
    return brackets


__all__ = ["successive_halving_brackets"]

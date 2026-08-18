"""Capacity constraints: scale scores by AUM capacity decay factor."""

from __future__ import annotations


def capacity_decay_factor(aum: float, half_capacity: float) -> float:
    """Return a [0, 1] decay factor: 1.0 at aum=0, 0.5 at aum=half_capacity."""
    if half_capacity <= 0:
        raise ValueError("half_capacity must be positive")
    return half_capacity / (half_capacity + aum)


def apply_capacity_penalty(score: float, aum: float, half_capacity: float) -> float:
    """Scale a score by the capacity decay factor."""
    return score * capacity_decay_factor(aum, half_capacity)


__all__ = ["capacity_decay_factor", "apply_capacity_penalty"]

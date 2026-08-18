"""Bayesian optimisation primitives: acquisition function utilities."""

from __future__ import annotations

import math


class BayesOptError(RuntimeError):
    """Raised when Bayesian optimisation encounters an irrecoverable state."""


def expected_improvement(
    mu: float,
    sigma: float,
    best_so_far: float,
    xi: float = 0.01,
) -> float:
    """Compute Expected Improvement acquisition value (Gaussian surrogate)."""
    z = (mu - best_so_far - xi) / (sigma + 1e-9)
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mu - best_so_far - xi) * phi + sigma * pdf


__all__ = ["BayesOptError", "expected_improvement"]

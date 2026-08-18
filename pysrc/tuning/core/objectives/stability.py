"""Stability metrics: variance of scores across folds to detect overfit candidates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fold_score_variance(fold_scores: NDArray[np.float64]) -> float:
    """Return variance of per-fold scores (ddof=1); 0 if fewer than 2 observations."""
    return float(np.var(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0


def is_stable(fold_scores: NDArray[np.float64], max_var: float) -> bool:
    """Return True if fold score variance is at or below max_var."""
    return fold_score_variance(fold_scores) <= max_var


def stability_score(fold_scores: NDArray[np.float64]) -> float:
    """Return 1 / (1 + variance); higher is more stable."""
    return 1.0 / (1.0 + fold_score_variance(fold_scores))


__all__ = ["fold_score_variance", "is_stable", "stability_score"]

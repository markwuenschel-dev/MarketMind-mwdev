"""Pareto front computation for multi-objective tuning."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def is_pareto_efficient(scores: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return boolean mask of Pareto-efficient rows (all objectives: maximise)."""
    n = scores.shape[0]
    is_eff = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if is_eff[i]:
            others = np.where(is_eff)[0]
            dominated = np.all(scores[others] >= scores[i], axis=1) & np.any(
                scores[others] > scores[i], axis=1
            )
            is_eff[others[dominated]] = False
            is_eff[i] = True
    return is_eff


def pareto_front(
    candidates: list[dict[str, float]],
    objectives: list[str],
) -> list[dict[str, float]]:
    """Return the subset of candidates on the Pareto front for the given objectives."""
    if not candidates:
        return []
    mat = np.array([[c[o] for o in objectives] for c in candidates], dtype=np.float64)
    mask = is_pareto_efficient(mat)
    return [c for c, m in zip(candidates, mask, strict=False) if m]


__all__ = ["is_pareto_efficient", "pareto_front"]

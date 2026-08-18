"""ContextFeatures: typed feature vector describing a tuning job's search context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextFeatures:
    """Features used by the meta-learning algorithm recommender."""

    n_dimensions: int
    n_continuous: int
    n_discrete: int
    n_categorical: int
    budget_trials: int
    n_symbols: int
    n_folds: int
    has_prior_experiments: bool


__all__ = ["ContextFeatures"]

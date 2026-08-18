"""Algorithm recommender: suggest a search algorithm given context features."""

from __future__ import annotations

from pysrc.tuning.core.meta_learning.context_features import ContextFeatures


def recommend_algorithm(ctx: ContextFeatures) -> str:
    """Return the recommended search algorithm name for the given context."""
    if ctx.n_dimensions <= 5 and ctx.budget_trials >= 100:
        return "bayes_opt"
    if ctx.n_dimensions > 20:
        return "evolutionary"
    if ctx.budget_trials < 30:
        return "random"
    return "bayes_opt"


__all__ = ["recommend_algorithm"]

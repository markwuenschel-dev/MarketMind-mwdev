"""CompositeScore: weighted combination of metrics plus penalty terms."""

from __future__ import annotations


def composite_score(
    metrics: dict[str, float],
    weights: dict[str, float],
    penalty_terms: list[float] | None = None,
) -> float:
    """Compute a weighted sum of named metrics plus optional penalty terms."""
    base = sum(weights.get(k, 0.0) * v for k, v in metrics.items())
    penalties = sum(penalty_terms) if penalty_terms else 0.0
    return base + penalties


__all__ = ["composite_score"]

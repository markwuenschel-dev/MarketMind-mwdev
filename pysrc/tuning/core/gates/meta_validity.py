"""Meta-validity gate: check candidate consistency with meta-learning priors."""

from __future__ import annotations

from pysrc.tuning.core.meta_learning.experiment_memory import ExperimentMemory


def passes_meta_gate(
    candidate_score: float,
    memory: ExperimentMemory,
    space_hash: str,
    percentile: float = 0.25,
) -> bool:
    """Return True if candidate_score is in the top (1 - percentile) of past experiments."""
    past = memory.by_space(space_hash)
    if not past:
        return True
    sorted_scores = sorted(r.best_score for r in past)
    cutoff_idx = int(len(sorted_scores) * percentile)
    return candidate_score >= sorted_scores[cutoff_idx]


__all__ = ["passes_meta_gate"]

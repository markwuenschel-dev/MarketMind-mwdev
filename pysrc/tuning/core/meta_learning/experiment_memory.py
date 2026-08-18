"""Typed record of past tuning experiments for meta-learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ExperimentRecord:
    """Immutable record of a completed tuning experiment."""

    job_id: str
    space_hash: str
    best_candidate_id: str
    best_score: float
    n_trials: int
    completed_at: datetime
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentMemory:
    """Immutable collection of past experiment records."""

    records: tuple[ExperimentRecord, ...]

    def by_space(self, space_hash: str) -> tuple[ExperimentRecord, ...]:
        """Return all records for a given search space hash."""
        return tuple(r for r in self.records if r.space_hash == space_hash)

    def top_k(self, k: int) -> tuple[ExperimentRecord, ...]:
        """Return the top-k records by best_score."""
        return tuple(sorted(self.records, key=lambda r: r.best_score, reverse=True)[:k])


__all__ = ["ExperimentRecord", "ExperimentMemory"]

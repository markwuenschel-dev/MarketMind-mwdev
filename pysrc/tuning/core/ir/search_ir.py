"""SearchIR: canonical IR for the search algorithm state."""

from __future__ import annotations

from dataclasses import dataclass, field

from pysrc.tuning.core.ir.nodes import HParam, IRMetadata

__all__ = ["Trial", "SearchIR"]


@dataclass(frozen=True)
class Trial:
    """A single evaluated trial: params + scalar scores."""

    trial_id: str
    params: tuple[HParam, ...]
    scores: dict[str, float] = field(default_factory=dict)
    feasible: bool = True


@dataclass(frozen=True)
class SearchIR:
    """Immutable snapshot of search state after each update."""

    job_id: str
    algorithm: str
    space_hash: str
    meta: IRMetadata
    trials: tuple[Trial, ...] = field(default_factory=tuple)
    best_trial_id: str | None = None
    n_pending: int = 0

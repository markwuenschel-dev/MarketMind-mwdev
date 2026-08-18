"""SearchPlan: executable plan derived from SearchIR + budget."""

from __future__ import annotations

from dataclasses import dataclass

from pysrc.tuning.core.planning.plan_models import ExecutionBudget, PlanMetadata

__all__ = ["SearchStep", "SearchPlan"]


@dataclass(frozen=True)
class SearchStep:
    """One unit of work in the search plan: propose + evaluate N candidates."""

    step_index: int
    n_candidates: int
    algorithm_snapshot: str  # serialised algorithm state hash
    partition_id: str = ""


@dataclass(frozen=True)
class SearchPlan:
    """Executable search plan: an ordered sequence of SearchSteps."""

    job_id: str
    space_hash: str
    steps: tuple[SearchStep, ...]
    budget: ExecutionBudget
    meta: PlanMetadata

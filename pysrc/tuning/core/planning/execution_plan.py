"""ExecutionPlan: fully resolved plan for a complete tuning job."""

from __future__ import annotations

from dataclasses import dataclass

from pysrc.tuning.core.planning.plan_models import PlanMetadata
from pysrc.tuning.core.planning.search_plan import SearchPlan

__all__ = ["ExecutionPlan"]


@dataclass(frozen=True)
class ExecutionPlan:
    """Combines search, validation, and task plans into one executable unit."""

    job_id: str
    search_plan: SearchPlan
    n_folds: int
    n_candidates_per_step: int
    feature_hash: str
    meta: PlanMetadata

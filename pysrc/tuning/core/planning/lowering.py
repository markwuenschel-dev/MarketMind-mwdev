"""Lowering: pure functions that transform validated IR into executable plans."""

from __future__ import annotations

import time

from pysrc.tuning.core.ir.search_ir import SearchIR
from pysrc.tuning.core.planning.plan_models import ExecutionBudget, PlanMetadata
from pysrc.tuning.core.planning.search_plan import SearchPlan, SearchStep

__all__ = ["lower_search_ir"]


def lower_search_ir(
    ir: SearchIR,
    budget: ExecutionBudget,
    plan_hash: str,
) -> SearchPlan:
    """Lower a SearchIR + budget into an executable SearchPlan."""
    steps = tuple(
        SearchStep(
            step_index=i,
            n_candidates=budget.max_parallel,
            algorithm_snapshot=ir.meta.spec_hash,
        )
        for i in range(budget.max_trials)
    )
    meta = PlanMetadata(
        plan_hash=plan_hash,
        spec_hash=ir.meta.spec_hash,
        created_at_ns=time.monotonic_ns(),
        determinism_tier=ir.meta.determinism_tier,
    )
    return SearchPlan(
        job_id=ir.job_id,
        space_hash=ir.space_hash,
        steps=steps,
        budget=budget,
        meta=meta,
    )

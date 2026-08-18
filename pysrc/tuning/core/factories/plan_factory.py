"""PlanFactory: build plan objects from IR + budget."""

from __future__ import annotations

from pysrc.tuning.core.ir.search_ir import SearchIR
from pysrc.tuning.core.planning.lowering import lower_search_ir
from pysrc.tuning.core.planning.plan_models import ExecutionBudget
from pysrc.tuning.core.planning.search_plan import SearchPlan

__all__ = ["build_search_plan"]


def build_search_plan(
    ir: SearchIR,
    budget: ExecutionBudget,
    plan_hash: str,
) -> SearchPlan:
    """Delegate to lowering.lower_search_ir."""
    return lower_search_ir(ir, budget, plan_hash)

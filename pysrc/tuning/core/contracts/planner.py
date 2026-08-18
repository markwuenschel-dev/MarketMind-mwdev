"""PlannerProtocol: interface for IR → Plan lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pysrc.tuning.core.ir.search_ir import SearchIR
    from pysrc.tuning.core.planning.search_plan import SearchPlan


@runtime_checkable
class PlannerProtocol(Protocol):
    """Converts validated IR into an executable plan."""

    def lower(self, ir: SearchIR) -> SearchPlan: ...

    def validate_plan(self, plan: SearchPlan) -> bool: ...


__all__ = ["PlannerProtocol"]

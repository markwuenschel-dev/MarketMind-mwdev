"""Typed planning and IR-lowering errors."""

from __future__ import annotations


class PlanningError(RuntimeError):
    """Base class for planning and lowering errors."""


class LoweringError(PlanningError):
    """Raised when IR cannot be lowered to a valid plan."""


class InfeasiblePlanError(PlanningError):
    """Raised when a plan is structurally infeasible (e.g. zero steps, budget conflict)."""


__all__ = ["PlanningError", "LoweringError", "InfeasiblePlanError"]

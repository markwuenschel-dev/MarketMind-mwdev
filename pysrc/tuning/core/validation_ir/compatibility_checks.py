"""Cross-IR compatibility checks: ensure spec versions and hash references are consistent."""

from __future__ import annotations

from pysrc.tuning.core.ir.objective_ir import ObjectiveIR
from pysrc.tuning.core.ir.search_ir import SearchIR
from pysrc.tuning.core.ir.validation_ir import ValidationIR

__all__ = [
    "CompatibilityError",
    "check_search_validation_compatibility",
    "check_search_objective_compatibility",
]


class CompatibilityError(ValueError):
    """Raised when two IR objects are incompatible (version mismatch, hash mismatch, etc.)."""


def check_search_validation_compatibility(search: SearchIR, validation: ValidationIR) -> None:
    """Verify search and validation IRs reference the same job."""
    if search.job_id != validation.job_id:
        raise CompatibilityError(
            f"SearchIR job_id '{search.job_id}' != ValidationIR job_id '{validation.job_id}'"
        )


def check_search_objective_compatibility(search: SearchIR, objective: ObjectiveIR) -> None:
    """Verify search and objective IRs reference the same job."""
    if search.job_id != objective.job_id:
        raise CompatibilityError(
            f"SearchIR job_id '{search.job_id}' != ObjectiveIR job_id '{objective.job_id}'"
        )

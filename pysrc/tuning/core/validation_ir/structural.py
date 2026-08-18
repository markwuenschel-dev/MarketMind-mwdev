"""Structural IR validation: checks node counts, non-empty fields, type invariants."""

from __future__ import annotations

from pysrc.tuning.core.ir.search_ir import SearchIR
from pysrc.tuning.core.ir.task_ir import TaskIR

__all__ = ["StructuralValidationError", "validate_search_ir", "validate_task_ir"]


class StructuralValidationError(ValueError):
    """Raised when an IR object violates a structural invariant."""


def validate_search_ir(ir: SearchIR) -> SearchIR:
    """Assert structural invariants on a SearchIR; return it if valid."""
    if not ir.job_id:
        raise StructuralValidationError("SearchIR.job_id must be non-empty")
    if not ir.algorithm:
        raise StructuralValidationError("SearchIR.algorithm must be non-empty")
    if not ir.space_hash:
        raise StructuralValidationError("SearchIR.space_hash must be non-empty")
    return ir


def validate_task_ir(ir: TaskIR) -> TaskIR:
    """Assert structural invariants on a TaskIR; return it if valid."""
    if not ir.task_id:
        raise StructuralValidationError("TaskIR.task_id must be non-empty")
    if not ir.candidate_id:
        raise StructuralValidationError("TaskIR.candidate_id must be non-empty")
    if ir.fold.train_end <= ir.fold.train_start:
        raise StructuralValidationError("Fold train window is empty or reversed")
    if ir.fold.test_end <= ir.fold.test_start:
        raise StructuralValidationError("Fold test window is empty or reversed")
    return ir

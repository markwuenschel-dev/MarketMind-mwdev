"""Point-in-time (PIT) boundary checks: ensure no future data leaks into training windows."""

from __future__ import annotations

from datetime import datetime

from pysrc.tuning.core.ir.task_ir import FoldBoundary, TaskIR

__all__ = ["PITViolationError", "validate_fold_pit", "validate_no_leakage", "validate_task_pit"]


class PITViolationError(ValueError):
    """Raised when a fold boundary would allow future data into training."""


def validate_fold_pit(fold: FoldBoundary, as_of: datetime) -> FoldBoundary:
    """Ensure no fold boundary extends beyond *as_of*."""
    if fold.train_end > as_of:
        raise PITViolationError(f"Fold train_end {fold.train_end} exceeds as_of boundary {as_of}")
    if fold.test_end > as_of:
        raise PITViolationError(f"Fold test_end {fold.test_end} exceeds as_of boundary {as_of}")
    return fold


def validate_no_leakage(fold: FoldBoundary) -> FoldBoundary:
    """Ensure test window starts strictly after train window ends."""
    if fold.test_start <= fold.train_end:
        raise PITViolationError(
            f"Test window starts at {fold.test_start} before train_end {fold.train_end}: "
            "embargo required between train and test"
        )
    return fold


def validate_task_pit(task_ir: TaskIR, as_of: datetime) -> TaskIR:
    """Run all PIT checks against a TaskIR; return it if all pass."""
    validate_fold_pit(task_ir.fold, as_of)
    validate_no_leakage(task_ir.fold)
    return task_ir

"""Walk-forward cross-validation split generator."""

from __future__ import annotations

from datetime import datetime, timedelta

from pysrc.tuning.core.ir.task_ir import FoldBoundary
from pysrc.tuning.core.ir.validation_ir import ValidationIR


def walkforward_splits(
    ir: ValidationIR,
    start: datetime,
    end: datetime,
    bar_duration: timedelta,
) -> list[FoldBoundary]:
    """Return walk-forward FoldBoundaries with embargo from a ValidationIR config."""
    from pysrc.tuning.core.tasks.segmentation import split_walkforward

    return split_walkforward(
        start=start,
        end=end,
        n_splits=ir.n_splits,
        embargo_bars=ir.embargo.periods,
        bar_duration=bar_duration,
    )


__all__ = ["walkforward_splits"]

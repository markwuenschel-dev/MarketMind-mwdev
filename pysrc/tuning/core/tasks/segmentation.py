"""Pure time-series segmentation logic for task partitioning."""

from __future__ import annotations

from datetime import datetime, timedelta

from pysrc.tuning.core.ir.task_ir import FoldBoundary

__all__ = ["split_walkforward"]


def split_walkforward(
    start: datetime,
    end: datetime,
    n_splits: int,
    embargo_bars: int,
    bar_duration: timedelta,
) -> list[FoldBoundary]:
    """Generate walk-forward FoldBoundaries with an embargo gap.

    Each fold's train window grows from `start` to the fold boundary; the test
    window follows after an embargo of `embargo_bars * bar_duration` seconds.
    """
    total = (end - start).total_seconds()
    fold_len = total / n_splits
    embargo_seconds = embargo_bars * bar_duration.total_seconds()
    folds: list[FoldBoundary] = []
    for i in range(n_splits):
        train_start = start
        train_end = start + timedelta(seconds=fold_len * (i + 1))
        test_start = train_end + timedelta(seconds=embargo_seconds)
        test_end = test_start + timedelta(seconds=fold_len)
        if test_end > end:
            test_end = end
        if test_start >= test_end:
            break
        folds.append(
            FoldBoundary(
                fold_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds

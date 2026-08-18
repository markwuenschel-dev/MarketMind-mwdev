"""Replay consistency: verify re-running a task with the same seed produces identical splits."""

from __future__ import annotations

import pandas as pd


def assert_splits_identical(
    splits_a: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    splits_b: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> None:
    """Raise AssertionError if two split sets differ (D0 determinism check)."""
    if len(splits_a) != len(splits_b):
        raise AssertionError(f"Split counts differ: {len(splits_a)} vs {len(splits_b)}")
    for i, ((tr_a, te_a), (tr_b, te_b)) in enumerate(zip(splits_a, splits_b, strict=False)):
        if not tr_a.equals(tr_b):
            raise AssertionError(f"Train split {i} differs between runs")
        if not te_a.equals(te_b):
            raise AssertionError(f"Test split {i} differs between runs")


__all__ = ["assert_splits_identical"]

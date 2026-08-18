"""Purged cross-validation: removes training samples that overlap the test window."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def purge_training_index(
    train_idx: pd.DatetimeIndex,
    test_start: datetime,
    test_end: datetime,
    embargo_periods: int,
    bar_duration: timedelta,
) -> pd.DatetimeIndex:
    """Remove training observations whose label window overlaps the test set."""
    embargo_delta = embargo_periods * bar_duration
    return train_idx[train_idx < pd.Timestamp(test_start - embargo_delta)]


def purged_splits(
    index: pd.DatetimeIndex,
    n_splits: int,
    embargo_periods: int,
    bar_duration: timedelta,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return (train, test) purged index pairs for n_splits folds."""
    fold_size = len(index) // n_splits
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for i in range(n_splits):
        test_idx = index[i * fold_size : (i + 1) * fold_size]
        if len(test_idx) == 0:
            continue
        raw_train = index[: i * fold_size]
        test_start = test_idx[0].to_pydatetime()
        test_end = test_idx[-1].to_pydatetime()
        train_idx = purge_training_index(
            raw_train, test_start, test_end, embargo_periods, bar_duration
        )
        splits.append((train_idx, test_idx))
    return splits


__all__ = ["purge_training_index", "purged_splits"]

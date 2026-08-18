"""CPCV: combinatorial purged cross-validation split generation."""

from __future__ import annotations

from datetime import timedelta
from itertools import combinations

import pandas as pd


def cpcv_splits(
    index: pd.DatetimeIndex,
    n_splits: int,
    n_test_splits: int,
    embargo_periods: int,
    bar_duration: timedelta,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return CPCV (train, test) pairs: all C(n_splits, n_test_splits) combinations."""
    fold_size = len(index) // n_splits
    folds = [index[i * fold_size : (i + 1) * fold_size] for i in range(n_splits)]
    result: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for test_combo in combinations(range(n_splits), n_test_splits):
        test_set = set(test_combo)
        test_parts = [folds[i] for i in sorted(test_set)]
        test_idx = test_parts[0]
        for part in test_parts[1:]:
            test_idx = test_idx.union(part)
        train_folds = [folds[i] for i in range(n_splits) if i not in test_set]
        if not train_folds:
            continue
        train_idx = train_folds[0]
        for f in train_folds[1:]:
            train_idx = train_idx.union(f)
        if len(test_idx):
            from pysrc.tuning.core.validation.purged_cv import purge_training_index

            test_start = test_idx[0].to_pydatetime()
            test_end = test_idx[-1].to_pydatetime()
            train_idx = purge_training_index(
                train_idx, test_start, test_end, embargo_periods, bar_duration
            )
        result.append((train_idx, test_idx))
    return result


__all__ = ["cpcv_splits"]

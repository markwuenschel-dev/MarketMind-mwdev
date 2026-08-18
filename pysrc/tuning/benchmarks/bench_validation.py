"""Benchmark: measure throughput of purged cross-validation split generation."""

from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd


def bench_purged_splits(
    n_obs: int = 2000,
    n_splits: int = 5,
    embargo_periods: int = 10,
) -> dict[str, float]:
    """Measure time to generate purged CV splits for n_obs observations."""
    from pysrc.tuning.core.validation.purged_cv import purged_splits

    idx = pd.date_range(start="2018-01-01", periods=n_obs, freq="B")
    start = time.perf_counter()
    splits = purged_splits(idx, n_splits, embargo_periods, timedelta(days=1))
    elapsed = time.perf_counter() - start
    return {
        "n_obs": float(n_obs),
        "n_splits": float(n_splits),
        "n_splits_produced": float(len(splits)),
        "elapsed_s": elapsed,
    }


__all__ = ["bench_purged_splits"]

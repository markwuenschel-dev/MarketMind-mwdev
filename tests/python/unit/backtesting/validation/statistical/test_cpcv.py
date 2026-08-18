"""Unit tests for combinatorial purged cross-validation skeleton."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pysrc.backtesting.validation.statistical.cpcv import (
    CPCVConfig,
    CPCVDataError,
    CPCVSplitter,
    cpcv_splits,
)


@pytest.mark.determinism("d1")
def test_cpcv_config_n_combinations(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = CPCVConfig(n_splits=6, n_test_splits=2)
    assert cfg.n_combinations == math.comb(6, 2)


@pytest.mark.determinism("d1")
def test_cpcv_splitter_yields_expected_count(deterministic_seed: int) -> None:
    _ = deterministic_seed
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    frame = pd.DataFrame({"ret": np.linspace(-0.01, 0.01, len(index))}, index=index)
    cfg = CPCVConfig(n_splits=4, n_test_splits=2, min_train_size=10)
    splits = list(CPCVSplitter(cfg).split(frame))
    assert len(splits) == math.comb(4, 2)
    for split in splits:
        assert split.n_train >= cfg.min_train_size
        assert split.n_test > 0
        assert len(np.intersect1d(split.train_indices, split.test_indices)) == 0


@pytest.mark.determinism("d1")
def test_cpcv_splits_convenience_wrapper(deterministic_seed: int) -> None:
    _ = deterministic_seed
    index = pd.date_range("2021-01-01", periods=60, freq="B")
    series = pd.Series(np.zeros(len(index)), index=index)
    splits = cpcv_splits(series, n_splits=3, n_test_splits=1, min_train_size=5)
    assert len(splits) == 3


@pytest.mark.determinism("d1")
def test_cpcv_config_rejects_invalid_test_splits(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(CPCVDataError, match="n_test_splits"):
        CPCVConfig(n_splits=4, n_test_splits=4)

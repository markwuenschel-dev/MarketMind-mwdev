# py/backtesting/validation/statistical/cpcv.py
"""
Combinatorial Purged Cross-Validation (CPCV).

Generates all C(N, k) train/test splits from N groups, with purging and
embargo applied at each split boundary to prevent leakage.

Distinct from py/preprocessor/splits.py (which handles ML train/test
purge/embargo). This module is for combinatorial backtest validation —
evaluating every possible combination of train/test paths so that
performance distribution is computed over all paths, not just one.

Reference:
    López de Prado (2018) "Advances in Financial Machine Learning",
    Chapter 12 — Combinatorial Purged Cross-Validation.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pysrc.core.errors import BaseError
from pysrc.core.runtime.optional_imports import optional_import
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

pl = optional_import("polars")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CPCVError(BaseError):
    """Base for CPCV errors."""


class CPCVDataError(CPCVError):
    """Invalid input to CPCV splitter."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CPCVSplit:
    """A single CPCV train/test split."""

    split_index: int  # ordinal index of this combination
    train_indices: np.ndarray  # row indices into the original DataFrame
    test_indices: np.ndarray  # row indices (after purge + embargo)
    test_group_ids: list[int]  # which of the N groups are in the test set
    n_train: int = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_train = len(self.train_indices)
        self.n_test = len(self.test_indices)


@dataclass
class CPCVConfig:
    """Configuration for a CPCV run."""

    n_splits: int = 6  # N: number of groups to divide the data into
    n_test_splits: int = 2  # k: number of groups held out per combination
    purge_periods: int = 0  # bars to purge at each train/test boundary
    embargo_periods: int = 0  # bars to embargo after each test group ends
    min_train_size: int = 30  # minimum training observations required

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise CPCVDataError("n_splits must be >= 2", details={"n_splits": self.n_splits})
        if self.n_test_splits < 1 or self.n_test_splits >= self.n_splits:
            raise CPCVDataError(
                "n_test_splits must be in [1, n_splits - 1]",
                details={"n_test_splits": self.n_test_splits, "n_splits": self.n_splits},
            )
        if self.purge_periods < 0:
            raise CPCVDataError(
                "purge_periods must be >= 0", details={"purge_periods": self.purge_periods}
            )
        if self.embargo_periods < 0:
            raise CPCVDataError(
                "embargo_periods must be >= 0", details={"embargo_periods": self.embargo_periods}
            )

    @property
    def n_combinations(self) -> int:
        """Total number of C(N, k) train/test splits."""
        return math.comb(self.n_splits, self.n_test_splits)


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------


class CPCVSplitter:
    """
    Generates all C(N, k) combinatorial purged cross-validation splits.

    Usage:
        splitter = CPCVSplitter(CPCVConfig(n_splits=6, n_test_splits=2,
                                           purge_periods=5, embargo_periods=2))
        for split in splitter.split(returns_df):
            model.fit(returns_df.iloc[split.train_indices])
            preds = model.predict(returns_df.iloc[split.test_indices])
    """

    def __init__(self, config: CPCVConfig | None = None) -> None:
        self.config = config or CPCVConfig()

    def split(
        self,
        data: Any,
        *,
        return_splits_only: bool = False,
    ) -> Generator[CPCVSplit, None, None]:
        """
        Yield CPCVSplit objects for every C(N, k) combination.

        Args:
            data:               DataFrame or Series indexed by time. Row order
                                must be chronological. Index is not modified.
            return_splits_only: If True, yield index arrays only (faster, no
                                dataclass overhead). For internal use.

        Yields:
            CPCVSplit for each of the C(N, k) combinations.
        """
        idx = self._validate_and_get_index(data)
        n = len(idx)
        cfg = self.config

        # Divide into N roughly equal groups
        group_edges = np.array_split(np.arange(n), cfg.n_splits)
        groups: list[np.ndarray] = [g for g in group_edges if len(g) > 0]
        actual_n = len(groups)

        if actual_n < cfg.n_splits:
            LOG.warning(
                "cpcv_fewer_groups_than_requested",
                requested=cfg.n_splits,
                actual=actual_n,
                n_obs=n,
            )

        LOG.info(
            "cpcv_split_start",
            n_obs=n,
            n_splits=actual_n,
            n_test_splits=cfg.n_test_splits,
            n_combinations=math.comb(actual_n, cfg.n_test_splits),
            purge=cfg.purge_periods,
            embargo=cfg.embargo_periods,
        )

        for split_idx, test_group_ids in enumerate(
            itertools.combinations(range(actual_n), cfg.n_test_splits)
        ):
            test_group_set = set(test_group_ids)
            train_group_ids = [i for i in range(actual_n) if i not in test_group_set]

            # Raw test indices (all rows in test groups)
            raw_test = np.concatenate([groups[i] for i in sorted(test_group_ids)])

            # Raw train indices
            raw_train = np.concatenate([groups[i] for i in train_group_ids])

            # Apply purge and embargo
            test_indices = self._apply_purge_embargo(
                raw_test, raw_train, n, cfg.purge_periods, cfg.embargo_periods
            )
            train_indices = raw_train  # train is never modified — only test boundary is cleaned

            if len(train_indices) < cfg.min_train_size:
                LOG.warning(
                    "cpcv_split_skipped_insufficient_train",
                    split_index=split_idx,
                    n_train=len(train_indices),
                    min_required=cfg.min_train_size,
                )
                continue

            if len(test_indices) == 0:
                LOG.warning(
                    "cpcv_split_skipped_empty_test",
                    split_index=split_idx,
                    test_group_ids=list(test_group_ids),
                )
                continue

            yield CPCVSplit(
                split_index=split_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                test_group_ids=list(test_group_ids),
            )

    def n_splits_total(self, n_obs: int) -> int:
        """Return total number of valid splits for a dataset of n_obs rows."""
        return math.comb(self.config.n_splits, self.config.n_test_splits)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_get_index(data: Any) -> pd.Index:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise CPCVDataError("Input DataFrame is empty")
            return data.index
        if isinstance(data, pd.Series):
            if data.empty:
                raise CPCVDataError("Input Series is empty")
            return data.index
        if pl and isinstance(data, (pl.DataFrame, pl.Series)):
            n = data.shape[0] if isinstance(data, pl.DataFrame) else len(data)
            return pd.RangeIndex(n)
        if isinstance(data, np.ndarray):
            return pd.RangeIndex(len(data.ravel()))
        raise CPCVDataError(
            "Unsupported data type for CPCV split",
            details={"type": type(data).__name__},
        )

    @staticmethod
    def _apply_purge_embargo(
        test_idx: np.ndarray,
        train_idx: np.ndarray,
        n_total: int,
        purge: int,
        embargo: int,
    ) -> np.ndarray:
        """
        Remove rows from test_idx that fall within purge/embargo windows
        adjacent to training boundaries.

        Purge: removes test rows within `purge` bars BEFORE any train group.
        Embargo: removes test rows within `embargo` bars AFTER any train group.
        """
        if purge == 0 and embargo == 0:
            return test_idx

        # Build set of train indices for fast lookup
        train_set = set(train_idx.tolist())

        # Find boundary rows: train rows adjacent to test rows
        set(test_idx.tolist())
        forbidden: set[int] = set()

        for t in test_idx:
            # Purge: t is too close to the START of a train segment
            for offset in range(1, purge + 1):
                if (t + offset) in train_set:
                    forbidden.add(t)
                    break
            # Embargo: t is too close to the END of a train segment
            for offset in range(1, embargo + 1):
                if (t - offset) in train_set:
                    forbidden.add(t)
                    break

        clean_test = np.array([t for t in test_idx if t not in forbidden], dtype=np.intp)
        return clean_test


# ---------------------------------------------------------------------------
# Convenience function: collect all split indices as lists
# ---------------------------------------------------------------------------


def cpcv_splits(
    data: Any,
    n_splits: int = 6,
    n_test_splits: int = 2,
    purge_periods: int = 0,
    embargo_periods: int = 0,
    min_train_size: int = 30,
) -> list[CPCVSplit]:
    """
    Return all CPCV splits as a list.

    Convenience wrapper around CPCVSplitter.split() for cases where
    you want all splits in memory (e.g., for parallel evaluation).

    Returns:
        List of CPCVSplit, one per C(N, k) combination.
    """
    cfg = CPCVConfig(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        purge_periods=purge_periods,
        embargo_periods=embargo_periods,
        min_train_size=min_train_size,
    )
    splitter = CPCVSplitter(cfg)
    return list(splitter.split(data))

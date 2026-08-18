"""
Property-based tests for data leakage invariants in MarketMind.

Tests the 5 critical leakage patterns that MUST be caught per the Phase 0 checklist:
1. Temporal ordering: max(train_timestamps) < min(test_timestamps)
2. Sample disjointness: train_indices ∩ test_indices = ∅
3. Purge gap: test_start - train_end ≥ purge_window
4. Fit-on-train-only: Scaler params computed only from train
5. No look-ahead: Features at time t use only data ≤ t

Reference: MarketMind Implementation Plan v5.1, Known Leakage Patterns section

Integration:
    - Uses pysrc.preprocessor.splits (TimeSeriesSplitter, PurgedKFold, SplitResult, validate_no_leakage)
    - Uses pysrc.preprocessor.core (add_returns, add_sma, add_rsi)
    - All data uses Polars DataFrames to match production code
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pysrc.preprocessor.core import (
    add_returns,
    add_sma,
)

# =============================================================================
# Production Imports
# =============================================================================
from pysrc.preprocessor.splits import (
    SPLITS_SCHEMA_VERSION,
    PurgedKFold,
    SplitBoundary,
    SplitResult,
    TimeSeriesSplitter,
    create_splits_manifest,
    validate_no_leakage,
)

# =============================================================================
# Test Configuration
# =============================================================================

SLOW_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

FAST_SETTINGS = settings(
    max_examples=50,
    deadline=timedelta(seconds=10),
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def ohlcv_dataframe_strategy(draw, min_rows: int = 100, max_rows: int = 1000) -> pl.DataFrame:
    """Generate a time series OHLCV DataFrame suitable for splitting."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    start_date = datetime(2020, 1, 1)
    dates = []
    current = start_date
    for _ in range(n_rows):
        dates.append(current.date())
        days_forward = draw(st.sampled_from([1, 1, 1, 1, 1, 3, 3]))
        current += timedelta(days=days_forward)

    initial_price = draw(st.floats(min_value=20.0, max_value=500.0))
    daily_vol = draw(st.floats(min_value=0.005, max_value=0.03))
    log_returns = np.random.normal(0, daily_vol, n_rows)
    close_prices = initial_price * np.exp(np.cumsum(log_returns))

    intraday_range = np.abs(np.random.normal(0, daily_vol * 1.5, n_rows))
    open_prices = close_prices * (1 + np.random.uniform(-0.005, 0.005, n_rows))
    high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range)
    low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range)
    high_prices = np.maximum.reduce([high_prices, open_prices, close_prices])
    low_prices = np.minimum.reduce([low_prices, open_prices, close_prices])
    volumes = np.random.randint(100_000, 10_000_000, n_rows)

    return pl.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    ).with_columns(pl.col("date").cast(pl.Date))


# =============================================================================
# Mock Scaler
# =============================================================================


class StandardScaler:
    """Simple scaler for testing fit-on-train invariant."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise ValueError("Scaler not fitted")
        return (X - self.mean_) / (self.std_ + 1e-8)


# =============================================================================
# Leaky Implementations for Detection Testing
# =============================================================================


class LeakyTimeSeriesSplitter:
    """Intentionally leaky splitter for testing invariant detection."""

    def __init__(
        self,
        leak_type: str = "shuffle",
        n_splits: int = 3,
        test_size: int = 50,
        min_train_size: int = 100,
        timestamp_col: str = "date",
    ):
        self.leak_type = leak_type
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.timestamp_col = timestamp_col

    def split(self, df: pl.DataFrame) -> Iterator[SplitResult]:
        n_rows = df.height
        df_indexed = df.with_row_index("_idx")

        for fold_id in range(self.n_splits):
            folds_remaining = self.n_splits - 1 - fold_id
            test_end_idx = n_rows - (folds_remaining * self.test_size)
            test_start_idx = test_end_idx - self.test_size
            train_end_idx = test_start_idx

            if self.leak_type == "shuffle":
                all_indices = list(range(n_rows))
                np.random.shuffle(all_indices)
                train_indices = all_indices[:train_end_idx]
                test_indices = all_indices[test_start_idx:test_end_idx]
            elif self.leak_type == "overlap":
                train_indices = list(range(0, train_end_idx + 5))
                test_indices = list(range(test_start_idx, test_end_idx))
            elif self.leak_type == "no_purge":
                train_indices = list(range(0, train_end_idx))
                test_indices = list(range(test_start_idx, test_end_idx))
            else:
                raise ValueError(f"Unknown leak_type: {self.leak_type}")

            train_df = df_indexed.filter(pl.col("_idx").is_in(train_indices)).drop("_idx")
            test_df = df_indexed.filter(pl.col("_idx").is_in(test_indices)).drop("_idx")

            boundary = SplitBoundary(
                fold_id=fold_id,
                train_start=train_df[self.timestamp_col][0]
                if train_df.height > 0
                else datetime.now(),
                train_end=train_df[self.timestamp_col][-1]
                if train_df.height > 0
                else datetime.now(),
                test_start=test_df[self.timestamp_col][0] if test_df.height > 0 else datetime.now(),
                test_end=test_df[self.timestamp_col][-1] if test_df.height > 0 else datetime.now(),
                train_count=train_df.height,
                test_count=test_df.height,
                purged_count=0,
            )

            yield SplitResult(
                train_df=train_df,
                test_df=test_df,
                boundary=boundary,
                _train_indices=train_indices,
                _test_indices=test_indices,
            )


def add_sma_leaky(df: pl.DataFrame, column: str = "close", window: int = 20) -> pl.DataFrame:
    """Leaky SMA using centered window (includes future data)."""
    return df.with_columns(
        pl.col(column).rolling_mean(window_size=window, center=True).alias(f"sma_{window}_leaky")
    )


# =============================================================================
# Invariant Assertion Functions
# =============================================================================


def assert_temporal_ordering(result: SplitResult, timestamp_col: str = "date") -> None:
    """Invariant 1: max(train_timestamps) < min(test_timestamps)"""
    if result.train_df.height == 0 or result.test_df.height == 0:
        return
    train_max = result.train_df[timestamp_col].max()
    test_min = result.test_df[timestamp_col].min()
    assert train_max < test_min, (
        f"Temporal ordering violated: max(train)={train_max} >= min(test)={test_min}"
    )


def assert_disjoint_indices(result: SplitResult) -> None:
    """Invariant 2: train_indices ∩ test_indices = ∅"""
    train_set = set(result.train_indices)
    test_set = set(result.test_indices)
    overlap = train_set & test_set
    assert len(overlap) == 0, f"Sample overlap detected: {len(overlap)} indices in both"


def assert_purge_gap(
    result: SplitResult, required_purge_days: int, timestamp_col: str = "date"
) -> None:
    """Invariant 3: test_start - train_end ≥ purge_window"""
    if result.train_df.height == 0 or result.test_df.height == 0:
        return
    train_end = result.train_df[timestamp_col].max()
    test_start = result.test_df[timestamp_col].min()

    if hasattr(train_end, "year"):
        train_end_dt = datetime(train_end.year, train_end.month, train_end.day)
    else:
        train_end_dt = train_end
    if hasattr(test_start, "year"):
        test_start_dt = datetime(test_start.year, test_start.month, test_start.day)
    else:
        test_start_dt = test_start

    actual_gap = (test_start_dt - train_end_dt).days
    assert actual_gap >= required_purge_days, (
        f"Purge gap violation: actual={actual_gap} < required={required_purge_days}"
    )


def assert_scaler_fit_on_train_only(
    scaler: StandardScaler, train_data: np.ndarray, full_data: np.ndarray
) -> None:
    """Invariant 4: Scaler .mean_ computed only from train"""
    train_mean = np.mean(train_data, axis=0)
    full_mean = np.mean(full_data, axis=0)

    assert np.allclose(scaler.mean_, train_mean, rtol=1e-5), (
        f"Scaler mean does not match train mean. Scaler: {scaler.mean_}, Train: {train_mean}"
    )
    if not np.allclose(train_mean, full_mean, rtol=0.05):
        assert not np.allclose(scaler.mean_, full_mean, rtol=1e-3), (
            "LEAKAGE DETECTED: Scaler fit on full dataset, not train only"
        )


def assert_no_lookahead_in_feature(
    df: pl.DataFrame, feature_col: str, source_col: str, window: int, offset: int = 0
) -> None:
    """Invariant 5: Features at time t use only data ≤ t

    Args:
        offset: Starting index in original data (use when passing a sliced DataFrame)
    """
    source_values = df[source_col].to_numpy()
    feature_values = df[feature_col].to_numpy()

    for i in range(len(feature_values)):
        original_idx = i + offset
        if np.isnan(feature_values[i]):
            continue
        if original_idx < window - 1:
            raise AssertionError(
                f"Non-NaN at index {original_idx} with only {original_idx + 1} values. Possible look-ahead bias."
            )
        expected = np.mean(source_values[max(0, i - window + 1) : i + 1])
        if not np.isclose(feature_values[i], expected, rtol=1e-5):
            raise AssertionError(
                f"Feature at {original_idx}: got {feature_values[i]:.6f}, expected {expected:.6f}. Possible look-ahead bias."
            )


# =============================================================================
# Test Class: Invariant 1 - Temporal Ordering
# =============================================================================


class TestTemporalOrdering:
    """Tests for Invariant 1: max(train_timestamps) < min(test_timestamps)"""

    @given(df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1500))
    @FAST_SETTINGS
    def test_timeseries_splitter_maintains_temporal_order(self, df: pl.DataFrame) -> None:
        splitter = TimeSeriesSplitter(
            n_splits=3, test_size=50, min_train_size=100, purge_window_days=5
        )
        for result in splitter.split(df):
            assert_temporal_ordering(result)

    @given(df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1500))
    @FAST_SETTINGS
    def test_purged_kfold_no_timestamp_overlap(self, df: pl.DataFrame) -> None:
        splitter = PurgedKFold(n_splits=5, purge_window_days=3, embargo_window_days=2)
        for result in splitter.split(df):
            train_ts = set(result.train_df["date"].to_list())
            test_ts = set(result.test_df["date"].to_list())
            assert len(train_ts & test_ts) == 0

    @given(df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1200))
    @SLOW_SETTINGS
    def test_validate_no_leakage_catches_temporal_issues(self, df: pl.DataFrame) -> None:
        splitter = TimeSeriesSplitter(
            n_splits=3, test_size=50, min_train_size=100, purge_window_days=5
        )
        for result in splitter.split(df):
            validation = validate_no_leakage(
                result.train_df, result.test_df, timestamp_col="date", purge_window_days=5
            )
            assert validation["valid"], f"Validation failed: {validation['violations']}"


# =============================================================================
# Test Class: Invariant 2 - Sample Disjointness
# =============================================================================


class TestSampleDisjointness:
    """Tests for Invariant 2: train_indices ∩ test_indices = ∅"""

    @given(df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1500))
    @FAST_SETTINGS
    def test_timeseries_splitter_disjoint(self, df: pl.DataFrame) -> None:
        splitter = TimeSeriesSplitter(
            n_splits=3, test_size=50, min_train_size=100, purge_window_days=5
        )
        for result in splitter.split(df):
            assert_disjoint_indices(result)

    @given(df=ohlcv_dataframe_strategy(min_rows=500, max_rows=1000))
    @FAST_SETTINGS
    def test_purged_kfold_disjoint(self, df: pl.DataFrame) -> None:
        splitter = PurgedKFold(n_splits=5, purge_window_days=3, embargo_window_days=2)
        for result in splitter.split(df):
            assert_disjoint_indices(result)

    @given(df=ohlcv_dataframe_strategy(min_rows=500, max_rows=800))
    @FAST_SETTINGS
    def test_detects_overlapping_samples(self, df: pl.DataFrame) -> None:
        leaky = LeakyTimeSeriesSplitter(
            leak_type="overlap", n_splits=2, test_size=50, min_train_size=100
        )
        overlaps_detected = 0
        for result in leaky.split(df):
            try:
                assert_disjoint_indices(result)
            except AssertionError:
                overlaps_detected += 1
        assert overlaps_detected > 0, "Failed to detect intentional overlap"


# =============================================================================
# Test Class: Invariant 3 - Purge Gap
# =============================================================================


class TestPurgeGap:
    """Tests for Invariant 3: test_start - train_end ≥ purge_window"""

    @given(
        df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1500),
        purge_days=st.integers(min_value=1, max_value=10),
    )
    @FAST_SETTINGS
    def test_timeseries_splitter_respects_purge(self, df: pl.DataFrame, purge_days: int) -> None:
        splitter = TimeSeriesSplitter(
            n_splits=3, test_size=50, min_train_size=100, purge_window_days=purge_days
        )
        for result in splitter.split(df):
            assert_purge_gap(result, purge_days)

    @given(df=ohlcv_dataframe_strategy(min_rows=500, max_rows=800))
    @FAST_SETTINGS
    def test_detects_purge_violation(self, df: pl.DataFrame) -> None:
        leaky = LeakyTimeSeriesSplitter(
            leak_type="no_purge", n_splits=2, test_size=50, min_train_size=100
        )
        violations = 0
        for result in leaky.split(df):
            try:
                assert_purge_gap(result, required_purge_days=5)
            except AssertionError:
                violations += 1
        assert violations > 0, "Failed to detect purge violation"


# =============================================================================
# Test Class: Invariant 4 - Fit-on-Train-Only
# =============================================================================


class TestFitOnTrainOnly:
    """Tests for Invariant 4: Scaler .mean_ computed only from train"""

    @given(df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1500))
    @FAST_SETTINGS
    def test_scaler_fit_on_train_passes(self, df: pl.DataFrame) -> None:
        splitter = TimeSeriesSplitter(
            n_splits=2, test_size=100, min_train_size=200, purge_window_days=5
        )
        for result in splitter.split(df):
            train_data = result.train_df.select(["close", "volume"]).to_numpy()
            full_data = df.select(["close", "volume"]).to_numpy()
            scaler = StandardScaler()
            scaler.fit(train_data)
            assert_scaler_fit_on_train_only(scaler, train_data, full_data)

    @given(
        train_size=st.integers(min_value=50, max_value=200),
        test_size=st.integers(min_value=20, max_value=100),
    )
    @FAST_SETTINGS
    def test_detects_fit_on_all_leakage(self, train_size: int, test_size: int) -> None:
        np.random.seed(42)
        train_data = np.random.normal(0, 1, (train_size, 3))
        test_data = np.random.normal(100, 10, (test_size, 3))
        full_data = np.vstack([train_data, test_data])
        leaky_scaler = StandardScaler()
        leaky_scaler.fit(full_data)
        with pytest.raises(AssertionError, match="LEAKAGE|does not match"):
            assert_scaler_fit_on_train_only(leaky_scaler, train_data, full_data)


# =============================================================================
# Test Class: Invariant 5 - No Look-Ahead Bias
# =============================================================================


class TestNoLookahead:
    """Tests for Invariant 5: Features at time t use only data ≤ t"""

    @given(
        df=ohlcv_dataframe_strategy(min_rows=200, max_rows=500),
        window=st.integers(min_value=5, max_value=50),
    )
    @FAST_SETTINGS
    def test_add_sma_no_lookahead(self, df: pl.DataFrame, window: int) -> None:
        assume(df.height > window + 10)
        df_with_sma = add_sma(df, column="close", window=window)
        assert_no_lookahead_in_feature(df_with_sma, f"sma_{window}", "close", window)

    @given(df=ohlcv_dataframe_strategy(min_rows=200, max_rows=500))
    @FAST_SETTINGS
    def test_add_returns_no_lookahead(self, df: pl.DataFrame) -> None:
        df_with_returns = add_returns(df, column="close")
        close = df["close"].to_numpy()
        returns = df_with_returns["returns"].to_numpy()
        assert np.isnan(returns[0]) or returns[0] is None
        for i in range(1, len(returns)):
            expected = (close[i] / close[i - 1]) - 1
            if not np.isnan(returns[i]):
                assert np.isclose(returns[i], expected, rtol=1e-5)

    @given(df=ohlcv_dataframe_strategy(min_rows=100, max_rows=300))
    @FAST_SETTINGS
    def test_detects_lookahead_in_leaky_sma(self, df: pl.DataFrame) -> None:
        df_leaky = add_sma_leaky(df, column="close", window=20)
        with pytest.raises(AssertionError, match="look-ahead|Possible"):
            assert_no_lookahead_in_feature(df_leaky, "sma_20_leaky", "close", 20)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCombinedInvariants:
    """Tests that verify all invariants together."""

    @given(df=ohlcv_dataframe_strategy(min_rows=1000, max_rows=2000))
    @SLOW_SETTINGS
    def test_full_pipeline_all_invariants(self, df: pl.DataFrame) -> None:
        df = add_sma(df, column="close", window=20)
        df = add_returns(df, column="close")
        splitter = TimeSeriesSplitter(
            n_splits=3, test_size=100, min_train_size=200, purge_window_days=5
        )
        full_data = df.select(["close", "volume"]).to_numpy()

        for result in splitter.split(df):
            assert_temporal_ordering(result)
            assert_disjoint_indices(result)
            assert_purge_gap(result, required_purge_days=5)
            train_data = result.train_df.select(["close", "volume"]).to_numpy()
            scaler = StandardScaler()
            scaler.fit(train_data)
            assert_scaler_fit_on_train_only(scaler, train_data, full_data)

        # Invariant 5: Verify SMA has no look-ahead on original DataFrame
        # Check that valid SMA values (from index 19 onwards) match expected
        close_vals = df["close"].to_numpy()
        sma_vals = df["sma_20"].to_numpy()
        for i in range(19, len(sma_vals)):
            if np.isnan(sma_vals[i]):
                continue
            expected = np.mean(close_vals[i - 19 : i + 1])
            assert np.isclose(sma_vals[i], expected, rtol=1e-5), (
                f"SMA at {i}: got {sma_vals[i]:.6f}, expected {expected:.6f}"
            )

    @given(df=ohlcv_dataframe_strategy(min_rows=800, max_rows=1500))
    @SLOW_SETTINGS
    def test_manifest_generation_consistency(self, df: pl.DataFrame) -> None:
        splitter = TimeSeriesSplitter(
            n_splits=3, test_size=50, min_train_size=100, purge_window_days=5
        )
        manifest = create_splits_manifest(df, splitter)
        assert manifest.schema_version == SPLITS_SCHEMA_VERSION
        assert manifest.n_splits == 3
        assert manifest.purge_window == 5
        assert len(manifest.splits) == 3


# =============================================================================
# Regression Tests: Known Leakage Patterns
# =============================================================================


class TestKnownLeakagePatterns:
    """Tests for specific leakage patterns from Implementation Plan v5.1."""

    def test_pattern_shuffled_time_series_detected(self) -> None:
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pl.DataFrame(
            {"date": dates, "close": np.linspace(100, 200, 100), "volume": [1000000] * 100}
        ).with_columns(pl.col("date").cast(pl.Date))
        leaky = LeakyTimeSeriesSplitter(
            leak_type="shuffle", n_splits=1, test_size=20, min_train_size=50
        )
        for result in leaky.split(df):
            validation = validate_no_leakage(result.train_df, result.test_df)
            train_sorted = result.train_df["date"].equals(result.train_df["date"].sort())
            if not train_sorted:
                assert (
                    not validation.get("checks", {})
                    .get("monotonicity", {})
                    .get("train_monotonic", True)
                )

    def test_pattern_overlapping_samples_detected(self) -> None:
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pl.DataFrame(
            {"date": dates, "close": np.linspace(100, 200, 100), "volume": [1000000] * 100}
        ).with_columns(pl.col("date").cast(pl.Date))
        leaky = LeakyTimeSeriesSplitter(
            leak_type="overlap", n_splits=1, test_size=20, min_train_size=50
        )
        for result in leaky.split(df):
            with pytest.raises(AssertionError, match="overlap"):
                assert_disjoint_indices(result)

    def test_pattern_purge_violation_detected(self) -> None:
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pl.DataFrame(
            {"date": dates, "close": np.linspace(100, 200, 100), "volume": [1000000] * 100}
        ).with_columns(pl.col("date").cast(pl.Date))
        leaky = LeakyTimeSeriesSplitter(
            leak_type="no_purge", n_splits=1, test_size=20, min_train_size=50
        )
        for result in leaky.split(df):
            with pytest.raises(AssertionError, match="Purge gap"):
                assert_purge_gap(result, required_purge_days=10)

    def test_pattern_fit_on_all_detected(self) -> None:
        np.random.seed(42)
        train_data = np.random.normal(0, 1, (100, 2))
        test_data = np.random.normal(100, 10, (50, 2))
        full_data = np.vstack([train_data, test_data])
        leaky_scaler = StandardScaler()
        leaky_scaler.fit(full_data)
        with pytest.raises(AssertionError, match="LEAKAGE|does not match"):
            assert_scaler_fit_on_train_only(leaky_scaler, train_data, full_data)

    def test_pattern_lookahead_bias_detected(self) -> None:
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pl.DataFrame(
            {"date": dates, "close": np.linspace(100, 200, 100), "volume": [1000000] * 100}
        ).with_columns(pl.col("date").cast(pl.Date))
        df_leaky = add_sma_leaky(df, column="close", window=10)
        with pytest.raises(AssertionError, match="look-ahead|Possible"):
            assert_no_lookahead_in_feature(df_leaky, "sma_10_leaky", "close", 10)

    def assert_embargo_gap(
        self: SplitResult, required_embargo_days: int, timestamp_col: str = "date"
    ) -> None:
        """Invariant: first_train_after_test - test_end >= embargo_window"""
        # Find training data that comes AFTER test period
        test_end = self.test_df[timestamp_col].max()
        train_after_test = self.train_df.filter(pl.col(timestamp_col) > test_end)

        if train_after_test.height == 0:
            return  # No train data after test (walk-forward case)

        first_train_after = train_after_test[timestamp_col].min()
        actual_gap = (first_train_after - test_end).days

        assert actual_gap >= required_embargo_days, (
            f"Embargo gap violation: actual={actual_gap} < required={required_embargo_days}"
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    np.random.seed(42)
    n = 500
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    return pl.DataFrame(
        {
            "date": dates,
            "open": prices * 1.001,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(100000, 10000000, n),
        }
    ).with_columns(pl.col("date").cast(pl.Date))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

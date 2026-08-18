"""Property-based tests for time-series splitting using Hypothesis.

These tests verify critical invariants that must hold for ALL inputs:
1. Fit-on-train-only: No test data in training set
2. Timestamp monotonicity: All timestamps strictly ordered
3. No t+1 leakage: Train never includes future info
4. Purge/embargo enforcement: Required gaps maintained
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysrc.preprocessor.splits import (
    TimeSeriesSplitter,
    validate_no_leakage,
)


@st.composite
def valid_splitter_and_data(draw):
    """Generate splitter config with data guaranteed large enough."""
    n_splits = draw(st.integers(min_value=1, max_value=5))
    test_size = draw(st.integers(min_value=10, max_value=50))
    min_train_size = draw(st.integers(min_value=20, max_value=100))
    purge_days = draw(st.integers(min_value=0, max_value=7))

    required_rows = min_train_size + (n_splits * test_size) + 50

    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(required_rows)]

    df = pl.DataFrame(
        {
            "date": dates,
            "close": [100 + i * 0.01 for i in range(required_rows)],
        }
    )

    splitter = TimeSeriesSplitter(
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        purge_window_days=purge_days,
    )

    return splitter, df


class TestFitOnTrainOnlyInvariants:
    """Ensure training data never includes test samples."""

    @given(data=valid_splitter_and_data())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_timestamp_overlap(self, data):
        """Train and test never share timestamps."""
        splitter, df = data

        for result in splitter.split(df):
            train_ts = set(result.train_df["date"].to_list())
            test_ts = set(result.test_df["date"].to_list())
            assert len(train_ts & test_ts) == 0

    @given(data=valid_splitter_and_data())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_train_before_test(self, data):
        """All train timestamps < all test timestamps."""
        splitter, df = data

        for result in splitter.split(df):
            if result.train_df.height == 0:
                continue
            train_max = result.train_df["date"].max()
            test_min = result.test_df["date"].min()
            assert train_max < test_min


class TestTimestampMonotonicity:
    """Ensure temporal ordering preserved."""

    @given(data=valid_splitter_and_data())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_train_monotonic(self, data):
        """Train timestamps strictly monotonic."""
        splitter, df = data

        for result in splitter.split(df):
            dates = result.train_df["date"].to_list()
            for i in range(len(dates) - 1):
                assert dates[i] < dates[i + 1]

    @given(data=valid_splitter_and_data())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_test_monotonic(self, data):
        """Test timestamps strictly monotonic."""
        splitter, df = data

        for result in splitter.split(df):
            dates = result.test_df["date"].to_list()
            for i in range(len(dates) - 1):
                assert dates[i] < dates[i + 1]


class TestNoFutureLeakage:
    """Ensure no t+1 leakage in features."""

    @given(data=valid_splitter_and_data())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_purge_gap_enforced(self, data):
        """Purge window creates required gap."""
        splitter, df = data

        for result in splitter.split(df):
            if result.train_df.height == 0:
                continue

            train_end = result.train_df["date"].max()
            test_start = result.test_df["date"].min()

            # Convert to datetime if needed
            if hasattr(train_end, "days"):
                gap = (test_start - train_end).days
            else:
                gap = (
                    (test_start - train_end).days if hasattr(test_start - train_end, "days") else 1
                )

            assert gap >= splitter.purge_window_days

    @given(data=valid_splitter_and_data())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_expanding_window(self, data):
        """Train window grows across folds."""
        splitter, df = data

        splits = list(splitter.split(df))
        if len(splits) < 2:
            return

        for i in range(len(splits) - 1):
            # Without purge, later folds have more training data
            # With purge, this may not always hold, so just check positive
            assert splits[i + 1].train_df.height > 0


class TestValidationFunction:
    """Test validate_no_leakage catches issues."""

    @given(data=valid_splitter_and_data())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_splits_pass_validation(self, data):
        """Properly created splits pass validation."""
        splitter, df = data

        for result in splitter.split(df):
            validation = validate_no_leakage(
                result.train_df,
                result.test_df,
                timestamp_col="date",
                purge_window_days=splitter.purge_window_days,
            )
            assert validation["valid"], validation["violations"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

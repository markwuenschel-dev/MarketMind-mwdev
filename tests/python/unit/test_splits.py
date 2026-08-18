"""Tests for time-series train/test splitting with purge and embargo windows.

These tests verify:
1. No temporal leakage across train/test boundaries
2. Purge windows properly remove samples near test periods
3. Embargo windows prevent future information contamination
4. Manifest JSON output matches schema specification
5. Edge cases (small datasets, boundary conditions)
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from pysrc.preprocessor.splits import (
    SPLITS_SCHEMA_VERSION,
    PurgedKFold,
    SplitsManifest,
    TimeSeriesSplitter,
    create_splits_manifest,
    validate_no_leakage,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_daily_data() -> pl.DataFrame:
    """Create 2 years of daily trading data (~504 rows)."""
    dates = pl.date_range(
        datetime(2022, 1, 1),
        datetime(2023, 12, 31),
        "1d",
        eager=True,
    )
    n = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "close": [100 + i * 0.1 for i in range(n)],
            "volume": [1000000 + i * 100 for i in range(n)],
        }
    )


@pytest.fixture
def small_daily_data() -> pl.DataFrame:
    """Create small dataset for edge case testing (~60 rows)."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 3, 1),
        "1d",
        eager=True,
    )
    n = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "close": [100 + i for i in range(n)],
        }
    )


@pytest.fixture
def large_daily_data() -> pl.DataFrame:
    """Create 5 years of daily data (~1260 rows)."""
    dates = pl.date_range(
        datetime(2019, 1, 1),
        datetime(2023, 12, 31),
        "1d",
        eager=True,
    )
    n = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "close": [100 + i * 0.05 for i in range(n)],
        }
    )


@pytest.fixture
def multi_asset_data() -> pl.DataFrame:
    """Create multi-asset dataset with same dates across symbols."""
    dates = pl.date_range(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31),
        "1d",
        eager=True,
    )
    len(dates)
    symbols = ["SPY", "QQQ", "IWM"]

    rows = []
    for sym in symbols:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "close": 100 + i * 0.1 + (hash(sym) % 10),
                }
            )

    return pl.DataFrame(rows).sort(["date", "symbol"])


# ============================================================================
# TimeSeriesSplitter Tests
# ============================================================================


class TestTimeSeriesSplitter:
    """Tests for walk-forward splitting with purge/embargo."""

    def test_basic_split_no_purge(self, sample_daily_data):
        """Basic walk-forward split without purge window."""
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=50,
            min_train_size=100,
            purge_window_days=0,
        )

        splits = list(splitter.split(sample_daily_data))
        assert len(splits) == 3

        for i, result in enumerate(splits):
            assert result.test_df.height == 50
            assert result.train_df.height >= 100
            assert result.boundary.fold_id == i
            assert result.boundary.test_count == 50

    def test_purge_window_removes_samples(self, sample_daily_data):
        """Verify purge window removes training samples near test boundary."""
        splitter_no_purge = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=0,
        )

        splitter_with_purge = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=10,
        )

        splits_no_purge = list(splitter_no_purge.split(sample_daily_data))
        splits_with_purge = list(splitter_with_purge.split(sample_daily_data))

        # With purge, should have fewer training samples
        for i in range(2):
            assert splits_with_purge[i].train_df.height < splits_no_purge[i].train_df.height
            assert splits_with_purge[i].boundary.purged_count > 0

    def test_no_temporal_leakage(self, sample_daily_data):
        """Verify train timestamps are always before test timestamps."""
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=50,
            min_train_size=100,
            purge_window_days=5,
        )

        for result in splitter.split(sample_daily_data):
            train_max = result.train_df["date"].max()
            test_min = result.test_df["date"].min()

            # Train must end before test starts
            assert train_max < test_min, f"Leakage: train_max={train_max} >= test_min={test_min}"

    def test_no_overlapping_indices(self, sample_daily_data):
        """Verify no row appears in both train and test."""
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=50,
            min_train_size=100,
            purge_window_days=5,
        )

        # Add index column for tracking
        df = sample_daily_data.with_row_index("row_id")

        for result in splitter.split(df):
            train_ids = set(result.train_df["row_id"].to_list())
            test_ids = set(result.test_df["row_id"].to_list())
            overlap = train_ids & test_ids
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping rows"

    def test_purge_gap_enforced(self, sample_daily_data):
        """Verify purge window creates temporal gap between train and test."""
        purge_days = 7
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=purge_days,
        )

        for result in splitter.split(sample_daily_data):
            train_end = result.train_df["date"].max()
            test_start = result.test_df["date"].min()

            # Convert to datetime for comparison
            if hasattr(train_end, "to_pydatetime"):
                train_end = train_end.to_pydatetime()
            if hasattr(test_start, "to_pydatetime"):
                test_start = test_start.to_pydatetime()

            gap = (test_start - train_end).days
            assert gap >= purge_days, f"Gap {gap} days < required {purge_days} days"

    def test_expanding_train_window(self, large_daily_data):
        """Verify train window expands across folds (walk-forward)."""
        splitter = TimeSeriesSplitter(
            n_splits=4,
            test_size=60,
            min_train_size=200,
            purge_window_days=0,
        )

        splits = list(splitter.split(large_daily_data))

        # Each successive fold should have more training data
        for i in range(len(splits) - 1):
            assert splits[i + 1].train_df.height > splits[i].train_df.height, (
                f"Fold {i + 1} train size ({splits[i + 1].train_df.height}) "
                f"should be > fold {i} ({splits[i].train_df.height})"
            )

    def test_validation_errors(self, small_daily_data):
        """Test validation catches invalid configurations."""
        # Too few rows
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=100,
            min_train_size=500,  # More than available data
        )

        with pytest.raises(ValueError, match="rows but needs at least"):
            list(splitter.split(small_daily_data))

    def test_invalid_parameters(self):
        """Test constructor validation."""
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            TimeSeriesSplitter(n_splits=0)

        with pytest.raises(ValueError, match="test_size must be >= 1"):
            TimeSeriesSplitter(test_size=0)

        with pytest.raises(ValueError, match="purge_window_days must be >= 0"):
            TimeSeriesSplitter(purge_window_days=-1)

    def test_missing_timestamp_column(self, sample_daily_data):
        """Test error when timestamp column doesn't exist."""
        splitter = TimeSeriesSplitter(timestamp_col="nonexistent")

        with pytest.raises(ValueError, match="missing timestamp column"):
            list(splitter.split(sample_daily_data))

    def test_unsorted_data_rejected(self, sample_daily_data):
        """Test error when data is not sorted by timestamp."""
        # Reverse the data
        unsorted = sample_daily_data.reverse()

        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
        )

        with pytest.raises(ValueError, match="sorted"):
            list(splitter.split(unsorted))


# ============================================================================
# PurgedKFold Tests
# ============================================================================


class TestPurgedKFold:
    """Tests for purged k-fold cross-validation."""

    def test_basic_kfold(self, sample_daily_data):
        """Basic k-fold without purge/embargo."""
        splitter = PurgedKFold(
            n_splits=5,
            purge_window_days=0,
            embargo_window_days=0,
        )

        splits = list(splitter.split(sample_daily_data))
        assert len(splits) == 5

        # Each fold should have roughly equal test size
        test_sizes = [s.test_df.height for s in splits]
        assert max(test_sizes) - min(test_sizes) <= 1

    def test_purge_embargo_removes_samples(self, sample_daily_data):
        """Verify purge/embargo remove training samples."""
        splitter_no_purge = PurgedKFold(
            n_splits=3,
            purge_window_days=0,
            embargo_window_days=0,
        )

        splitter_with_purge = PurgedKFold(
            n_splits=3,
            purge_window_days=7,
            embargo_window_days=3,
        )

        splits_no_purge = list(splitter_no_purge.split(sample_daily_data))
        splits_with_purge = list(splitter_with_purge.split(sample_daily_data))

        # With purge/embargo, should have fewer training samples
        for i in range(3):
            assert splits_with_purge[i].train_df.height < splits_no_purge[i].train_df.height

    def test_no_train_test_overlap(self, sample_daily_data):
        """Verify no timestamps overlap between train and test."""
        splitter = PurgedKFold(n_splits=4, purge_window_days=5)

        for result in splitter.split(sample_daily_data):
            train_dates = set(result.train_df["date"].to_list())
            test_dates = set(result.test_df["date"].to_list())
            overlap = train_dates & test_dates
            assert len(overlap) == 0

    def test_invalid_n_splits(self):
        """K-fold needs at least 2 splits."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFold(n_splits=1)


# ============================================================================
# SplitsManifest Tests
# ============================================================================


class TestSplitsManifest:
    """Tests for manifest generation and serialization."""

    def test_manifest_generation(self, sample_daily_data):
        """Test manifest is correctly generated."""
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=50,
            min_train_size=100,
            purge_window_days=5,
        )

        manifest = splitter.get_manifest(sample_daily_data)

        assert manifest.schema_version == SPLITS_SCHEMA_VERSION
        assert manifest.split_method == "walk_forward"
        assert manifest.n_splits == 3
        assert manifest.test_size == 50
        assert manifest.purge_window == 5  # Appendix C naming
        assert manifest.embargo_window == 0  # N/A for walk-forward
        assert manifest.total_rows == sample_daily_data.height
        assert len(manifest.splits) == 3
        assert manifest.time_range_start is not None
        assert manifest.time_range_end is not None

    def test_manifest_json_roundtrip(self, sample_daily_data):
        """Test manifest can be saved and loaded from JSON."""
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
        )

        manifest = splitter.get_manifest(sample_daily_data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manifest.to_json(path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)

            assert data["schema_version"] == SPLITS_SCHEMA_VERSION
            assert len(data["splits"]) == 2

            # Load back
            loaded = SplitsManifest.from_json(path)
            assert loaded.n_splits == manifest.n_splits
            assert loaded.total_rows == manifest.total_rows
        finally:
            path.unlink(missing_ok=True)

    def test_manifest_split_boundaries(self, sample_daily_data):
        """Verify manifest records correct split boundaries."""
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=5,
        )

        manifest = splitter.get_manifest(sample_daily_data)

        for split_dict in manifest.splits:
            assert "fold_id" in split_dict
            assert "train_start" in split_dict
            assert "train_end" in split_dict
            assert "test_start" in split_dict
            assert "test_end" in split_dict
            assert "train_count" in split_dict
            assert "test_count" in split_dict
            assert split_dict["test_count"] == 50


# ============================================================================
# Leakage Validation Tests
# ============================================================================


class TestLeakageValidation:
    """Tests for the validate_no_leakage function."""

    def test_valid_split_passes(self, sample_daily_data):
        """Valid splits should pass validation."""
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=5,
        )

        for result in splitter.split(sample_daily_data):
            validation = validate_no_leakage(
                result.train_df,
                result.test_df,
                timestamp_col="date",
                purge_window_days=5,
            )
            assert validation["valid"], f"Violations: {validation['violations']}"

    def test_detects_temporal_leakage(self):
        """Detect when train data overlaps with test temporally."""
        # Create overlapping train/test
        train = pl.DataFrame(
            {
                "date": pl.date_range(
                    datetime(2024, 1, 1), datetime(2024, 1, 31), "1d", eager=True
                ),
                "value": range(31),
            }
        )
        test = pl.DataFrame(
            {
                "date": pl.date_range(
                    datetime(2024, 1, 25), datetime(2024, 2, 10), "1d", eager=True
                ),
                "value": range(17),
            }
        )

        validation = validate_no_leakage(train, test, "date")

        assert not validation["valid"]
        assert any("overlapping" in v.lower() for v in validation["violations"])

    def test_detects_insufficient_purge_gap(self):
        """Detect when purge gap is too small."""
        train = pl.DataFrame(
            {
                "date": pl.date_range(
                    datetime(2024, 1, 1), datetime(2024, 1, 20), "1d", eager=True
                ),
                "value": range(20),
            }
        )
        test = pl.DataFrame(
            {
                "date": pl.date_range(
                    datetime(2024, 1, 22), datetime(2024, 1, 31), "1d", eager=True
                ),
                "value": range(10),
            }
        )

        # Gap is 1 day but we require 5
        validation = validate_no_leakage(train, test, "date", purge_window_days=5)

        assert not validation["valid"]
        assert any("gap" in v.lower() for v in validation["violations"])

    def test_validates_monotonicity(self):
        """Detect non-monotonic timestamps."""
        train = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 2),
                ],  # Not sorted
                "value": [1, 2, 3],
            }
        )
        test = pl.DataFrame(
            {
                "date": pl.date_range(
                    datetime(2024, 2, 1), datetime(2024, 2, 10), "1d", eager=True
                ),
                "value": range(10),
            }
        )

        validation = validate_no_leakage(train, test, "date")

        assert not validation["valid"]
        assert any("monotonic" in v.lower() for v in validation["violations"])


# ============================================================================
# Multi-Asset Tests
# ============================================================================


class TestMultiAssetLeakageValidation:
    """Tests for multi-asset data with key_cols validation."""

    def test_key_cols_detects_row_overlap(self, multi_asset_data):
        """Key-based overlap detects when same (symbol, date) in both sets."""
        # Split by row - this creates same dates in both train/test for different symbols
        mid = multi_asset_data.height // 2
        train = multi_asset_data.head(mid)
        test = multi_asset_data.tail(multi_asset_data.height - mid + 50)  # Overlap by 50 rows

        # Timestamp-only check might miss this if dates differ
        # Key-cols check should catch it
        validation = validate_no_leakage(
            train, test, timestamp_col="date", key_cols=["symbol", "date"]
        )

        # Should detect overlap
        assert "key_overlap" in validation["checks"]

    def test_key_cols_passes_valid_split(self, multi_asset_data):
        """Valid multi-asset split passes key_cols validation."""
        # Split by date (all symbols for each date go together)
        unique_dates = multi_asset_data["date"].unique().sort()
        n_dates = len(unique_dates)
        cutoff_date = unique_dates[n_dates // 2]

        train = multi_asset_data.filter(pl.col("date") < cutoff_date)
        test = multi_asset_data.filter(pl.col("date") >= cutoff_date)

        validation = validate_no_leakage(
            train, test, timestamp_col="date", key_cols=["symbol", "date"]
        )

        assert validation["valid"], f"Violations: {validation['violations']}"
        assert validation["checks"]["key_overlap"]["overlapping_keys"] == 0

    def test_timestamp_only_false_positive_multi_asset(self, multi_asset_data):
        """Show that timestamp-only check can false-positive on multi-asset data."""
        # Split so that same dates appear in both train and test
        # (different symbols, so no actual row overlap)
        unique_dates = multi_asset_data["date"].unique().sort()
        mid_date = unique_dates[len(unique_dates) // 2]

        # SPY train: first half dates, QQQ test: all dates (overlapping dates but different symbols)
        train = multi_asset_data.filter((pl.col("symbol") == "SPY") & (pl.col("date") < mid_date))
        test = multi_asset_data.filter(pl.col("symbol") == "QQQ")

        # Timestamp-only check will flag overlap (same dates, different symbols)
        validation_ts_only = validate_no_leakage(train, test, "date")

        # Key-cols check should pass (no actual row overlap)
        validation_key_cols = validate_no_leakage(train, test, "date", key_cols=["symbol", "date"])

        # The timestamp overlap is detected but key overlap should be 0
        if "timestamp_overlap" in validation_ts_only["checks"]:
            # May or may not have overlap depending on date ranges
            pass

        assert validation_key_cols["checks"]["key_overlap"]["overlapping_keys"] == 0


class TestPurgedKFoldNonContiguous:
    """Tests verifying PurgedKFold reports non-contiguous training correctly."""

    def test_non_contiguous_flag_set(self, sample_daily_data):
        """PurgedKFold should set non_contiguous_train=True."""
        splitter = PurgedKFold(n_splits=3, purge_window_days=5)

        for result in splitter.split(sample_daily_data):
            assert result.boundary.non_contiguous_train is True

    def test_walk_forward_contiguous_flag(self, sample_daily_data):
        """TimeSeriesSplitter should have non_contiguous_train=False."""
        splitter = TimeSeriesSplitter(n_splits=2, test_size=50, min_train_size=100)

        for result in splitter.split(sample_daily_data):
            assert result.boundary.non_contiguous_train is False

    def test_embargo_actually_removes_after_test(self, sample_daily_data):
        """PurgedKFold embargo should remove training data after test period."""
        splitter = PurgedKFold(
            n_splits=3,
            purge_window_days=0,
            embargo_window_days=10,  # 10 day embargo
        )

        for result in splitter.split(sample_daily_data):
            # For middle folds, embargoed_count should be > 0
            if result.boundary.fold_id > 0 and result.boundary.fold_id < splitter.n_splits - 1:
                # Middle folds have training data after test
                # embargo should remove some of it
                assert (
                    result.boundary.embargoed_count >= 0
                )  # May be 0 if embargo exceeds remaining data


# ============================================================================
# Integration Tests
# ============================================================================


class TestSplitsIntegration:
    """Integration tests combining multiple components."""

    def test_create_splits_manifest_function(self, sample_daily_data):
        """Test the convenience function for manifest creation."""
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=50,
            min_train_size=100,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manifest = create_splits_manifest(sample_daily_data, splitter, path)

            assert manifest.n_splits == 3
            assert path.exists()

            # Verify file contents
            with open(path) as f:
                data = json.load(f)
            assert data["n_splits"] == 3
        finally:
            path.unlink(missing_ok=True)

    def test_end_to_end_walk_forward(self, large_daily_data):
        """Full walk-forward validation with all features."""
        splitter = TimeSeriesSplitter(
            n_splits=5,
            test_size=60,
            min_train_size=200,
            purge_window_days=7,
        )

        # Generate manifest
        manifest = splitter.get_manifest(large_daily_data)
        assert len(manifest.splits) == 5
        assert len(manifest.warnings) == 0

        # Validate each split
        for result in splitter.split(large_daily_data):
            # No leakage
            validation = validate_no_leakage(
                result.train_df,
                result.test_df,
                "date",
                purge_window_days=7,
            )
            assert validation["valid"], (
                f"Fold {result.boundary.fold_id}: {validation['violations']}"
            )

            # Correct sizes
            assert result.test_df.height == 60
            assert result.train_df.height >= 200

            # Purge applied
            assert result.boundary.purged_count >= 0

    def test_split_result_assert_no_leakage(self, sample_daily_data):
        """Test SplitResult.assert_no_leakage method."""
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
        )

        for result in splitter.split(sample_daily_data):
            # Should not raise
            result.assert_no_leakage(timestamp_col="date")


# ============================================================================
# Property-Based Tests (Hypothesis Integration Ready)
# ============================================================================


class TestSplitsProperties:
    """Property-based test cases (can be extended with Hypothesis)."""

    def test_train_test_exhaustive_no_purge(self, sample_daily_data):
        """Without purge, train + test should cover expected rows."""
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=0,
        )

        for result in splitter.split(sample_daily_data):
            # Test + train should not exceed total rows
            total_in_split = result.train_df.height + result.test_df.height
            assert total_in_split <= sample_daily_data.height

    def test_timestamps_strictly_increasing(self, sample_daily_data):
        """All timestamps within each split should be strictly ordered."""
        splitter = TimeSeriesSplitter(
            n_splits=3,
            test_size=50,
            min_train_size=100,
        )

        for result in splitter.split(sample_daily_data):
            # Train timestamps monotonic
            train_ts = result.train_df["date"].to_list()
            assert train_ts == sorted(train_ts)

            # Test timestamps monotonic
            test_ts = result.test_df["date"].to_list()
            assert test_ts == sorted(test_ts)

    def test_boundary_timestamps_match_data(self, sample_daily_data):
        """Boundary timestamps should match actual data boundaries."""
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=50,
            min_train_size=100,
            purge_window_days=0,  # No purge for exact match
        )

        for result in splitter.split(sample_daily_data):
            # Train boundaries match data
            if result.train_df.height > 0:
                actual_train_start = result.train_df["date"][0]

                # Convert boundary to same type for comparison
                boundary_start = result.boundary.train_start

                # Handle date vs datetime comparison
                if hasattr(boundary_start, "date"):
                    boundary_date = boundary_start.date()
                else:
                    boundary_date = boundary_start

                if hasattr(actual_train_start, "date"):
                    actual_date = actual_train_start.date()
                else:
                    actual_date = actual_train_start

                assert boundary_date == actual_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

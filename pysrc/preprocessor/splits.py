"""Time-series aware train/test splitting with purge and embargo windows.

This module implements Marcos López de Prado's purge and embargo methodology
for preventing information leakage in financial time-series cross-validation.

Purge Window: Gap between train and test sets to remove overlapping observations.
    - Removes training samples whose labels overlap with test period
    - Prevents look-ahead bias from label windows spanning train/test boundary
    - Applies to BOTH walk-forward and k-fold splitters

Embargo Window: Gap after test set where no training data is used.
    - Prevents information from test period leaking into future training folds
    - Only applies to PurgedKFold where training data exists on BOTH sides of test
    - NOT applicable to walk-forward (TimeSeriesSplitter) since train always precedes test

Reference: López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# Schema version for splits_manifest.json
SPLITS_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class SplitBoundary:
    """Defines a single train/test split boundary with timestamps.

    For PurgedKFold, training data may be non-contiguous (exists on both
    sides of the test period). The non_contiguous_train flag indicates this,
    and train_start/train_end represent the overall span (not a contiguous range).
    """

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_count: int
    test_count: int
    purged_count: int = 0
    embargoed_count: int = 0
    non_contiguous_train: bool = False  # True for k-fold where train spans both sides of test

    def to_dict(self) -> dict:
        """Convert to dictionary with ISO timestamps."""
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
            "train_count": self.train_count,
            "test_count": self.test_count,
            "purged_count": self.purged_count,
            "embargoed_count": self.embargoed_count,
            "non_contiguous_train": self.non_contiguous_train,
        }


@dataclass
class SplitResult:
    """Result of applying a split to data."""

    train_df: pl.DataFrame
    test_df: pl.DataFrame
    boundary: SplitBoundary
    # Store indices explicitly so they survive column drops
    _train_indices: list[int] = field(default_factory=list)
    _test_indices: list[int] = field(default_factory=list)

    @property
    def train_indices(self) -> list[int]:
        """Row indices of training data in original DataFrame."""
        return self._train_indices

    @property
    def test_indices(self) -> list[int]:
        """Row indices of test data in original DataFrame."""
        return self._test_indices

    def assert_no_leakage(self, timestamp_col: str = "date", key_cols: list[str] = None) -> None:
        """Verify train and test sets don't overlap.

        Args:
            timestamp_col: Name of timestamp column
            key_cols: Optional list of columns forming unique row key (e.g., ["symbol", "date"])
                     If None, uses timestamp_col only (assumes single-asset data)
        """
        # Check by index if available
        if self._train_indices and self._test_indices:
            train_idx = set(self._train_indices)
            test_idx = set(self._test_indices)
            overlap = train_idx & test_idx
            if overlap:
                raise ValueError(f"Leakage detected: {len(overlap)} overlapping row indices")

        # Check by key columns
        if key_cols:
            cols = key_cols
        elif timestamp_col in self.train_df.columns and timestamp_col in self.test_df.columns:
            cols = [timestamp_col]
        else:
            return  # Cannot verify without key columns

        # Build composite keys
        train_keys = set(zip(*[self.train_df[c].to_list() for c in cols], strict=False))
        test_keys = set(zip(*[self.test_df[c].to_list() for c in cols], strict=False))
        overlap = train_keys & test_keys
        if overlap:
            raise ValueError(f"Leakage detected: {len(overlap)} overlapping keys")


@dataclass
class SplitsManifest:
    """Manifest documenting all splits for a run bundle.

    Field names match Appendix C specification:
    - purge_window (not purge_window_days)
    - embargo_window (not embargo_window_days)
    """

    schema_version: str = SPLITS_SCHEMA_VERSION
    split_method: str = "walk_forward"
    timestamp_column: str = "date"
    n_splits: int = 0
    purge_window: int = 0  # Appendix C naming
    embargo_window: int = 0  # Appendix C naming (0 for walk-forward)
    min_train_size: int = 0
    test_size: int = 0
    total_rows: int = 0
    time_range_start: str | None = None
    time_range_end: str | None = None
    splits: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Write manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> SplitsManifest:
        """Load manifest from JSON file.

        Raises:
            ValueError: If schema_version has unknown major version
        """
        with open(path) as f:
            data = json.load(f)

        # Validate schema version
        file_version = data.get("schema_version", "0.0.0")
        file_major = int(file_version.split(".")[0])
        supported_major = int(SPLITS_SCHEMA_VERSION.split(".")[0])

        if file_major > supported_major:
            raise ValueError(
                f"UNKNOWN_SCHEMA_VERSION: manifest version {file_version} "
                f"not supported (max supported major: {supported_major})"
            )

        # Handle legacy field names (backwards compatibility)
        if "purge_window_days" in data and "purge_window" not in data:
            data["purge_window"] = data.pop("purge_window_days")
        if "embargo_window_days" in data and "embargo_window" not in data:
            data["embargo_window"] = data.pop("embargo_window_days")

        # Filter to known fields only
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)


def _to_datetime(val) -> datetime:
    """Convert polars date/datetime to Python datetime."""
    if isinstance(val, datetime):
        return val
    if hasattr(val, "to_pydatetime"):
        return val.to_pydatetime()
    # Handle polars Date type
    if hasattr(val, "year"):
        return datetime(val.year, val.month, val.day)
    raise TypeError(f"Cannot convert {type(val)} to datetime")


class TimeSeriesSplitter:
    """Walk-forward splitter with purge window.

    This implements expanding-window walk-forward validation where:
    - Training set grows over time (anchored at start)
    - Test set is fixed size
    - Purge window removes training samples too close to test

    Note: Embargo is NOT applicable to walk-forward validation because
    training data always precedes test data. Embargo only applies to
    splitters like PurgedKFold where training data exists on both sides
    of the test period.

    Example with 5-day purge:

        |------- TRAIN -------|[PURGE]|-- TEST --|
        t0                    t1      t2         t3

    Where:
    - t1 = test_start - purge_window (train ends here)
    - t2 = test_start
    - t3 = test_end
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,  # ~1 year of trading days
        min_train_size: int = 504,  # ~2 years minimum
        purge_window_days: int = 5,
        timestamp_col: str = "date",
    ):
        """Initialize splitter.

        Args:
            n_splits: Number of walk-forward folds
            test_size: Number of rows in each test set
            min_train_size: Minimum rows required for training
            purge_window_days: Days to remove before test period
            timestamp_col: Name of timestamp column in DataFrame
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
        if purge_window_days < 0:
            raise ValueError(f"purge_window_days must be >= 0, got {purge_window_days}")

        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.purge_window_days = purge_window_days
        self.timestamp_col = timestamp_col

    def _validate_dataframe(self, df: pl.DataFrame) -> None:
        """Validate DataFrame has required columns and structure."""
        if self.timestamp_col not in df.columns:
            raise ValueError(f"DataFrame missing timestamp column: {self.timestamp_col}")

        # Check timestamp column is datetime type
        col_dtype = df[self.timestamp_col].dtype
        if col_dtype not in (pl.Date, pl.Datetime):
            raise TypeError(
                f"Column '{self.timestamp_col}' must be Date or Datetime, got {col_dtype}"
            )

        # Check data is sorted by timestamp
        timestamps = df[self.timestamp_col]
        if not timestamps.equals(timestamps.sort()):
            raise ValueError(
                f"DataFrame must be sorted by '{self.timestamp_col}' in ascending order"
            )

    def _get_required_rows(self) -> int:
        """Calculate minimum rows needed for valid splitting."""
        return self.min_train_size + (self.n_splits * self.test_size)

    def split(self, df: pl.DataFrame) -> Iterator[SplitResult]:
        """Generate train/test splits with purge window.

        Args:
            df: DataFrame with timestamp column, sorted ascending

        Yields:
            SplitResult for each fold

        Raises:
            ValueError: If DataFrame is too small or improperly formatted
        """
        self._validate_dataframe(df)

        n_rows = df.height
        required = self._get_required_rows()

        if n_rows < required:
            raise ValueError(
                f"DataFrame has {n_rows} rows but needs at least {required} "
                f"(min_train_size={self.min_train_size} + "
                f"n_splits={self.n_splits} × test_size={self.test_size})"
            )

        # Add original index for tracking
        df_indexed = df.with_row_index("_original_index")

        for fold_id in range(self.n_splits):
            # Work backwards from end: fold 0 is earliest, fold n-1 is latest
            folds_remaining = self.n_splits - 1 - fold_id
            test_end_idx = n_rows - (folds_remaining * self.test_size)
            test_start_idx = test_end_idx - self.test_size

            # Training ends before test starts
            train_end_idx = test_start_idx
            train_start_idx = 0

            # Extract timestamps for purge calculation
            test_start_ts = _to_datetime(df_indexed[test_start_idx, self.timestamp_col])
            test_end_ts = _to_datetime(df_indexed[test_end_idx - 1, self.timestamp_col])

            # Apply purge window: remove training samples within purge_window_days of test_start
            purge_cutoff = test_start_ts - timedelta(days=self.purge_window_days)

            # Build train mask: rows before train_end AND before purge cutoff
            train_mask = (pl.col("_original_index") >= train_start_idx) & (
                pl.col("_original_index") < train_end_idx
            )

            # Apply purge if window > 0
            if self.purge_window_days > 0:
                train_mask = train_mask & (pl.col(self.timestamp_col) <= purge_cutoff)

            train_df = df_indexed.filter(train_mask)

            # Test set: simple slice
            test_df = df_indexed.slice(test_start_idx, self.test_size)

            # Capture indices before dropping the column
            train_indices = train_df["_original_index"].to_list()
            test_indices = test_df["_original_index"].to_list()

            # Calculate counts
            original_train_count = train_end_idx - train_start_idx
            actual_train_count = train_df.height
            purged_count = original_train_count - actual_train_count

            # Get actual timestamps from filtered data
            if train_df.height > 0:
                train_start_ts = _to_datetime(train_df[0, self.timestamp_col])
                train_end_ts = _to_datetime(train_df[-1, self.timestamp_col])
            else:
                train_start_ts = _to_datetime(df_indexed[train_start_idx, self.timestamp_col])
                train_end_ts = purge_cutoff

            boundary = SplitBoundary(
                fold_id=fold_id,
                train_start=train_start_ts,
                train_end=train_end_ts,
                test_start=test_start_ts,
                test_end=test_end_ts,
                train_count=train_df.height,
                test_count=test_df.height,
                purged_count=purged_count,
                embargoed_count=0,  # N/A for walk-forward
            )

            yield SplitResult(
                train_df=train_df.drop("_original_index"),
                test_df=test_df.drop("_original_index"),
                boundary=boundary,
                _train_indices=train_indices,
                _test_indices=test_indices,
            )

    def get_manifest(self, df: pl.DataFrame) -> SplitsManifest:
        """Generate manifest documenting all splits.

        Args:
            df: DataFrame to be split

        Returns:
            SplitsManifest with all split boundaries and metadata
        """
        self._validate_dataframe(df)

        manifest = SplitsManifest(
            split_method="walk_forward",
            timestamp_column=self.timestamp_col,
            n_splits=self.n_splits,
            purge_window=self.purge_window_days,
            embargo_window=0,  # N/A for walk-forward
            min_train_size=self.min_train_size,
            test_size=self.test_size,
            total_rows=df.height,
        )

        # Get time range
        timestamps = df[self.timestamp_col]
        if len(timestamps) > 0:
            manifest.time_range_start = _to_datetime(timestamps[0]).isoformat()
            manifest.time_range_end = _to_datetime(timestamps[-1]).isoformat()

        # Generate splits and record boundaries
        try:
            for result in self.split(df):
                manifest.splits.append(result.boundary.to_dict())
        except ValueError as e:
            manifest.warnings.append(str(e))

        return manifest


class PurgedKFold:
    """K-Fold cross-validation with purge and embargo for time series.

    Unlike standard k-fold which randomly assigns rows to folds, this
    implementation:
    1. Maintains temporal order within each fold
    2. Applies purge windows to remove training samples near test periods
    3. Applies embargo windows to prevent information from test leaking forward

    This is suitable for time-series with overlapping labels where standard
    train/test splits would leak information.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window_days: int = 5,
        embargo_window_days: int = 3,
        timestamp_col: str = "date",
    ):
        """Initialize purged k-fold splitter.

        Args:
            n_splits: Number of folds
            purge_window_days: Days to remove around test boundaries
            embargo_window_days: Days to embargo after test period
            timestamp_col: Name of timestamp column
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2 for k-fold, got {n_splits}")

        self.n_splits = n_splits
        self.purge_window_days = purge_window_days
        self.embargo_window_days = embargo_window_days
        self.timestamp_col = timestamp_col

    def split(self, df: pl.DataFrame) -> Iterator[SplitResult]:
        """Generate purged k-fold splits.

        Args:
            df: DataFrame with timestamp column, sorted ascending

        Yields:
            SplitResult for each fold
        """
        if self.timestamp_col not in df.columns:
            raise ValueError(f"DataFrame missing timestamp column: {self.timestamp_col}")

        n_rows = df.height
        fold_size = n_rows // self.n_splits

        if fold_size < 1:
            raise ValueError(
                f"DataFrame has {n_rows} rows, need at least {self.n_splits} "
                f"for {self.n_splits}-fold split"
            )

        df_indexed = df.with_row_index("_original_index")

        for fold_id in range(self.n_splits):
            # Test fold boundaries
            test_start_idx = fold_id * fold_size
            test_end_idx = (fold_id + 1) * fold_size if fold_id < self.n_splits - 1 else n_rows

            # Get test period timestamps
            test_start_ts = _to_datetime(df_indexed[test_start_idx, self.timestamp_col])
            test_end_ts = _to_datetime(df_indexed[test_end_idx - 1, self.timestamp_col])

            # Calculate purge/embargo boundaries
            purge_start = test_start_ts - timedelta(days=self.purge_window_days)
            embargo_end = test_end_ts + timedelta(days=self.embargo_window_days)

            # Base training mask: NOT in test fold
            base_train_mask = (pl.col("_original_index") < test_start_idx) | (
                pl.col("_original_index") >= test_end_idx
            )
            base_train_df = df_indexed.filter(base_train_mask)
            original_train_count = base_train_df.height

            # Apply purge: remove rows in purge window (before test)
            purge_mask = pl.col(self.timestamp_col) < purge_start
            after_purge_df = base_train_df.filter(
                purge_mask | (pl.col(self.timestamp_col) > test_end_ts)
            )
            purged_count = original_train_count - after_purge_df.height

            # Apply embargo: remove rows in embargo window (after test)
            embargo_mask = pl.col(self.timestamp_col) > embargo_end
            train_df = after_purge_df.filter(
                (pl.col(self.timestamp_col) < purge_start) | embargo_mask
            )
            embargoed_count = after_purge_df.height - train_df.height

            test_df = df_indexed.slice(test_start_idx, test_end_idx - test_start_idx)

            # Capture indices before dropping
            train_indices = train_df["_original_index"].to_list()
            test_indices = test_df["_original_index"].to_list()

            # Get train timestamps
            if train_df.height > 0:
                train_start_ts = _to_datetime(train_df[0, self.timestamp_col])
                train_end_ts = _to_datetime(train_df[-1, self.timestamp_col])
            else:
                train_start_ts = test_start_ts
                train_end_ts = test_end_ts

            boundary = SplitBoundary(
                fold_id=fold_id,
                train_start=train_start_ts,
                train_end=train_end_ts,
                test_start=test_start_ts,
                test_end=test_end_ts,
                train_count=train_df.height,
                test_count=test_df.height,
                purged_count=purged_count,
                embargoed_count=embargoed_count,
                non_contiguous_train=True,  # K-fold train spans both sides of test
            )

            yield SplitResult(
                train_df=train_df.drop("_original_index"),
                test_df=test_df.drop("_original_index"),
                boundary=boundary,
                _train_indices=train_indices,
                _test_indices=test_indices,
            )


def validate_no_leakage(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    timestamp_col: str = "date",
    purge_window_days: int = 0,
    embargo_window_days: int = 0,
    key_cols: list[str] = None,
) -> dict:
    """Validate that train and test sets have no temporal leakage.

    Checks:
    1. No overlapping rows by key columns (supports multi-asset data)
    2. Train timestamps < test timestamps (with purge gap)
    3. Test timestamps < any post-test training (with embargo gap)
    4. Timestamps monotonic within each set

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        timestamp_col: Name of timestamp column
        purge_window_days: Expected gap between train end and test start
        embargo_window_days: Expected gap between test end and any training after
        key_cols: Columns forming unique row key (e.g., ["symbol", "date"])
                  If None, uses timestamp_col only (single-asset assumption)

    Returns:
        dict with validation results and any violations found
    """
    result = {
        "valid": True,
        "checks": {},
        "violations": [],
    }

    # Check 1: Timestamp columns exist
    if timestamp_col not in train_df.columns or timestamp_col not in test_df.columns:
        result["valid"] = False
        result["violations"].append(f"Missing timestamp column: {timestamp_col}")
        return result

    # Check 2: No overlapping rows by key
    if key_cols:
        # Multi-asset: check composite key overlap
        if not all(c in train_df.columns and c in test_df.columns for c in key_cols):
            result["warnings"] = [f"Some key_cols not in both DataFrames: {key_cols}"]
        else:
            train_keys = set(zip(*[train_df[c].to_list() for c in key_cols], strict=False))
            test_keys = set(zip(*[test_df[c].to_list() for c in key_cols], strict=False))
            overlap_keys = train_keys & test_keys

            result["checks"]["key_overlap"] = {
                "key_columns": key_cols,
                "train_unique_keys": len(train_keys),
                "test_unique_keys": len(test_keys),
                "overlapping_keys": len(overlap_keys),
            }

            if overlap_keys:
                result["valid"] = False
                result["violations"].append(
                    f"Found {len(overlap_keys)} overlapping keys across train/test"
                )
    else:
        # Single-asset: check timestamp overlap (may false-positive on multi-asset)
        train_ts = set(train_df[timestamp_col].to_list())
        test_ts = set(test_df[timestamp_col].to_list())
        overlap_ts = train_ts & test_ts

        result["checks"]["timestamp_overlap"] = {
            "train_unique_timestamps": len(train_ts),
            "test_unique_timestamps": len(test_ts),
            "overlapping_timestamps": len(overlap_ts),
        }

        if overlap_ts:
            result["valid"] = False
            result["violations"].append(f"Found {len(overlap_ts)} overlapping timestamps")

    # Check 3: Train ends before test starts (purge gap)
    if train_df.height > 0 and test_df.height > 0:
        train_end = _to_datetime(train_df[timestamp_col].max())
        test_start = _to_datetime(test_df[timestamp_col].min())
        test_end = _to_datetime(test_df[timestamp_col].max())

        # Use total_seconds for sub-day precision
        actual_gap = test_start - train_end
        actual_gap_seconds = actual_gap.total_seconds()
        expected_gap_seconds = purge_window_days * 86400  # days to seconds

        result["checks"]["purge_gap"] = {
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "expected_gap_days": purge_window_days,
            "actual_gap_seconds": actual_gap_seconds,
            "actual_gap_days": actual_gap_seconds / 86400,
        }

        if train_end >= test_start:
            result["valid"] = False
            result["violations"].append(f"Train end ({train_end}) >= test start ({test_start})")
        elif actual_gap_seconds < expected_gap_seconds:
            result["valid"] = False
            result["violations"].append(
                f"Purge gap ({actual_gap_seconds / 86400:.2f} days) < required ({purge_window_days} days)"
            )

        # Check 4: Embargo validation (for k-fold where train exists after test)
        # Find any training data that comes after test_end
        train_after_test = train_df.filter(pl.col(timestamp_col) > test_end)
        if train_after_test.height > 0 and embargo_window_days > 0:
            first_train_after = _to_datetime(train_after_test[timestamp_col].min())
            embargo_gap = first_train_after - test_end
            embargo_gap_seconds = embargo_gap.total_seconds()
            expected_embargo_seconds = embargo_window_days * 86400

            result["checks"]["embargo_gap"] = {
                "test_end": test_end.isoformat(),
                "first_train_after_test": first_train_after.isoformat(),
                "expected_embargo_days": embargo_window_days,
                "actual_embargo_seconds": embargo_gap_seconds,
                "actual_embargo_days": embargo_gap_seconds / 86400,
            }

            if embargo_gap_seconds < expected_embargo_seconds:
                result["valid"] = False
                result["violations"].append(
                    f"Embargo gap ({embargo_gap_seconds / 86400:.2f} days) < required ({embargo_window_days} days)"
                )

    # Check 5: Monotonic timestamps within each set
    train_sorted = train_df[timestamp_col].equals(train_df[timestamp_col].sort())
    test_sorted = test_df[timestamp_col].equals(test_df[timestamp_col].sort())

    result["checks"]["monotonicity"] = {
        "train_monotonic": train_sorted,
        "test_monotonic": test_sorted,
    }

    if not train_sorted:
        result["valid"] = False
        result["violations"].append("Train timestamps not monotonic")
    if not test_sorted:
        result["valid"] = False
        result["violations"].append("Test timestamps not monotonic")

    return result


def create_splits_manifest(
    df: pl.DataFrame,
    splitter: TimeSeriesSplitter | PurgedKFold,
    output_path: Path | None = None,
) -> SplitsManifest:
    """Create and optionally save a splits manifest.

    Args:
        df: DataFrame to split
        splitter: Configured splitter instance
        output_path: Optional path to save manifest JSON

    Returns:
        SplitsManifest documenting all splits
    """
    if isinstance(splitter, TimeSeriesSplitter):
        manifest = splitter.get_manifest(df)
    else:
        # PurgedKFold - build manifest manually
        manifest = SplitsManifest(
            split_method="purged_kfold",
            timestamp_column=splitter.timestamp_col,
            n_splits=splitter.n_splits,
            purge_window=splitter.purge_window_days,
            embargo_window=splitter.embargo_window_days,
            total_rows=df.height,
        )

        # Set time range for PurgedKFold too
        if splitter.timestamp_col in df.columns and df.height > 0:
            manifest.time_range_start = _to_datetime(df[splitter.timestamp_col][0]).isoformat()
            manifest.time_range_end = _to_datetime(df[splitter.timestamp_col][-1]).isoformat()

        try:
            for result in splitter.split(df):
                manifest.splits.append(result.boundary.to_dict())
        except ValueError as e:
            manifest.warnings.append(str(e))

    if output_path:
        manifest.to_json(output_path)

    return manifest


__all__ = [
    "SPLITS_SCHEMA_VERSION",
    "SplitBoundary",
    "SplitResult",
    "SplitsManifest",
    "TimeSeriesSplitter",
    "PurgedKFold",
    "validate_no_leakage",
    "create_splits_manifest",
]

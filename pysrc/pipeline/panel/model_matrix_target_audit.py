"""Phase 1 target root-cause audits for panel train-matrix."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pysrc.pipeline.panel.train_model_matrix import FoldBoundary, TrainRowPolicy

_KEY_COLUMNS: tuple[str, ...] = ("date", "instrument", "interval")
_SUPERVISION_CONTEXT_COLUMNS: tuple[str, ...] = (
    "forward_return_horizon",
    "adjusted_return_1d",
    "raw_return_1d",
)
_ABS_GT_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 1.0, 10.0, 100.0)
_EXTREME_ROW_LIMIT = 100
_MAX_FEATURE_COLUMNS = 5
_BATCH_SIZE = 250_000
_TAIL_QUANTILE_LEVELS: tuple[tuple[str, float], ...] = (
    ("p0.001", 0.001),
    ("p0.01", 0.01),
    ("p0.1", 0.1),
    ("p1", 0.01),  # 1st percentile
    ("p50", 0.50),
    ("p99", 0.99),
    ("p99.9", 0.999),
    ("p99.99", 0.9999),
)
_TAIL_QUANTILE_KEYS: tuple[str, ...] = tuple(label for label, _ in _TAIL_QUANTILE_LEVELS)


def _extended_quantile_summary(values: np.ndarray) -> dict[str, float | None]:
    """Tail-aware quantile summary for train-target audits."""

    clean = values[np.isfinite(values)]
    if clean.size == 0:
        payload: dict[str, float | None] = {"min": None, "max": None, "count": 0}
        for label, _ in _TAIL_QUANTILE_LEVELS:
            payload[label] = None
        return payload
    payload = {
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "count": int(clean.size),
    }
    for label, level in _TAIL_QUANTILE_LEVELS:
        payload[label] = float(np.quantile(clean, level))
    return payload


def _count_abs_gt(values: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, int]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {f"abs_gt_{threshold:g}": 0 for threshold in thresholds}
    abs_values = np.abs(clean)
    return {
        f"abs_gt_{threshold:g}": int(np.sum(abs_values > threshold)) for threshold in thresholds
    }


def _discover_feature_columns(
    columns: list[str],
    *,
    target_column: str,
) -> tuple[str, ...]:
    blocked = frozenset(
        {
            *_KEY_COLUMNS,
            *_SUPERVISION_CONTEXT_COLUMNS,
            target_column,
            "adj_close",
            "raw_close",
            "volume",
        }
    )
    features: list[str] = []
    for column in columns:
        if column in blocked:
            continue
        features.append(column)
        if len(features) >= _MAX_FEATURE_COLUMNS:
            break
    return tuple(features)


def _iter_panel_batches(path: Path, *, columns: list[str], batch_size: int = _BATCH_SIZE):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
        yield batch.to_pandas()


def _collect_unique_dates(panel_path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    from pysrc.pipeline.panel.train_model_matrix import normalize_date_labels

    parquet = pq.ParquetFile(panel_path)
    if "date" not in parquet.schema_arrow.names:
        raise ValueError(f"Panel missing date column: {panel_path}")
    unique_dates: set[str] = set()
    for batch in parquet.iter_batches(columns=["date"], batch_size=_BATCH_SIZE):
        pdf = batch.to_pandas()
        unique_dates.update(normalize_date_labels(pdf["date"]).tolist())
    return np.asarray(sorted(unique_dates), dtype=object)


@dataclass
class _ExtremeRowTracker:
    capacity: int = _EXTREME_ROW_LIMIT
    _heap: list[tuple[float, int, dict[str, object]]] = field(default_factory=list)
    _counter: int = 0

    def consider(self, *, abs_value: float, row: dict[str, object]) -> None:
        self._counter += 1
        entry = (float(abs_value), self._counter, row)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            return
        if abs_value > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def rows(self) -> list[dict[str, object]]:
        return [item[2] for item in sorted(self._heap, key=lambda item: (-item[0], item[1]))]


@dataclass
class _FoldTrainAccumulator:
    targets: list[float] = field(default_factory=list)
    extreme_rows: _ExtremeRowTracker = field(default_factory=_ExtremeRowTracker)

    def add_targets_batch(
        self,
        *,
        targets: np.ndarray,
        frame: pd.DataFrame,
        dates: pd.Series,
        target_column: str,
        context_columns: tuple[str, ...],
        feature_columns: tuple[str, ...],
    ) -> None:
        finite = targets[np.isfinite(targets)]
        if finite.size == 0:
            return
        self.targets.extend(finite.tolist())
        batch_max_abs = float(np.max(np.abs(finite)))
        heap_min = (
            self.extreme_rows._heap[0][0]
            if len(self.extreme_rows._heap) >= self.extreme_rows.capacity
            else 0.0
        )
        if batch_max_abs <= heap_min:
            return
        for row_idx, target_value in enumerate(targets):
            if not np.isfinite(target_value):
                continue
            abs_value = abs(float(target_value))
            if (
                len(self.extreme_rows._heap) >= self.extreme_rows.capacity
                and abs_value <= self.extreme_rows._heap[0][0]
            ):
                continue
            row_payload: dict[str, object] = {
                "date": str(dates.iloc[row_idx]),
                "instrument": str(frame.iloc[row_idx]["instrument"]),
                "interval": str(frame.iloc[row_idx]["interval"]),
                target_column: float(target_value),
            }
            for column in context_columns:
                value = frame.iloc[row_idx][column]
                row_payload[column] = float(value) if pd.notna(value) else None
            for column in feature_columns:
                value = frame.iloc[row_idx][column]
                row_payload[column] = float(value) if pd.notna(value) else None
            self.extreme_rows.consider(abs_value=abs_value, row=row_payload)

    def finalize(self, thresholds: tuple[float, ...]) -> dict[str, object]:
        values = np.asarray(self.targets, dtype=np.float64)
        summary = _extended_quantile_summary(values)
        payload: dict[str, object] = {
            **summary,
            "count_abs_gt": _count_abs_gt(values, thresholds),
            "extreme_rows": self.extreme_rows.rows(),
        }
        return payload


def _train_mask_for_boundary(date_codes: np.ndarray, boundary: FoldBoundary) -> np.ndarray:
    from pysrc.pipeline.panel.train_model_matrix import fold_masks_from_boundaries

    train_mask, _ = fold_masks_from_boundaries(date_codes, boundary)
    return train_mask


def build_train_target_fold_audit(
    panel_path: Path,
    target_column: str,
    boundaries: list[FoldBoundary],
    train_row_policy: TrainRowPolicy,
    random_seed: int,
) -> dict[str, object]:
    """Stream panel parquet and summarize train-fold target distributions."""

    _ = random_seed
    panel_path = Path(panel_path)
    unique_dates = _collect_unique_dates(panel_path)
    date_map = {str(value): idx for idx, value in enumerate(unique_dates.tolist())}

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(panel_path)
    schema_columns = list(parquet.schema_arrow.names)
    if target_column not in schema_columns:
        raise ValueError(f"Panel missing target column {target_column!r}: {panel_path}")

    context_columns = [
        column
        for column in _SUPERVISION_CONTEXT_COLUMNS
        if column in schema_columns and column != target_column
    ]
    feature_columns = _discover_feature_columns(schema_columns, target_column=target_column)
    scan_columns = [
        *list(_KEY_COLUMNS),
        target_column,
        *context_columns,
        *feature_columns,
    ]

    accumulators = {boundary.fold_id: _FoldTrainAccumulator() for boundary in boundaries}
    rows_scanned = 0

    from pysrc.pipeline.panel.panel_keys import normalize_key_columns

    for pdf in _iter_panel_batches(panel_path, columns=scan_columns):
        rows_scanned += len(pdf)
        pdf = normalize_key_columns(pdf)
        dates = pdf["date"]
        date_codes = dates.map(date_map).to_numpy(dtype=np.intp)
        targets = pd.to_numeric(pdf[target_column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(targets)

        for boundary in boundaries:
            train_mask = _train_mask_for_boundary(date_codes, boundary) & finite
            if not train_mask.any():
                continue
            fold_frame = pdf.loc[train_mask].reset_index(drop=True)
            fold_targets = targets[train_mask]
            fold_dates = dates.loc[train_mask].reset_index(drop=True)
            accumulator = accumulators[boundary.fold_id]
            accumulator.add_targets_batch(
                targets=fold_targets,
                frame=fold_frame,
                dates=fold_dates,
                target_column=target_column,
                context_columns=tuple(context_columns),
                feature_columns=feature_columns,
            )

    folds: list[dict[str, object]] = []
    for boundary in boundaries:
        fold_payload = accumulators[boundary.fold_id].finalize(_ABS_GT_THRESHOLDS)
        folds.append(
            {
                "fold_id": boundary.fold_id,
                "train_date_start": boundary.train_date_start,
                "train_date_end": boundary.train_date_end,
                "test_date_start": boundary.test_date_start,
                "test_date_end": boundary.test_date_end,
                "train_end_code": boundary.train_end_code,
                "train_target_count": fold_payload.get("count", 0),
                "eligible_train_row_count": fold_payload.get("count", 0),
                "target_summary": {
                    key: fold_payload[key] for key in ("min", "max", "count", *_TAIL_QUANTILE_KEYS)
                },
                "count_abs_gt": fold_payload["count_abs_gt"],
                "extreme_rows": fold_payload["extreme_rows"],
            }
        )

    return {
        "schema_version": "train_target_fold_audit.v1",
        "panel_path": str(panel_path),
        "target_column": target_column,
        "random_seed": random_seed,
        "rows_scanned": rows_scanned,
        "unique_date_count": int(len(unique_dates)),
        "train_row_policy": train_row_policy.as_dict(),
        "abs_gt_thresholds": list(_ABS_GT_THRESHOLDS),
        "feature_columns": list(feature_columns),
        "folds": folds,
    }


@dataclass
class _PredictionRangeAccumulator:
    prediction_min: float = float("inf")
    prediction_max: float = float("-inf")
    below_train_min_count: int = 0
    above_train_max_count: int = 0
    row_count: int = 0
    nonfinite_count: int = 0

    def update(self, predictions: np.ndarray, *, train_min: float, train_max: float) -> None:
        self.row_count += int(predictions.size)
        finite = predictions[np.isfinite(predictions)]
        self.nonfinite_count += int(predictions.size - finite.size)
        if finite.size == 0:
            return
        self.prediction_min = float(min(self.prediction_min, float(np.min(finite))))
        self.prediction_max = float(max(self.prediction_max, float(np.max(finite))))
        self.below_train_min_count += int(np.sum(finite < train_min))
        self.above_train_max_count += int(np.sum(finite > train_max))


def _fold_train_ranges(train_audit: dict[str, object]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    folds = train_audit.get("folds")
    if not isinstance(folds, list):
        return ranges
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        fold_id = str(fold.get("fold_id", ""))
        summary = fold.get("target_summary")
        if not isinstance(summary, dict):
            continue
        train_min = summary.get("min")
        train_max = summary.get("max")
        if train_min is None or train_max is None:
            continue
        ranges[fold_id] = (float(train_min), float(train_max))
    return ranges


def build_prediction_train_range_audit(
    pred_path: Path,
    panel_path: Path,
    boundaries: list[FoldBoundary],
    target_column: str,
    random_seed: int,
) -> dict[str, object]:
    """Compare streamed predictions against per-fold train-target ranges."""

    from pysrc.pipeline.panel.train_model_matrix import TrainRowPolicy

    _ = boundaries
    pred_path = Path(pred_path)
    train_audit = build_train_target_fold_audit(
        panel_path=panel_path,
        target_column=target_column,
        boundaries=boundaries,
        train_row_policy=TrainRowPolicy(general_max_rows=None, quantile_max_rows=None),
        random_seed=random_seed,
    )
    fold_ranges = _fold_train_ranges(train_audit)

    from pysrc.pipeline.panel.model_matrix_validation import _iter_prediction_batches

    accumulators: dict[tuple[str, str], _PredictionRangeAccumulator] = {}
    rows_scanned = 0

    for batch in _iter_prediction_batches(pred_path, batch_size=_BATCH_SIZE):
        rows_scanned += len(batch)
        if "interval" not in batch.columns:
            batch = batch.assign(interval="1d")
        batch["model_id"] = batch["model_id"].astype(str)
        batch["fold_id"] = batch["fold_id"].astype(str)
        batch["prediction"].to_numpy(dtype=np.float64)
        for (model_id, fold_id), fold_frame in batch.groupby(["model_id", "fold_id"], sort=True):
            train_range = fold_ranges.get(str(fold_id))
            if train_range is None:
                continue
            key = (str(model_id), str(fold_id))
            accumulator = accumulators.setdefault(key, _PredictionRangeAccumulator())
            fold_preds = fold_frame["prediction"].to_numpy(dtype=np.float64)
            accumulator.update(fold_preds, train_min=train_range[0], train_max=train_range[1])

    by_model_fold: list[dict[str, object]] = []
    for (model_id, fold_id), accumulator in sorted(accumulators.items()):
        train_min, train_max = fold_ranges[fold_id]
        pred_min = (
            None if accumulator.prediction_min == float("inf") else accumulator.prediction_min
        )
        pred_max = (
            None if accumulator.prediction_max == float("-inf") else accumulator.prediction_max
        )
        by_model_fold.append(
            {
                "model_id": model_id,
                "fold_id": fold_id,
                "train_target_min": train_min,
                "train_target_max": train_max,
                "prediction_min": pred_min,
                "prediction_max": pred_max,
                "below_train_min_count": accumulator.below_train_min_count,
                "above_train_max_count": accumulator.above_train_max_count,
                "row_count": accumulator.row_count,
                "nonfinite_prediction_count": accumulator.nonfinite_count,
            }
        )

    return {
        "schema_version": "prediction_train_range_audit.v1",
        "pred_path": str(pred_path),
        "panel_path": str(panel_path),
        "target_column": target_column,
        "random_seed": random_seed,
        "rows_scanned": rows_scanned,
        "train_target_fold_audit": train_audit,
        "by_model_fold": by_model_fold,
    }


__all__ = [
    "build_prediction_train_range_audit",
    "build_train_target_fold_audit",
]

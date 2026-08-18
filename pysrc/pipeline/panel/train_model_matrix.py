"""Walk-forward training of diverse panel models from canonical indicator panel."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.run_layout import allocate_run_dir
from pysrc.contracts.meta_router import (
    FORBIDDEN_FEATURE_PATTERNS,
    MODEL_PREDICTION_PANEL_COLUMNS,
)
from pysrc.models.registry import create_panel_model, model_entries_from_yaml
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.canonical_data import (
    CanonicalDataAudit,
    CanonicalPanelSource,
    assert_canonical_source_unchanged,
    require_canonical_panel_source_for_real_run,
    validate_canonical_panel,
)
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.p2_config_loader import (
    MetaRouterExperimentSpec,
    load_p2_config,
    parse_meta_router_experiment,
)
from pysrc.pipeline.panel.feature_usage import feature_usage_report
from pysrc.pipeline.panel.model_diversity import (
    build_diagnostic_coverage_report,
    build_low_memory_model_diversity_report,
    build_model_diversity_report,
    build_panel_target_lookup,
)
from pysrc.pipeline.panel.model_matrix_validation import (
    build_model_matrix_validation_bundle,
    build_streaming_prediction_sanity_report,
    extract_elastic_net_fit_diagnostics,
)
from pysrc.pipeline.panel.panel_targets import (
    resolve_panel_target_from_schema,
)
from pysrc.pipeline.panel.runtime import resolve_sklearn_n_jobs
from pysrc.pipeline.panel.sequence_data import build_sequence_windows

LOG = get_logger(__name__)

_HEAVY_TRAIN_FAMILIES = frozenset({"random_forest", "extra_trees", "xgboost", "mlp"})
_TREE_TRAIN_FAMILIES = frozenset({"random_forest", "extra_trees", "xgboost"})
_KEY_COLUMNS = ("date", "instrument", "interval")
_SCRATCH_OWNER_FILENAME = "scratch_owner.json"
MemoryMode = Literal["auto", "low_memory", "in_memory"]
TrainingMemoryCapability = Literal[
    "memmap_direct",
    "requires_contiguous_copy",
    "external_memory",
    "in_memory_only",
]


@dataclass(frozen=True, slots=True)
class FoldSlice:
    fold_id: str
    split: str
    train_mask: np.ndarray
    test_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class FoldBoundary:
    fold_id: str
    split: str
    train_start_code: int
    train_end_code: int
    test_start_code: int
    test_end_code: int
    purge_start_code: int
    train_date_start: str | None
    train_date_end: str | None
    test_date_start: str
    test_date_end: str
    purge_dates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainRowPolicy:
    general_max_rows: int | None
    quantile_max_rows: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "general_max_rows": self.general_max_rows,
            "quantile_max_rows": self.quantile_max_rows,
        }


@dataclass(frozen=True, slots=True)
class MemoryModeDecision:
    requested_memory_mode: str
    resolved_memory_mode: Literal["low_memory", "in_memory"]
    selection_reason: str
    estimated_in_memory_peak_bytes: int
    available_memory_bytes: int | None
    memory_safety_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_memory_mode": self.requested_memory_mode,
            "resolved_memory_mode": self.resolved_memory_mode,
            "selection_reason": self.selection_reason,
            "estimated_in_memory_peak_bytes": self.estimated_in_memory_peak_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_safety_fraction": self.memory_safety_fraction,
        }


@dataclass(frozen=True, slots=True)
class ScratchPanel:
    scratch_dir: Path
    feature_path: Path
    target_path: Path
    date_code_path: Path
    instrument_code_path: Path
    interval_code_path: Path
    finite_target_path: Path
    group_index_dir: Path
    n_rows: int
    n_features: int
    features: tuple[str, ...]
    target_column: str
    unique_dates: tuple[str, ...]
    unique_instruments: tuple[str, ...]
    unique_intervals: tuple[str, ...]
    duplicate_key_count: int
    target_finite_count: int
    chronological_source_order: bool
    scratch_bytes: int
    group_index_row_count: int


def resolve_train_matrix_scratch_dir(*, config: P2Config, run_path: Path) -> Path:
    """Resolve run-scoped scratch directory for low-memory train-matrix."""
    if config.panel_train_scratch_dir:
        return Path(config.panel_train_scratch_dir) / run_path.name / "train_matrix"
    return run_path / "scratch" / "train_matrix"


def initialize_train_matrix_scratch_dir(
    scratch_dir: Path,
    *,
    run_id: str,
    source: CanonicalPanelSource,
) -> None:
    """Create a clean scratch directory and stamp run ownership."""
    owner_path = scratch_dir / _SCRATCH_OWNER_FILENAME
    if scratch_dir.exists():
        if owner_path.is_file():
            payload = json.loads(owner_path.read_text(encoding="utf-8"))
            owner_run_id = str(payload.get("run_id", ""))
            if owner_run_id and owner_run_id != run_id:
                raise RuntimeError(
                    "Train-matrix scratch directory is owned by another run: "
                    f"expected run_id={run_id}, found run_id={owner_run_id}, "
                    f"path={scratch_dir}"
                )
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        owner_path,
        {
            "schema_version": "train_matrix_scratch_owner.v1",
            "run_id": run_id,
            "source": source.as_dict(),
        },
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    return value


def _schema_from_frame(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _is_numeric_schema_type(type_name: str) -> bool:
    lowered = type_name.lower()
    return any(token in lowered for token in ("float", "double", "int", "uint", "decimal"))


def resolve_schema_target_and_features(
    config: P2Config,
    schema: dict[str, str],
    *,
    manifest: dict[str, object] | None = None,
) -> tuple[str, list[str], dict[str, object], dict[str, object]]:
    """Resolve target and eligible numeric features without a full pandas panel."""

    manifest = dict(manifest or {})
    target_column, target_metadata = resolve_panel_target_from_schema(config, schema, manifest)
    excluded_names = {
        "date",
        "instrument",
        "symbol",
        "ticker",
        "interval",
        target_column,
        "forward_return",
        "forward_return_horizon",
        "adjusted_return_1d",
        "raw_return_1d",
    }
    indicator_columns = manifest.get("indicator_columns")
    ordered_candidates = (
        [str(col) for col in indicator_columns]
        if isinstance(indicator_columns, list) and indicator_columns
        else sorted(schema)
    )
    features: list[str] = []
    excluded: list[dict[str, object]] = []
    forbidden = tuple(FORBIDDEN_FEATURE_PATTERNS)
    for column in ordered_candidates:
        if column in excluded_names:
            continue
        lowered = column.lower()
        if any(pattern in lowered for pattern in forbidden) or lowered.endswith("_label"):
            excluded.append({"feature": column, "reason": "forbidden_feature_name"})
            continue
        type_name = schema.get(column)
        if type_name is None:
            excluded.append({"feature": column, "reason": "missing_from_schema"})
            continue
        if not _is_numeric_schema_type(type_name):
            excluded.append({"feature": column, "reason": "non_numeric_unsupported"})
            continue
        features.append(column)

    if not features:
        for column, type_name in sorted(schema.items()):
            if column in excluded_names:
                continue
            lowered = column.lower()
            if any(
                pattern in lowered for pattern in FORBIDDEN_FEATURE_PATTERNS
            ) or lowered.endswith("_label"):
                continue
            if _is_numeric_schema_type(type_name):
                features.append(column)

    report = feature_usage_report(
        eligible_feature_count=len(features),
        used_feature_count=len(features),
        used_all_eligible_features=True,
        excluded_features=excluded[:100],
    )
    return target_column, features, report, target_metadata


def normalize_date_labels(dates: np.ndarray | pd.Series | list[object]) -> pd.Series:
    """Normalize canonical panel dates to comparable UTC ISO date labels."""

    date_series = pd.Series(dates, dtype="string")
    return pd.to_datetime(date_series, errors="raise", utc=True, format="mixed").dt.strftime(
        "%Y-%m-%d"
    )


def build_chronological_date_codes(
    dates: np.ndarray | pd.Series | list[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row chronological date ordinals and sorted ISO date labels."""

    normalized = normalize_date_labels(dates)
    unique_dates = np.asarray(sorted(normalized.dropna().unique().tolist()), dtype=object)
    mapping = {str(value): idx for idx, value in enumerate(unique_dates.tolist())}
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        raise ValueError("Canonical panel contains missing or un-normalizable dates")
    codes = mapped.to_numpy(dtype=np.intp)
    return codes, unique_dates


def _resolve_walk_forward_fold_count(config: P2Config) -> int:
    configured = int(config.panel_walk_forward_folds)
    if config.smoke_test:
        return configured if configured > 0 else 2
    return configured if configured > 0 else 3


def _selected_instruments(config: P2Config, instruments: list[str] | set[str]) -> frozenset[str]:
    cap = int(config.panel_max_instruments)
    unique = sorted({str(item) for item in instruments})
    if cap <= 0 or len(unique) <= cap:
        return frozenset(unique)
    digest = hashlib.sha256(f"{int(config.random_seed)}:panel_max_instruments".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    picked = rng.choice(np.asarray(unique, dtype=object), size=cap, replace=False)
    return frozenset(str(item) for item in picked.tolist())


def _filter_panel_instruments(panel: pd.DataFrame, config: P2Config) -> pd.DataFrame:
    if "instrument" not in panel.columns or int(config.panel_max_instruments) <= 0:
        return panel
    allowed = _selected_instruments(config, panel["instrument"].astype(str).tolist())
    if len(allowed) == int(panel["instrument"].astype(str).nunique()):
        return panel
    return panel.loc[panel["instrument"].astype(str).isin(allowed)].copy()


def build_walk_forward_boundaries(
    unique_dates: np.ndarray,
    *,
    n_folds: int = 3,
    target_horizon_days: int = 0,
) -> list[FoldBoundary]:
    """Build chronological fold boundaries with label-horizon purging."""

    n_unique_dates = int(len(unique_dates))
    if n_unique_dates < 2:
        return []
    if n_unique_dates < n_folds + 1:
        n_folds = max(1, n_unique_dates - 1)
    fold_size = max(1, n_unique_dates // (n_folds + 1))
    # Horizon is measured in canonical trading-date ordinals, not calendar days.
    # A 1-bar forward target for Friday may end on Monday, so calendar timedelta
    # arithmetic would leak across weekends and holidays.
    horizon_width = int(target_horizon_days)
    folds: list[FoldBoundary] = []
    for fold_idx in range(n_folds):
        test_start = fold_size * (fold_idx + 1)
        test_end = min(test_start + fold_size, n_unique_dates)
        if test_start >= n_unique_dates:
            break
        train_end = max(0, test_start - horizon_width)
        purge_start = train_end
        purge_dates = tuple(str(value) for value in unique_dates[purge_start:test_start].tolist())
        train_start_label = str(unique_dates[0]) if train_end > 0 else None
        train_end_label = str(unique_dates[train_end - 1]) if train_end > 0 else None
        folds.append(
            FoldBoundary(
                fold_id=f"fold_{fold_idx}",
                split="test",
                train_start_code=0,
                train_end_code=train_end,
                test_start_code=test_start,
                test_end_code=test_end,
                purge_start_code=purge_start,
                train_date_start=train_start_label,
                train_date_end=train_end_label,
                test_date_start=str(unique_dates[test_start]),
                test_date_end=str(unique_dates[test_end - 1]),
                purge_dates=purge_dates,
            )
        )
    return folds


def boundaries_from_fold_date_policy(
    unique_dates: np.ndarray,
    fold_policy: dict[str, Any],
) -> list[FoldBoundary]:
    """Resolve fold boundaries from saved date labels when numeric codes are absent."""

    entries = fold_policy.get("fold_checks") or fold_policy.get("folds") or []
    if not entries:
        return []
    date_map = {str(value): idx for idx, value in enumerate(unique_dates.tolist())}

    def _exclusive_end_code(label: object) -> int | None:
        if label is None or str(label).strip() == "":
            return None
        normalized = str(normalize_date_labels([label]).iloc[0])
        idx = date_map.get(normalized)
        if idx is None:
            return None
        return int(idx) + 1

    def _inclusive_start_code(label: object) -> int | None:
        if label is None or str(label).strip() == "":
            return None
        normalized = str(normalize_date_labels([label]).iloc[0])
        return date_map.get(normalized)

    boundaries: list[FoldBoundary] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        train_start_label = entry.get("train_date_start")
        train_end_label = entry.get("train_date_end")
        test_start_label = entry.get("test_date_start")
        test_end_label = entry.get("test_date_end")
        train_end_code = _exclusive_end_code(train_end_label)
        test_start_code = _inclusive_start_code(test_start_label)
        test_end_code = _exclusive_end_code(test_end_label)
        if train_end_code is None or test_start_code is None or test_end_code is None:
            continue
        train_start_code = _inclusive_start_code(train_start_label)
        if train_start_code is None:
            train_start_code = 0
        purge_dates = tuple(str(value) for value in (entry.get("purge_dates") or ()))
        purge_start_code = train_end_code
        if purge_dates:
            purge_idx = _inclusive_start_code(purge_dates[0])
            if purge_idx is not None:
                purge_start_code = int(purge_idx)
        boundaries.append(
            FoldBoundary(
                fold_id=str(entry.get("fold_id", "")),
                split=str(entry.get("split", "test")),
                train_start_code=int(train_start_code),
                train_end_code=int(train_end_code),
                test_start_code=int(test_start_code),
                test_end_code=int(test_end_code),
                purge_start_code=int(purge_start_code),
                train_date_start=str(train_start_label) if train_start_label else None,
                train_date_end=str(train_end_label) if train_end_label else None,
                test_date_start=str(test_start_label),
                test_date_end=str(test_end_label),
                purge_dates=purge_dates,
            )
        )
    return boundaries


def fold_masks_from_boundaries(
    date_codes: np.ndarray,
    boundary: FoldBoundary,
) -> tuple[np.ndarray, np.ndarray]:
    train_mask = (date_codes >= boundary.train_start_code) & (date_codes < boundary.train_end_code)
    test_mask = (date_codes >= boundary.test_start_code) & (date_codes < boundary.test_end_code)
    return train_mask, test_mask


def _sequence_length_from_entry(entry: dict[str, Any]) -> int | None:
    raw = entry.get("sequence_length")
    if raw is not None:
        return int(raw)
    if str(entry.get("family")) == "lstm":
        return 20
    return None


def _is_sequence_entry(entry: dict[str, Any]) -> bool:
    return _sequence_length_from_entry(entry) is not None


def _sequence_model_id(family: str, sequence_length: int) -> str:
    return f"{family}_seq{sequence_length}"


def _sequence_fold_masks(
    meta: pd.DataFrame,
    boundary: FoldBoundary,
    unique_dates: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_dates = set(unique_dates[boundary.train_start_code : boundary.train_end_code])
    test_dates = set(unique_dates[boundary.test_start_code : boundary.test_end_code])
    meta_dates = meta["date"].astype(str)
    train_mask = meta_dates.isin(train_dates).to_numpy(dtype=np.bool_)
    test_mask = meta_dates.isin(test_dates).to_numpy(dtype=np.bool_)
    return train_mask, test_mask


def build_walk_forward_folds(dates: list[str], *, n_folds: int = 3) -> list[FoldSlice]:
    """Build chronological fold masks from a per-row date list."""

    codes, unique_dates = build_chronological_date_codes(dates)
    boundaries = build_walk_forward_boundaries(unique_dates, n_folds=n_folds, target_horizon_days=0)
    return [
        FoldSlice(boundary.fold_id, boundary.split, *fold_masks_from_boundaries(codes, boundary))
        for boundary in boundaries
    ]


def build_walk_forward_folds_from_codes(
    date_codes: np.ndarray,
    *,
    n_unique_dates: int,
    n_folds: int = 3,
) -> list[FoldSlice]:
    """Vectorized walk-forward folds keyed by chronological date codes."""

    if n_unique_dates < 2:
        return []
    if n_unique_dates < n_folds + 1:
        n_folds = max(1, n_unique_dates - 1)
    fold_size = max(1, n_unique_dates // (n_folds + 1))
    folds: list[FoldSlice] = []
    for fold_idx in range(n_folds):
        test_start = fold_size * (fold_idx + 1)
        test_end = min(test_start + fold_size, n_unique_dates)
        if test_start >= n_unique_dates:
            break
        train_mask = date_codes < test_start
        test_mask = (date_codes >= test_start) & (date_codes < test_end)
        folds.append(FoldSlice(f"fold_{fold_idx}", "test", train_mask, test_mask))
    return folds


def _subsample_seed(master_seed: int, family: str, fold_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{family}:{fold_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def subsample_train_indices(
    train_indices: np.ndarray,
    *,
    max_rows: int,
    master_seed: int,
    family: str,
    fold_id: str,
) -> np.ndarray:
    """Deterministically cap training rows for model families."""

    if max_rows <= 0 or len(train_indices) <= max_rows:
        return train_indices
    rng = np.random.default_rng(_subsample_seed(master_seed, family, fold_id))
    chosen = rng.choice(train_indices, size=max_rows, replace=False)
    return np.sort(chosen.astype(np.intp, copy=False))


def _positive_or_none(value: int) -> int | None:
    if value < 0:
        raise ValueError("train row caps must be non-negative")
    return None if value == 0 else int(value)


def resolve_train_row_policy(
    config: P2Config,
    *,
    cli_max_train_rows: int | None = None,
    cli_quantile_max_train_rows: int | None = None,
) -> TrainRowPolicy:
    general = _positive_or_none(int(config.panel_train_max_rows_per_fold))
    quantile = _positive_or_none(int(config.panel_quantile_max_train_rows))
    if cli_max_train_rows is not None:
        general = _positive_or_none(int(cli_max_train_rows))
        quantile = _positive_or_none(int(cli_max_train_rows))
    if cli_quantile_max_train_rows is not None:
        quantile = _positive_or_none(int(cli_quantile_max_train_rows))
    return TrainRowPolicy(general_max_rows=general, quantile_max_rows=quantile)


def resolve_train_row_limit(*, family: str, config: P2Config) -> int:
    """Per-family training row cap; 0 means use all finite train rows."""

    policy = resolve_train_row_policy(config)
    limit = policy.quantile_max_rows if family == "quantile_regression" else policy.general_max_rows
    return 0 if limit is None else int(limit)


def resolve_matrix_sklearn_n_jobs(
    config: P2Config,
    *,
    family: str,
    train_rows: int,
) -> int:
    """Limit tree parallelism on very large folds to reduce memory spikes."""

    base = resolve_sklearn_n_jobs(config, parallel_workers=1)
    if family not in _TREE_TRAIN_FAMILIES or config.panel_train_max_rows_per_fold > 0:
        return base
    if train_rows > 1_500_000:
        return min(base, 4)
    if train_rows > 750_000:
        return min(base, 8)
    return base


def resolve_panel_memory_mode(
    config: P2Config,
    *,
    panel_rows: int,
    feature_count: int,
    model_count: int,
    largest_train_rows: int,
    largest_test_rows: int,
    available_memory_bytes: int | None,
) -> MemoryModeDecision:
    requested = str(config.panel_train_memory_mode)
    feature_bytes = int(panel_rows) * int(feature_count) * np.dtype(np.float32).itemsize
    target_bytes = int(panel_rows) * np.dtype(np.float32).itemsize
    key_bytes = int(panel_rows) * 3 * np.dtype(np.int32).itemsize
    train_copy_bytes = int(largest_train_rows) * int(feature_count) * np.dtype(np.float32).itemsize
    test_copy_bytes = int(largest_test_rows) * int(feature_count) * np.dtype(np.float32).itemsize
    prediction_bytes = int(largest_test_rows) * max(int(model_count), 1) * 64
    estimated = (
        feature_bytes
        + target_bytes
        + key_bytes
        + (2 * train_copy_bytes)
        + test_copy_bytes
        + prediction_bytes
    )
    if config.smoke_test or requested == "in_memory":
        return MemoryModeDecision(
            requested,
            "in_memory",
            "smoke_or_explicit_in_memory" if config.smoke_test else "explicit_in_memory",
            estimated,
            available_memory_bytes,
            float(config.panel_train_memory_safety_fraction),
        )
    if requested == "low_memory":
        return MemoryModeDecision(
            requested,
            "low_memory",
            "explicit_low_memory",
            estimated,
            available_memory_bytes,
            float(config.panel_train_memory_safety_fraction),
        )
    if requested != "auto":
        raise ValueError(f"Unsupported panel_train_memory_mode={requested!r}")
    if available_memory_bytes is None:
        return MemoryModeDecision(
            requested,
            "low_memory",
            "auto_no_available_memory_metric",
            estimated,
            available_memory_bytes,
            float(config.panel_train_memory_safety_fraction),
        )
    allowance = int(available_memory_bytes * float(config.panel_train_memory_safety_fraction))
    if estimated > allowance:
        return MemoryModeDecision(
            requested,
            "low_memory",
            "estimated_peak_exceeds_safety_allowance",
            estimated,
            available_memory_bytes,
            float(config.panel_train_memory_safety_fraction),
        )
    return MemoryModeDecision(
        requested,
        "in_memory",
        "estimated_peak_within_safety_allowance",
        estimated,
        available_memory_bytes,
        float(config.panel_train_memory_safety_fraction),
    )


def _available_memory_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except (AttributeError, OSError):
        return None


def _process_tree_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process()
        total = int(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
    except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None


class PeakRssMonitor:
    """Sample process-tree RSS while native estimators run."""

    def __init__(self, *, interval_s: float = 0.10) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes: int | None = _process_tree_rss_bytes()

    def __enter__(self) -> PeakRssMonitor:
        def sample() -> None:
            while not self._stop.wait(self._interval_s):
                rss = _process_tree_rss_bytes()
                if rss is not None:
                    self.peak_bytes = max(int(self.peak_bytes or 0), int(rss))

        self._thread = threading.Thread(target=sample, name="model-matrix-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        rss = _process_tree_rss_bytes()
        if rss is not None:
            self.peak_bytes = max(int(self.peak_bytes or 0), int(rss))


def _enforce_peak_rss_limit(peak_bytes: int | None, limit_bytes: int, *, context: str) -> None:
    if limit_bytes > 0 and peak_bytes is not None and peak_bytes > limit_bytes:
        raise MemoryError(
            f"model_matrix {context} peak process-tree RSS {peak_bytes} "
            f"exceeded configured limit {limit_bytes}"
        )


def _index_dtype(length: int) -> type[np.integer[Any]]:
    return np.int32 if length < np.iinfo(np.int32).max else np.intp


def _model_memory_capabilities(yaml_models: list[dict[str, Any]]) -> dict[str, str]:
    capabilities: dict[str, str] = {}
    for entry in yaml_models:
        family = str(entry["family"])
        if _is_sequence_entry(entry):
            capabilities[family] = "in_memory_only"
            continue
        if family in {
            "ridge",
            "elastic_net",
            "bayesian_ridge",
            "random_forest",
            "extra_trees",
            "xgboost",
            "quantile_regression",
        } or family in {"pcr", "pls", "mlp"}:
            capabilities[family] = "requires_contiguous_copy"
        else:
            capabilities[family] = "in_memory_only"
    return capabilities


def _validate_low_memory_capabilities(yaml_models: list[dict[str, Any]]) -> dict[str, str]:
    capabilities = _model_memory_capabilities(yaml_models)
    incompatible = [family for family, cap in capabilities.items() if cap == "in_memory_only"]
    if incompatible:
        raise ValueError(
            "Strict low-memory train-matrix cannot run unsupported model families: "
            f"{sorted(incompatible)}"
        )
    return capabilities


def rank_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    group_keys = ["model_id", "fold_id", "date", "interval"]
    ranked["prediction_rank"] = (
        ranked.groupby(group_keys, sort=False)["prediction"]
        .rank(ascending=False, method="dense")
        .astype(np.int32)
    )
    return ranked


def _prediction_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_family", pa.string(), nullable=False),
            pa.field("instrument", pa.string(), nullable=False),
            pa.field("date", pa.string(), nullable=False),
            pa.field("interval", pa.string(), nullable=False),
            pa.field("fold_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("prediction", pa.float64(), nullable=False),
            pa.field("prediction_rank", pa.int32(), nullable=False),
            pa.field("confidence", pa.float64(), nullable=False),
            pa.field("target_name", pa.string(), nullable=False),
        ]
    )


def _ordered_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in MODEL_PREDICTION_PANEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[list(MODEL_PREDICTION_PANEL_COLUMNS)].copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise").astype(float)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="raise").astype(float)
    if not np.isfinite(frame["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction panel contains non-finite predictions")
    if not np.isfinite(frame["confidence"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction panel contains non-finite confidence values")
    frame["prediction_rank"] = frame["prediction_rank"].astype(np.int32)
    return frame.sort_values(
        ["model_id", "fold_id", "date", "interval", "instrument"], kind="mergesort"
    ).reset_index(drop=True)


def _prediction_table(frame: pd.DataFrame) -> Any:
    import pyarrow as pa

    ordered = _ordered_prediction_frame(frame)
    return pa.Table.from_pandas(ordered, schema=_prediction_schema(), preserve_index=False)


def _write_prediction_fragment(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    pq.write_table(_prediction_table(frame), tmp, compression="snappy")
    os.replace(tmp, path)


def _append_prediction_chunk(writer: Any, frame: pd.DataFrame) -> None:
    writer.write_table(_prediction_table(frame))


def _resolve_source_manifest(source: CanonicalPanelSource) -> dict[str, object]:
    manifest = dict(source.manifest)
    if source.target_metadata:
        existing = manifest.get("target_metadata")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in source.target_metadata.items():
            if isinstance(value, dict):
                current = dict(merged.get(key, {})) if isinstance(merged.get(key), dict) else {}
                merged[key] = {**current, **value}
            else:
                merged[key] = value
        manifest["target_metadata"] = merged
    return manifest


def _canonical_audit_from_scratch(
    source: CanonicalPanelSource,
    scratch: ScratchPanel,
) -> CanonicalDataAudit:
    channel_manifest = {
        "primary_channel_id": "technical",
        "channel_ids": ["technical"],
    }
    return CanonicalDataAudit(
        canonical_source_used=source.product_identity == "full_indicator_feature_panel",
        legacy_artifact_fallback_used=False,
        grain_valid=scratch.duplicate_key_count == 0,
        duplicate_key_count=scratch.duplicate_key_count,
        row_count=scratch.n_rows,
        ticker_count=len(scratch.unique_instruments),
        date_start=scratch.unique_dates[0] if scratch.unique_dates else None,
        date_end=scratch.unique_dates[-1] if scratch.unique_dates else None,
        intervals_detected=scratch.unique_intervals,
        panel_path=str(source.panel_path) if source.panel_path is not None else None,
        channel_manifest=channel_manifest,
    )


def _estimate_low_memory_scratch_bytes(
    *, row_count: int, feature_count: int, model_count: int
) -> int:
    canonical_bytes = row_count * ((feature_count * 4) + 4 + (4 * 3) + 1 + 8 + 8)
    largest_training_memmap_bytes = int(row_count * 0.75) * feature_count * 4
    prediction_bytes = row_count * max(1, model_count) * 128
    temp_prediction_bytes = prediction_bytes
    return int(
        (canonical_bytes + largest_training_memmap_bytes + prediction_bytes + temp_prediction_bytes)
        * 1.35
    )


def _safe_max_packed(n_dates: int, n_instruments: int, n_intervals: int) -> int:
    for value, name in (
        (n_dates, "date"),
        (n_instruments, "instrument"),
        (n_intervals, "interval"),
    ):
        if value <= 0:
            raise ValueError(f"Cannot pack keys with empty {name} dictionary")
    max_value = (int(n_dates) * int(n_instruments) * int(n_intervals)) - 1
    if max_value > np.iinfo(np.uint64).max:
        raise OverflowError("Packed key cardinality exceeds uint64")
    return max_value


def _packed_keys(
    date_codes: np.ndarray,
    instrument_codes: np.ndarray,
    interval_codes: np.ndarray,
    *,
    n_instruments: int,
    n_intervals: int,
) -> np.ndarray:
    if (date_codes < 0).any() or (instrument_codes < 0).any() or (interval_codes < 0).any():
        raise ValueError("Cannot pack negative key codes")
    return (
        (date_codes.astype(np.uint64) * np.uint64(n_instruments))
        + instrument_codes.astype(np.uint64)
    ) * np.uint64(n_intervals) + interval_codes.astype(np.uint64)


def _prepare_scratch_panel(
    *,
    source: CanonicalPanelSource,
    config: P2Config,
    run_path: Path,
    features: list[str],
    target_column: str,
    model_count: int,
) -> ScratchPanel:
    if source.panel_path is None:
        raise ValueError("Low-memory training requires a parquet-backed canonical panel")
    import pyarrow.parquet as pq

    scratch_dir = resolve_train_matrix_scratch_dir(config=config, run_path=run_path)
    initialize_train_matrix_scratch_dir(scratch_dir, run_id=run_path.name, source=source)
    estimated_scratch_bytes = _estimate_low_memory_scratch_bytes(
        row_count=source.row_count,
        feature_count=len(features),
        model_count=model_count,
    )
    free_bytes = shutil.disk_usage(scratch_dir).free
    if free_bytes < estimated_scratch_bytes:
        raise OSError(
            "Insufficient scratch space before panel scan: "
            f"estimated={estimated_scratch_bytes} free={free_bytes} scratch_dir={scratch_dir}"
        )

    parquet = pq.ParquetFile(source.panel_path)
    key_columns = list(_KEY_COLUMNS)
    scan_columns = [*key_columns, target_column]
    unique_dates: set[str] = set()
    unique_instruments: set[str] = set()
    unique_intervals: set[str] = set()
    previous_date: str | None = None
    chronological = True
    target_finite_count = 0
    missing_key_count = 0
    chunk_rows = int(config.panel_train_chunk_rows)
    for batch in parquet.iter_batches(columns=scan_columns, batch_size=chunk_rows):
        pdf = batch.to_pandas()
        if pdf[list(_KEY_COLUMNS)].isna().any(axis=None):
            missing_key_count += int(pdf[list(_KEY_COLUMNS)].isna().any(axis=1).sum())
        dates = normalize_date_labels(pdf["date"])
        instruments = pdf["instrument"].astype(str)
        intervals = pdf["interval"].astype(str)
        if previous_date is not None and len(dates) and str(dates.iloc[0]) < previous_date:
            chronological = False
        if len(dates) > 1 and not dates.is_monotonic_increasing:
            chronological = False
        if len(dates):
            previous_date = str(dates.iloc[-1])
        unique_dates.update(dates.tolist())
        unique_instruments.update(instruments.tolist())
        unique_intervals.update(intervals.tolist())
        target_finite_count += int(
            np.isfinite(
                pd.to_numeric(pdf[target_column], errors="coerce").to_numpy(dtype=float)
            ).sum()
        )
    if missing_key_count:
        raise ValueError(
            f"Canonical panel has {missing_key_count} rows with missing key components"
        )

    date_values = np.asarray(sorted(unique_dates), dtype=object)
    instrument_values = np.asarray(sorted(unique_instruments), dtype=object)
    interval_values = np.asarray(sorted(unique_intervals), dtype=object)
    _safe_max_packed(len(date_values), len(instrument_values), len(interval_values))
    date_map = {str(value): idx for idx, value in enumerate(date_values.tolist())}
    instrument_map = {str(value): idx for idx, value in enumerate(instrument_values.tolist())}
    interval_map = {str(value): idx for idx, value in enumerate(interval_values.tolist())}

    n_rows = int(source.row_count)
    n_features = len(features)
    feature_path = scratch_dir / "features.float32.memmap"
    target_path = scratch_dir / "target.float32.memmap"
    date_code_path = scratch_dir / "date_codes.int32.memmap"
    instrument_code_path = scratch_dir / "instrument_codes.int32.memmap"
    interval_code_path = scratch_dir / "interval_codes.int32.memmap"
    finite_target_path = scratch_dir / "finite_target.bool.memmap"
    group_index_dir = scratch_dir / "group_indices"
    group_index_dir.mkdir(parents=True, exist_ok=True)
    x_mm = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features))
    y_mm = np.memmap(target_path, dtype=np.float32, mode="w+", shape=(n_rows,))
    date_codes_mm = np.memmap(date_code_path, dtype=np.int32, mode="w+", shape=(n_rows,))
    instrument_codes_mm = np.memmap(
        instrument_code_path, dtype=np.int32, mode="w+", shape=(n_rows,)
    )
    interval_codes_mm = np.memmap(interval_code_path, dtype=np.int32, mode="w+", shape=(n_rows,))
    finite_mm = np.memmap(finite_target_path, dtype=np.bool_, mode="w+", shape=(n_rows,))

    bucket_count = 256
    bucket_paths = [scratch_dir / f"packed_keys_{idx:03d}.bin" for idx in range(bucket_count)]
    bucket_files = [path.open("ab") for path in bucket_paths]
    try:
        offset = 0
        columns = [*key_columns, *features, target_column]
        for batch in parquet.iter_batches(columns=columns, batch_size=chunk_rows):
            pdf = batch.to_pandas()
            length = len(pdf)
            next_offset = offset + length
            if next_offset > n_rows:
                raise ValueError("Parquet scan produced more rows than metadata advertised")
            dates = normalize_date_labels(pdf["date"]).map(date_map).to_numpy(dtype=np.int32)
            instruments = pdf["instrument"].astype(str).map(instrument_map).to_numpy(dtype=np.int32)
            intervals = pdf["interval"].astype(str).map(interval_map).to_numpy(dtype=np.int32)
            if (dates < 0).any() or (instruments < 0).any() or (intervals < 0).any():
                raise ValueError("Dictionary encoding produced negative key codes")
            date_codes_mm[offset:next_offset] = dates
            instrument_codes_mm[offset:next_offset] = instruments
            interval_codes_mm[offset:next_offset] = intervals
            targets = pd.to_numeric(pdf[target_column], errors="coerce").to_numpy(dtype=np.float32)
            y_mm[offset:next_offset] = targets
            finite_mm[offset:next_offset] = np.isfinite(targets)
            values = pdf[features].to_numpy(dtype=np.float32, copy=True)
            np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            x_mm[offset:next_offset, :] = values
            keys = _packed_keys(
                dates,
                instruments,
                intervals,
                n_instruments=len(instrument_values),
                n_intervals=len(interval_values),
            )
            buckets = np.mod(keys, bucket_count).astype(np.int16, copy=False)
            for bucket_id in np.unique(buckets):
                part = keys[buckets == bucket_id]
                part.tofile(bucket_files[int(bucket_id)])
            row_indices = np.arange(offset, next_offset, dtype=np.int64)
            group_ids = (dates.astype(np.int64) * len(interval_values)) + intervals.astype(np.int64)
            order = np.argsort(group_ids, kind="mergesort")
            sorted_groups = group_ids[order]
            sorted_rows = row_indices[order]
            starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
            ends = np.r_[starts[1:], len(sorted_groups)]
            for start_idx, end_idx in zip(starts, ends, strict=True):
                group_id = int(sorted_groups[start_idx])
                group_path = group_index_dir / f"group_{group_id:08d}.int64"
                with group_path.open("ab") as group_handle:
                    sorted_rows[start_idx:end_idx].astype(np.int64, copy=False).tofile(group_handle)
            offset = next_offset
        if offset != n_rows:
            raise ValueError(f"Parquet scan produced {offset} rows, expected {n_rows}")
    finally:
        for handle in bucket_files:
            handle.close()
        x_mm.flush()
        y_mm.flush()
        date_codes_mm.flush()
        instrument_codes_mm.flush()
        interval_codes_mm.flush()
        finite_mm.flush()
        del x_mm, y_mm, date_codes_mm, instrument_codes_mm, interval_codes_mm, finite_mm

    duplicate_key_count = 0
    for bucket_path in bucket_paths:
        keys = np.fromfile(bucket_path, dtype=np.uint64)
        if len(keys) <= 1:
            continue
        keys.sort(kind="mergesort")
        duplicate_key_count += int(np.count_nonzero(keys[1:] == keys[:-1]))
    if duplicate_key_count:
        raise ValueError(
            f"Canonical panel has duplicate ticker/date/interval keys: {duplicate_key_count}"
        )

    group_index_row_count = _assert_group_index_row_count(
        group_index_dir=group_index_dir,
        expected_rows=n_rows,
    )

    scratch_bytes = sum(path.stat().st_size for path in scratch_dir.rglob("*") if path.is_file())
    return ScratchPanel(
        scratch_dir=scratch_dir,
        feature_path=feature_path,
        target_path=target_path,
        date_code_path=date_code_path,
        instrument_code_path=instrument_code_path,
        interval_code_path=interval_code_path,
        finite_target_path=finite_target_path,
        group_index_dir=group_index_dir,
        n_rows=n_rows,
        n_features=n_features,
        features=tuple(features),
        target_column=target_column,
        unique_dates=tuple(str(x) for x in date_values.tolist()),
        unique_instruments=tuple(str(x) for x in instrument_values.tolist()),
        unique_intervals=tuple(str(x) for x in interval_values.tolist()),
        duplicate_key_count=duplicate_key_count,
        target_finite_count=target_finite_count,
        chronological_source_order=chronological,
        scratch_bytes=scratch_bytes,
        group_index_row_count=group_index_row_count,
    )


def _fold_report(boundaries: list[FoldBoundary], target_horizon_days: int) -> dict[str, object]:
    return {
        "target_horizon_days": target_horizon_days,
        "folds": [
            {
                "fold_id": b.fold_id,
                "train_date_start": b.train_date_start,
                "train_date_end": b.train_date_end,
                "test_date_start": b.test_date_start,
                "test_date_end": b.test_date_end,
                "purge_dates": list(b.purge_dates),
            }
            for b in boundaries
        ],
    }


def _finite_train_indices(
    *,
    date_codes: np.ndarray,
    finite_target: np.ndarray,
    boundary: FoldBoundary,
    index_dtype: type[np.integer[Any]],
) -> np.ndarray:
    train_mask, _ = fold_masks_from_boundaries(date_codes, boundary)
    train_mask &= finite_target
    return np.flatnonzero(train_mask).astype(index_dtype, copy=False)


def _finite_train_indices_from_groups(
    *,
    scratch: ScratchPanel,
    finite_target: np.ndarray,
    instrument_codes: np.ndarray,
    boundary: FoldBoundary,
    index_dtype: type[np.integer[Any]],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    interval_count = len(scratch.unique_intervals)
    for date_code in range(0, boundary.train_end_code):
        for interval_code in range(interval_count):
            group_id = int(date_code * interval_count + interval_code)
            group_path = scratch.group_index_dir / f"group_{group_id:08d}.int64"
            if not group_path.is_file():
                continue
            rows = np.fromfile(group_path, dtype=np.int64)
            if len(rows) == 0:
                continue
            rows = rows[np.asarray(finite_target[rows], dtype=np.bool_)]
            if len(rows) == 0:
                continue
            order = np.argsort(instrument_codes[rows], kind="mergesort")
            parts.append(rows[order].astype(index_dtype, copy=False))
    if not parts:
        return np.empty(0, dtype=index_dtype)
    return np.concatenate(parts).astype(index_dtype, copy=False)


def _compact_training_matrix(
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_indices: np.ndarray,
    scratch_dir: Path | None,
    family: str,
    fold_id: str,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    y_train = np.asarray(y_all[train_indices], dtype=np.float32)
    if scratch_dir is None:
        return np.asarray(x_all[train_indices], dtype=np.float32), y_train, None
    path = scratch_dir / f"train_{family}_{fold_id}_{uuid4().hex}.float32.memmap"
    x_train = np.memmap(
        path, dtype=np.float32, mode="w+", shape=(len(train_indices), x_all.shape[1])
    )
    for start in range(0, len(train_indices), chunk_rows):
        end = min(start + chunk_rows, len(train_indices))
        x_train[start:end, :] = x_all[train_indices[start:end], :]
    x_train.flush()
    return x_train, y_train, path


def _group_index_row_count(path: Path) -> int:
    size = path.stat().st_size
    itemsize = np.dtype(np.int64).itemsize
    if size % itemsize:
        raise ValueError(f"Group-index file is truncated or corrupt: {path}")
    return size // itemsize


def _total_group_index_row_count(group_index_dir: Path) -> int:
    total = 0
    for path in sorted(group_index_dir.glob("group_*.int64")):
        total += _group_index_row_count(path)
    return total


def _assert_group_index_row_count(*, group_index_dir: Path, expected_rows: int) -> int:
    """Assert packed group-index files cover every canonical panel row exactly once."""
    indexed_rows = _total_group_index_row_count(group_index_dir)
    if indexed_rows != expected_rows:
        raise ValueError(
            "Group-index row count mismatch: "
            f"indexed_rows={indexed_rows} expected_panel_rows={expected_rows} "
            f"group_index_dir={group_index_dir}"
        )
    return indexed_rows


def _group_index_path(scratch: ScratchPanel, *, date_code: int, interval_code: int) -> Path:
    group_id = int(date_code * len(scratch.unique_intervals) + interval_code)
    return scratch.group_index_dir / f"group_{group_id:08d}.int64"


def _expected_test_rows_by_fold(
    *,
    scratch: ScratchPanel,
    boundaries: list[FoldBoundary],
) -> dict[str, int]:
    expected: dict[str, int] = {}
    for boundary in boundaries:
        fold_rows = 0
        for date_code in range(boundary.test_start_code, boundary.test_end_code):
            for interval_code in range(len(scratch.unique_intervals)):
                group_path = _group_index_path(
                    scratch, date_code=date_code, interval_code=interval_code
                )
                if group_path.is_file():
                    fold_rows += _group_index_row_count(group_path)
        if fold_rows <= 0:
            raise ValueError(f"Fold {boundary.fold_id} has zero independently expected test rows")
        expected[boundary.fold_id] = fold_rows
    return expected


def _read_group_indices(
    *,
    scratch: ScratchPanel,
    date_code: int,
    interval_code: int,
    index_dtype: type[np.integer[Any]],
) -> np.ndarray:
    group_path = _group_index_path(scratch, date_code=date_code, interval_code=interval_code)
    if not group_path.is_file():
        return np.empty(0, dtype=index_dtype)
    return np.fromfile(group_path, dtype=np.int64).astype(index_dtype, copy=False)


def _append_fragment_to_writer(writer: Any, fragment_path: Path, *, chunk_rows: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(fragment_path)
    rows = 0
    schema = _prediction_schema()
    for batch in parquet.iter_batches(batch_size=chunk_rows):
        table = pa.Table.from_batches([batch]).cast(schema)
        writer.write_table(table)
        rows += int(table.num_rows)
    return rows


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return int(metadata.num_rows)


def _low_memory_model_fold_worker(
    *,
    config: P2Config,
    scratch: ScratchPanel,
    entry: dict[str, Any],
    features: list[str],
    target_column: str,
    boundary: FoldBoundary,
    run_id: str,
    fragment_path: Path,
    status_path: Path,
    expected_rows: int,
) -> None:
    import pyarrow.parquet as pq

    writer: Any | None = None
    tmp_fragment = fragment_path.parent / f".{fragment_path.name}.{uuid4().hex}.tmp"
    family = str(entry["family"])
    peak_rss = _process_tree_rss_bytes()
    limit = int(config.panel_train_peak_rss_limit_bytes)
    fold_started = time.perf_counter()
    try:
        x_all = np.memmap(
            scratch.feature_path,
            dtype=np.float32,
            mode="r",
            shape=(scratch.n_rows, scratch.n_features),
        )
        y_all = np.memmap(scratch.target_path, dtype=np.float32, mode="r", shape=(scratch.n_rows,))
        instrument_codes = np.memmap(
            scratch.instrument_code_path, dtype=np.int32, mode="r", shape=(scratch.n_rows,)
        )
        finite_target = np.memmap(
            scratch.finite_target_path, dtype=np.bool_, mode="r", shape=(scratch.n_rows,)
        )
        index_dtype = _index_dtype(scratch.n_rows)
        train_indices = _finite_train_indices_from_groups(
            scratch=scratch,
            finite_target=finite_target,
            instrument_codes=instrument_codes,
            boundary=boundary,
            index_dtype=index_dtype,
        )
        row_limit = resolve_train_row_limit(family=family, config=config)
        if row_limit > 0 and len(train_indices) > row_limit:
            train_indices = subsample_train_indices(
                train_indices,
                max_rows=row_limit,
                master_seed=config.random_seed,
                family=family,
                fold_id=boundary.fold_id,
            ).astype(index_dtype, copy=False)
        if len(train_indices) == 0:
            raise ValueError(
                f"Fold {boundary.fold_id} has zero eligible training rows for {family}"
            )
        actual_train_rows = int(len(train_indices))
        model = _fit_model(
            family=family,
            entry=entry,
            features=features,
            config=config,
            train_rows=actual_train_rows,
            fold_id=boundary.fold_id,
        )
        x_train, y_train, train_path = _compact_training_matrix(
            x_all=x_all,
            y_all=y_all,
            train_indices=train_indices,
            scratch_dir=scratch.scratch_dir,
            family=family,
            fold_id=boundary.fold_id,
            chunk_rows=int(config.panel_train_chunk_rows),
        )
        with PeakRssMonitor() as fit_monitor:
            model.fit(x_train, y_train, fold_id=boundary.fold_id)
        peak_rss = max(int(peak_rss or 0), int(fit_monitor.peak_bytes or 0))
        elastic_net_fit = (
            extract_elastic_net_fit_diagnostics(model) if family == "elastic_net" else None
        )
        _enforce_peak_rss_limit(
            fit_monitor.peak_bytes, limit, context=f"fit {family}/{boundary.fold_id}"
        )
        if train_path is not None:
            del x_train
            try:
                train_path.unlink(missing_ok=True)
            except OSError:
                LOG.warning("model_matrix_train_memmap_cleanup_failed", path=str(train_path))
        del y_train, train_indices
        gc.collect()

        fragment_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(tmp_fragment, _prediction_schema(), compression="snappy")
        total_rows = 0
        stats = {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}
        for date_code in range(boundary.test_start_code, boundary.test_end_code):
            date_label = scratch.unique_dates[date_code]
            for interval_code, interval_label in enumerate(scratch.unique_intervals):
                group_indices = _read_group_indices(
                    scratch=scratch,
                    date_code=date_code,
                    interval_code=interval_code,
                    index_dtype=index_dtype,
                )
                if len(group_indices) == 0:
                    continue
                order = np.argsort(instrument_codes[group_indices], kind="mergesort")
                group_indices = group_indices[order]
                x_chunk = np.asarray(x_all[group_indices, :], dtype=np.float32)
                with PeakRssMonitor() as predict_monitor:
                    if hasattr(model, "predict_with_confidence"):
                        preds, conf = model.predict_with_confidence(x_chunk)
                    else:
                        preds = model.predict(x_chunk).reshape(-1)
                        conf = model.predict_confidence(x_chunk).reshape(-1)
                peak_rss = max(int(peak_rss or 0), int(predict_monitor.peak_bytes or 0))
                _enforce_peak_rss_limit(
                    predict_monitor.peak_bytes,
                    limit,
                    context=f"predict {family}/{boundary.fold_id}",
                )
                if len(preds) != len(group_indices) or len(conf) != len(group_indices):
                    raise ValueError("Prediction/confidence length mismatch")
                if not np.isfinite(preds).all() or not np.isfinite(conf).all():
                    raise ValueError("Model emitted non-finite prediction or confidence")
                instruments = np.asarray(scratch.unique_instruments, dtype=object)[
                    np.asarray(instrument_codes[group_indices], dtype=np.intp)
                ]
                frame = pd.DataFrame(
                    {
                        "run_id": run_id,
                        "model_id": family,
                        "model_family": family,
                        "instrument": instruments,
                        "date": date_label,
                        "interval": interval_label,
                        "fold_id": boundary.fold_id,
                        "split": boundary.split,
                        "prediction": preds.astype(float),
                        "confidence": conf.astype(float),
                        "target_name": target_column,
                    }
                )
                frame = rank_prediction_frame(frame)
                if frame.duplicated(
                    ["model_id", "fold_id", "instrument", "date", "interval"]
                ).any():
                    raise ValueError("Duplicate prediction output keys in fragment")
                _append_prediction_chunk(writer, frame)
                total_rows += int(len(frame))
                stats["count"] += float(len(preds))
                stats["sum"] += float(np.sum(preds, dtype=np.float64))
                stats["sum_sq"] += float(np.sum(preds.astype(np.float64) ** 2))
                del x_chunk, preds, conf, frame
        writer.close()
        writer = None
        if total_rows != expected_rows:
            raise ValueError(
                f"Worker coverage mismatch for {family}/{boundary.fold_id}: "
                f"wrote {total_rows}, expected {expected_rows}"
            )
        os.replace(tmp_fragment, fragment_path)
        status = {
            "ok": True,
            "model_family": family,
            "fold_id": boundary.fold_id,
            "fragment_path": str(fragment_path),
            "fragment_rows": total_rows,
            "expected_rows": expected_rows,
            "dispersion": stats,
            "model_metrics": {
                "model_family": family,
                "implementation_id": _model_implementation_id(family, entry.get("params")),
                "resolved_backend": str(
                    (entry.get("params") or {}).get("backend", "linear_program")
                )
                if family == "quantile_regression"
                else None,
                "fold_id": boundary.fold_id,
                "train_row_limit": row_limit or None,
                "actual_train_rows": actual_train_rows,
                "test_rows": total_rows,
                "elapsed_s": round(time.perf_counter() - fold_started, 2),
                "process_tree_peak_rss_bytes": peak_rss,
                "elastic_net_fit_diagnostics": elastic_net_fit,
            },
        }
        atomic_write_json(status_path, _json_ready(status))
        del x_all, y_all, instrument_codes, finite_target, model
        gc.collect()
    except BaseException as err:
        if writer is not None:
            writer.close()
        if tmp_fragment.exists():
            tmp_fragment.unlink()
        if fragment_path.exists():
            fragment_path.unlink()
        atomic_write_json(
            status_path,
            {
                "ok": False,
                "model_family": family,
                "fold_id": boundary.fold_id,
                "error": f"{type(err).__name__}: {err}",
            },
        )
        raise


def _terminate_worker_process(process: Any) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None:
        try:
            import psutil

            root = psutil.Process(int(pid))
            children = root.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                root.terminate()
            _, alive = psutil.wait_procs([*children, root], timeout=5.0)
            for proc in alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (ImportError, OSError, ValueError):
            if process.is_alive():
                process.terminate()
    elif process.is_alive():
        process.terminate()

    process.join(timeout=5.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=5.0)


def _poll_worker_until_exit(
    process: Any,
    *,
    limit_bytes: int,
    timeout_s: int,
    context: str,
    rss_sampler: Callable[[], int | None] = _process_tree_rss_bytes,
    poll_interval_s: float = 0.10,
) -> int | None:
    started = time.monotonic()
    peak_bytes: int | None = rss_sampler()
    while process.is_alive():
        rss = rss_sampler()
        if rss is not None:
            peak_bytes = max(int(peak_bytes or 0), int(rss))
            if limit_bytes > 0 and rss > limit_bytes:
                _terminate_worker_process(process)
                raise MemoryError(
                    f"model_matrix {context} active process-tree RSS {rss} "
                    f"exceeded configured limit {limit_bytes}; worker terminated"
                )
        if timeout_s > 0 and time.monotonic() - started > timeout_s:
            _terminate_worker_process(process)
            raise TimeoutError(
                f"model_matrix {context} exceeded timeout {timeout_s}s; worker terminated"
            )
        process.join(timeout=poll_interval_s)

    rss = rss_sampler()
    if rss is not None:
        peak_bytes = max(int(peak_bytes or 0), int(rss))
    _enforce_peak_rss_limit(peak_bytes, limit_bytes, context=context)
    return peak_bytes


def _run_low_memory_worker(
    *,
    config: P2Config,
    scratch: ScratchPanel,
    entry: dict[str, Any],
    features: list[str],
    target_column: str,
    boundary: FoldBoundary,
    run_id: str,
    fragment_path: Path,
    status_path: Path,
    expected_rows: int,
) -> dict[str, object]:
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_low_memory_model_fold_worker,
        kwargs={
            "config": config,
            "scratch": scratch,
            "entry": entry,
            "features": features,
            "target_column": target_column,
            "boundary": boundary,
            "run_id": run_id,
            "fragment_path": fragment_path,
            "status_path": status_path,
            "expected_rows": expected_rows,
        },
    )
    limit = int(config.panel_train_peak_rss_limit_bytes)
    process.start()
    worker_peak = _poll_worker_until_exit(
        process,
        limit_bytes=limit,
        timeout_s=int(config.panel_train_worker_timeout_seconds),
        context=f"worker {entry['family']}/{boundary.fold_id}",
    )
    status: dict[str, object] = {}
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    if process.exitcode != 0 or not status.get("ok"):
        error = status.get("error", f"worker exit code {process.exitcode}")
        raise RuntimeError(
            f"Low-memory worker failed for {entry['family']}/{boundary.fold_id}: {error}"
        )
    status["parent_observed_peak_rss_bytes"] = worker_peak
    return status


def _fit_model(
    *,
    family: str,
    entry: dict[str, Any],
    features: list[str],
    config: P2Config,
    train_rows: int,
    fold_id: str,
) -> Any:
    fold_n_jobs = resolve_matrix_sklearn_n_jobs(config, family=family, train_rows=train_rows)
    sequence_length = _sequence_length_from_entry(entry)
    if sequence_length is not None:
        from pysrc.models.lstm import create_lstm_panel_model

        model = create_lstm_panel_model(
            model_id=_sequence_model_id(family, sequence_length),
            sequence_length=sequence_length,
            params=entry.get("params"),
            random_seed=config.random_seed,
        )
    else:
        model = create_panel_model(
            family,
            model_id=family,
            params=entry.get("params"),
            random_seed=config.random_seed,
            sklearn_n_jobs=fold_n_jobs,
        )
    if hasattr(model, "set_feature_names"):
        model.set_feature_names(features)
    return model


def _validate_quantile_backend_policy(yaml_models: list[dict[str, Any]], config: P2Config) -> None:
    for entry in yaml_models:
        family = str(entry.get("family", ""))
        if family != "quantile_regression":
            continue
        params = entry.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        backend = str(params.get("backend", "linear_program"))
        row_limit = resolve_train_row_limit(family=family, config=config)
        if row_limit == 0 and backend != "hist_gradient_boosting":
            raise ValueError(
                "Uncapped quantile_regression requires params.backend='hist_gradient_boosting'; "
                "linear-program quantile regression is not allowed with --max-train-rows 0"
            )


def _model_implementation_id(family: str, params: dict[str, object] | None) -> str:
    params = params or {}
    if (
        family == "quantile_regression"
        and str(params.get("backend", "linear_program")) == "hist_gradient_boosting"
    ):
        return "hist_gradient_boosting_quantile"
    return family


def _panel_source_chronological_order(panel: pd.DataFrame) -> bool:
    dates = normalize_date_labels(panel["date"])
    if len(dates) <= 1:
        return True
    return bool(dates.is_monotonic_increasing)


def _collect_elastic_net_worker_diagnostics(
    worker_statuses: list[dict[str, object]],
) -> dict[str, object] | None:
    for status in reversed(worker_statuses):
        metrics = status.get("model_metrics")
        if not isinstance(metrics, dict):
            continue
        diagnostics = metrics.get("elastic_net_fit_diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
    return None


def _elastic_net_yaml_params(yaml_models: list[dict[str, Any]]) -> dict[str, object] | None:
    for entry in yaml_models:
        if str(entry.get("family")) == "elastic_net":
            params = entry.get("params") or {}
            if isinstance(params, dict):
                return dict(params)
    return None


def _emit_model_matrix_validation(
    *,
    config: P2Config,
    source: CanonicalPanelSource,
    run_path: Path,
    target_column: str,
    target_metadata: dict[str, object],
    unique_dates: tuple[str, ...] | list[str],
    boundaries: list[FoldBoundary],
    source_chronological_order: bool,
    fold_policy: dict[str, object],
    predictions: pd.DataFrame | None,
    panel: pd.DataFrame | None,
    yaml_models: list[dict[str, Any]],
    elastic_net_fit_diagnostics: dict[str, object] | None,
    diversity_report: dict[str, object],
    prediction_sanity: dict[str, object] | None = None,
) -> Path:
    validation = build_model_matrix_validation_bundle(
        source=source,
        config=config,
        target_column=target_column,
        target_metadata=target_metadata,
        unique_dates=unique_dates,
        boundaries=boundaries,
        source_chronological_order=source_chronological_order,
        fold_policy=fold_policy,
        predictions=predictions,
        panel=panel,
        yaml_models=yaml_models,
        run_id=run_path.name,
        elastic_net_params=_elastic_net_yaml_params(yaml_models),
        elastic_net_fit_diagnostics=elastic_net_fit_diagnostics,
        prediction_sanity=prediction_sanity,
        diversity_report=diversity_report,
    )
    validation_path = run_path / "reports" / "model_matrix_validation.json"
    atomic_write_json(validation_path, _json_ready(validation))
    return validation_path


def _diversity_report_summary(diversity: dict[str, object]) -> dict[str, object]:
    return {
        "low_diversity_warning": diversity.get("low_diversity_warning"),
        "redundant_pairs": len(diversity.get("redundant_pairs", [])),
        "nonredundant_child_count": diversity.get("nonredundant_child_count"),
        "active_prediction_child_count": diversity.get("active_prediction_child_count"),
        "positive_all_folds_count": diversity.get("positive_all_folds_count"),
        "positive_any_fold_count": diversity.get("positive_any_fold_count"),
        "eligible_router_child_count": diversity.get("eligible_router_child_count"),
        "diagnostic_coverage_satisfied": (diversity.get("diagnostic_coverage") or {}).get(
            "coverage_satisfied"
        ),
    }


def _train_model_matrix_in_memory(
    *,
    config: P2Config,
    source: CanonicalPanelSource,
    run_path: Path,
    yaml_models: list[dict[str, Any]],
    memory_decision: MemoryModeDecision,
    row_policy: TrainRowPolicy,
) -> dict[str, Path]:
    panel = source.load_pandas()
    if "instrument" not in panel.columns and "symbol" in panel.columns:
        panel = panel.rename(columns={"symbol": "instrument"})
    panel = _filter_panel_instruments(panel, config)
    if "interval" not in panel.columns:
        panel = panel.copy()
        panel["interval"] = "1d"
    manifest = _resolve_source_manifest(source)
    target_column, features, feature_report, target_metadata = resolve_schema_target_and_features(
        config,
        _schema_from_frame(panel),
        manifest=manifest,
    )
    if not features:
        raise ValueError("No numeric feature columns found for model matrix training")
    audit = source.audit or validate_canonical_panel(
        panel, panel_path=source.panel_path, smoke_test=config.smoke_test
    )
    if not audit.grain_valid or audit.duplicate_key_count > 0:
        raise ValueError(
            f"Canonical panel grain invalid: duplicate_key_count={audit.duplicate_key_count}"
        )
    audit_dict = audit.as_dict()
    horizon = int(target_metadata.get("horizon_days", 0))
    normalized_dates = normalize_date_labels(panel["date"])
    date_codes_arr, unique_dates = build_chronological_date_codes(normalized_dates.to_numpy())
    folds = build_walk_forward_boundaries(
        unique_dates,
        n_folds=_resolve_walk_forward_fold_count(config),
        target_horizon_days=horizon,
    )
    prediction_parts: list[pd.DataFrame] = []
    index_dtype = _index_dtype(len(panel))
    x_all = panel[features].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(x_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = panel[target_column].to_numpy(dtype=np.float32, copy=False)
    dates_arr = normalized_dates.to_numpy(dtype=object)
    instruments_arr = panel["instrument"].astype(str).to_numpy()
    interval_arr = panel["interval"].astype(str).to_numpy()
    finite_target = np.isfinite(y_all)

    LOG.info(
        "model_matrix_start",
        run_id=run_path.name,
        n_models=len(yaml_models),
        n_folds=len(folds),
        panel_rows=len(panel),
        feature_count=len(features),
        memory_mode=memory_decision.resolved_memory_mode,
        max_train_rows_per_fold=row_policy.general_max_rows,
        quantile_max_train_rows=row_policy.quantile_max_rows,
    )
    model_metrics: list[dict[str, object]] = []
    elastic_net_fit_diagnostics: dict[str, object] | None = None
    for _model_idx, entry in enumerate(yaml_models, start=1):
        family = str(entry["family"])
        sequence_length = _sequence_length_from_entry(entry)
        if sequence_length is not None:
            xs_all, ys_all, meta_all = build_sequence_windows(
                panel,
                features,
                target_column,
                sequence_length=sequence_length,
            )
            model_id = _sequence_model_id(family, sequence_length)
            for boundary in folds:
                fold_started = time.perf_counter()
                train_mask, test_mask = _sequence_fold_masks(meta_all, boundary, list(unique_dates))
                train_mask &= np.isfinite(ys_all)
                test_mask &= np.isfinite(ys_all)
                train_indices = np.flatnonzero(train_mask).astype(index_dtype, copy=False)
                test_indices = np.flatnonzero(test_mask).astype(index_dtype, copy=False)
                row_limit = resolve_train_row_limit(family=family, config=config)
                if row_limit > 0 and len(train_indices) > row_limit:
                    train_indices = subsample_train_indices(
                        train_indices,
                        max_rows=row_limit,
                        master_seed=config.random_seed,
                        family=family,
                        fold_id=boundary.fold_id,
                    ).astype(index_dtype, copy=False)
                if len(train_indices) == 0 or len(test_indices) == 0:
                    continue
                model = _fit_model(
                    family=family,
                    entry=entry,
                    features=features,
                    config=config,
                    train_rows=int(len(train_indices)),
                    fold_id=boundary.fold_id,
                )
                x_train = np.asarray(xs_all[train_indices], dtype=np.float32)
                y_train = np.asarray(ys_all[train_indices], dtype=np.float32)
                x_test = np.asarray(xs_all[test_indices], dtype=np.float32)
                model.fit(x_train, y_train, fold_id=boundary.fold_id)
                if hasattr(model, "predict_with_confidence"):
                    preds, conf = model.predict_with_confidence(x_test)
                else:
                    preds = model.predict(x_test)
                    conf = model.predict_confidence(x_test)
                if len(preds.reshape(-1)) != len(test_indices) or len(conf.reshape(-1)) != len(
                    test_indices
                ):
                    raise ValueError("Prediction/confidence length mismatch")
                meta_test = meta_all.iloc[test_indices].reset_index(drop=True)
                panel_keys = panel[["date", "instrument", "interval"]].drop_duplicates()
                panel_keys = panel_keys.assign(
                    date=normalize_date_labels(panel_keys["date"]).astype(str),
                    instrument=panel_keys["instrument"].astype(str),
                )
                meta_join = meta_test.assign(
                    date=meta_test["date"].astype(str),
                    instrument=meta_test["instrument"].astype(str),
                ).merge(panel_keys, on=["date", "instrument"], how="left")
                test_frame = pd.DataFrame(
                    {
                        "date": meta_join["date"].to_numpy(),
                        "instrument": meta_join["instrument"].to_numpy(),
                        "interval": meta_join["interval"].fillna("1d").astype(str).to_numpy(),
                        "model_id": model_id,
                        "model_family": family,
                        "fold_id": boundary.fold_id,
                        "split": boundary.split,
                        "prediction": preds.reshape(-1),
                        "confidence": conf.reshape(-1),
                        "target_name": target_column,
                        "run_id": run_path.name,
                    }
                )
                prediction_parts.append(rank_prediction_frame(test_frame))
                model_metrics.append(
                    {
                        "model_family": family,
                        "implementation_id": _model_implementation_id(family, entry.get("params")),
                        "fold_id": boundary.fold_id,
                        "train_row_limit": row_limit or None,
                        "actual_train_rows": int(len(train_indices)),
                        "test_rows": int(len(test_indices)),
                        "sequence_length": sequence_length,
                        "elapsed_s": round(time.perf_counter() - fold_started, 2),
                    }
                )
                del model, x_train, y_train, x_test, preds, conf, test_frame, meta_test
                gc.collect()
            continue
        for boundary in folds:
            fold_started = time.perf_counter()
            train_indices = _finite_train_indices(
                date_codes=date_codes_arr,
                finite_target=finite_target,
                boundary=boundary,
                index_dtype=index_dtype,
            )
            row_limit = resolve_train_row_limit(family=family, config=config)
            if row_limit > 0 and len(train_indices) > row_limit:
                train_indices = subsample_train_indices(
                    train_indices,
                    max_rows=row_limit,
                    master_seed=config.random_seed,
                    family=family,
                    fold_id=boundary.fold_id,
                ).astype(index_dtype, copy=False)
            _, test_mask = fold_masks_from_boundaries(date_codes_arr, boundary)
            test_indices = np.flatnonzero(test_mask).astype(index_dtype, copy=False)
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            model = _fit_model(
                family=family,
                entry=entry,
                features=features,
                config=config,
                train_rows=int(len(train_indices)),
                fold_id=boundary.fold_id,
            )
            x_train = np.asarray(x_all[train_indices], dtype=np.float32)
            y_train = np.asarray(y_all[train_indices], dtype=np.float32)
            x_test = np.asarray(x_all[test_indices], dtype=np.float32)
            model.fit(x_train, y_train, fold_id=boundary.fold_id)
            if family == "elastic_net":
                elastic_net_fit_diagnostics = extract_elastic_net_fit_diagnostics(model)
            if hasattr(model, "predict_with_confidence"):
                preds, conf = model.predict_with_confidence(x_test)
            else:
                preds = model.predict(x_test)
                conf = model.predict_confidence(x_test)
            if len(preds.reshape(-1)) != len(test_indices) or len(conf.reshape(-1)) != len(
                test_indices
            ):
                raise ValueError("Prediction/confidence length mismatch")
            test_frame = pd.DataFrame(
                {
                    "date": dates_arr[test_indices],
                    "instrument": instruments_arr[test_indices],
                    "interval": interval_arr[test_indices],
                    "model_id": family,
                    "model_family": family,
                    "fold_id": boundary.fold_id,
                    "split": boundary.split,
                    "prediction": preds.reshape(-1),
                    "confidence": conf.reshape(-1),
                    "target_name": target_column,
                    "run_id": run_path.name,
                }
            )
            prediction_parts.append(rank_prediction_frame(test_frame))
            model_metrics.append(
                {
                    "model_family": family,
                    "implementation_id": _model_implementation_id(family, entry.get("params")),
                    "fold_id": boundary.fold_id,
                    "train_row_limit": row_limit or None,
                    "actual_train_rows": int(len(train_indices)),
                    "test_rows": int(len(test_indices)),
                    "elapsed_s": round(time.perf_counter() - fold_started, 2),
                }
            )
            del model, x_train, y_train, x_test, preds, conf, test_frame
            gc.collect()

    if not prediction_parts:
        raise ValueError("Model matrix training produced zero prediction rows")
    predictions = _ordered_prediction_frame(pd.concat(prediction_parts, ignore_index=True))
    pred_path = run_path / "predictions" / "model_prediction_panel.parquet"
    assert_canonical_source_unchanged(source)
    _write_prediction_fragment(predictions, pred_path)

    report = {
        "schema_version": "model_prediction_panel.v1",
        "run_id": run_path.name,
        "experiment_id": config.experiment_id,
        "target_column": target_column,
        "target_metadata": target_metadata,
        "n_models": len(yaml_models),
        "n_rows": int(len(predictions)),
        "feature_count": len(features),
        "feature_usage": feature_report,
        "canonical_data": audit_dict,
        "canonical_source": source.as_dict(),
        "memory_mode": memory_decision.as_dict(),
        "train_row_policy": row_policy.as_dict(),
        "fold_policy": _fold_report(folds, horizon),
        "model_folds": model_metrics,
        "source_chronological_order": _panel_source_chronological_order(panel),
        "interval_count": int(predictions["interval"].nunique()),
        "process_tree_peak_rss_bytes": _process_tree_rss_bytes(),
        "prediction_schema_validated": True,
    }
    report_path = run_path / "reports" / "model_matrix_report.json"
    atomic_write_json(report_path, _json_ready(report))
    diversity = build_model_diversity_report(predictions, panel, target_column)
    diversity["diagnostic_coverage"] = build_diagnostic_coverage_report(
        predictions,
        expected_model_count=len(yaml_models),
        expected_fold_count=len(folds),
    )
    diversity_path = run_path / "reports" / "model_diversity_report.json"
    atomic_write_json(diversity_path, _json_ready(diversity))
    validation_path = _emit_model_matrix_validation(
        config=config,
        source=source,
        run_path=run_path,
        target_column=target_column,
        target_metadata=target_metadata,
        unique_dates=list(unique_dates),
        boundaries=folds,
        source_chronological_order=_panel_source_chronological_order(panel),
        fold_policy=report["fold_policy"],
        predictions=predictions,
        panel=panel,
        yaml_models=yaml_models,
        elastic_net_fit_diagnostics=elastic_net_fit_diagnostics,
        diversity_report=diversity,
    )
    report["diversity"] = _diversity_report_summary(diversity)
    report["validation_report"] = str(validation_path)
    atomic_write_json(report_path, _json_ready(report))
    LOG.info("model_matrix_complete", run_id=run_path.name, rows=len(predictions))
    return {
        "run_dir": run_path,
        "model_prediction_panel": pred_path,
        "report": report_path,
        "model_diversity_report": diversity_path,
        "model_matrix_validation": validation_path,
    }


def _train_model_matrix_low_memory(
    *,
    config: P2Config,
    source: CanonicalPanelSource,
    run_path: Path,
    yaml_models: list[dict[str, Any]],
    memory_decision: MemoryModeDecision,
    row_policy: TrainRowPolicy,
) -> dict[str, Path]:
    import pyarrow.parquet as pq

    capabilities = _validate_low_memory_capabilities(yaml_models)
    manifest = _resolve_source_manifest(source)
    target_column, features, feature_report, target_metadata = resolve_schema_target_and_features(
        config,
        source.schema,
        manifest=manifest,
    )
    horizon = int(target_metadata.get("horizon_days", 0))
    audit = source.audit
    if audit is None and source.panel_path is not None:
        sample = (
            pq.read_table(
                source.panel_path,
                columns=["date", "instrument", "interval"],
            )
            .to_pandas()
            .head(50_000)
        )
        audit = validate_canonical_panel(sample, panel_path=source.panel_path, smoke_test=False)
    if audit is None and source.smoke_panel is not None:
        audit = validate_canonical_panel(source.smoke_panel, smoke_test=config.smoke_test)
    if audit is None:
        raise ValueError("Canonical audit unavailable for model-matrix training")
    if not audit.grain_valid or audit.duplicate_key_count > 0:
        raise ValueError(
            f"Canonical panel grain invalid: duplicate_key_count={audit.duplicate_key_count}"
        )
    scratch: ScratchPanel | None = None
    pred_path = run_path / "predictions" / "model_prediction_panel.parquet"
    tmp_pred_path = pred_path.parent / f".{pred_path.name}.{uuid4().hex}.tmp"
    fragment_dir = run_path / "predictions" / "fragments"
    status_dir = run_path / "reports" / "worker_status"
    writer: Any | None = None
    failed = True
    peak_rss = _process_tree_rss_bytes()
    try:
        scratch = _prepare_scratch_panel(
            source=source,
            config=config,
            run_path=run_path,
            features=features,
            target_column=target_column,
            model_count=len(yaml_models),
        )
        boundaries = build_walk_forward_boundaries(
            np.asarray(scratch.unique_dates, dtype=object),
            n_folds=_resolve_walk_forward_fold_count(config),
            target_horizon_days=horizon,
        )
        if not boundaries:
            raise ValueError("No walk-forward folds available for low-memory train-matrix")
        expected_rows_by_fold = _expected_test_rows_by_fold(scratch=scratch, boundaries=boundaries)
        expected_output_rows = int(sum(expected_rows_by_fold.values()) * len(yaml_models))
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        fragment_dir.mkdir(parents=True, exist_ok=True)
        status_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(
            "model_matrix_start",
            run_id=run_path.name,
            n_models=len(yaml_models),
            n_folds=len(boundaries),
            panel_rows=scratch.n_rows,
            feature_count=len(features),
            memory_mode=memory_decision.resolved_memory_mode,
            capabilities=capabilities,
            max_train_rows_per_fold=row_policy.general_max_rows,
            quantile_max_train_rows=row_policy.quantile_max_rows,
            expected_prediction_rows=expected_output_rows,
        )

        model_metrics: list[dict[str, object]] = []
        dispersion: dict[str, dict[str, float]] = {}
        completed_model_folds: set[tuple[str, str]] = set()
        fragment_paths: list[Path] = []
        worker_statuses: list[dict[str, object]] = []
        for model_idx, entry in enumerate(yaml_models):
            family = str(entry["family"])
            stats = dispersion.setdefault(family, {"count": 0.0, "sum": 0.0, "sum_sq": 0.0})
            safe_family = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in family)
            for fold_idx, boundary in enumerate(boundaries):
                fragment_path = (
                    fragment_dir / f"{model_idx:03d}_{fold_idx:03d}_{safe_family}.parquet"
                )
                status_path = status_dir / f"{model_idx:03d}_{fold_idx:03d}_{safe_family}.json"
                status = _run_low_memory_worker(
                    config=config,
                    scratch=scratch,
                    entry=entry,
                    features=features,
                    target_column=target_column,
                    boundary=boundary,
                    run_id=run_path.name,
                    fragment_path=fragment_path,
                    status_path=status_path,
                    expected_rows=expected_rows_by_fold[boundary.fold_id],
                )
                fragment_rows = _parquet_row_count(fragment_path)
                if fragment_rows != expected_rows_by_fold[boundary.fold_id]:
                    raise ValueError(
                        f"Fragment row-count mismatch for {family}/{boundary.fold_id}: "
                        f"{fragment_rows} != {expected_rows_by_fold[boundary.fold_id]}"
                    )
                worker_peak = status.get("parent_observed_peak_rss_bytes")
                if worker_peak is not None:
                    peak_rss = max(int(peak_rss or 0), int(worker_peak))
                completed_model_folds.add((family, boundary.fold_id))
                fragment_paths.append(fragment_path)
                worker_statuses.append(status)
                model_metric = status.get("model_metrics")
                if isinstance(model_metric, dict):
                    model_metrics.append(model_metric)
                fold_dispersion = status.get("dispersion")
                if isinstance(fold_dispersion, dict):
                    stats["count"] += float(fold_dispersion.get("count", 0.0))
                    stats["sum"] += float(fold_dispersion.get("sum", 0.0))
                    stats["sum_sq"] += float(fold_dispersion.get("sum_sq", 0.0))

        expected_pairs = {
            (str(entry["family"]), boundary.fold_id)
            for entry in yaml_models
            for boundary in boundaries
        }
        missing_pairs = sorted(expected_pairs - completed_model_folds)
        if missing_pairs:
            raise ValueError(f"Missing model/fold prediction fragments: {missing_pairs}")

        writer = pq.ParquetWriter(tmp_pred_path, _prediction_schema(), compression="snappy")
        total_rows = 0
        for fragment_path in fragment_paths:
            total_rows += _append_fragment_to_writer(
                writer,
                fragment_path,
                chunk_rows=int(config.panel_train_chunk_rows),
            )
        writer.close()
        writer = None
        if total_rows == 0:
            raise ValueError("Model matrix training produced zero prediction rows")
        if total_rows != expected_output_rows:
            raise ValueError(
                f"Prediction coverage mismatch: wrote {total_rows}, expected {expected_output_rows}"
            )
        assert_canonical_source_unchanged(source)
        os.replace(tmp_pred_path, pred_path)
        fragment_bytes = sum(path.stat().st_size for path in fragment_paths if path.is_file())
        canonical_audit = _canonical_audit_from_scratch(source, scratch)
        report = {
            "schema_version": "model_prediction_panel.v1",
            "run_id": run_path.name,
            "experiment_id": config.experiment_id,
            "target_column": target_column,
            "target_metadata": target_metadata,
            "n_models": len(yaml_models),
            "n_rows": total_rows,
            "expected_prediction_rows": expected_output_rows,
            "feature_count": len(features),
            "feature_usage": feature_report,
            "canonical_data": canonical_audit.as_dict(),
            "canonical_source": source.as_dict(),
            "memory_mode": memory_decision.as_dict(),
            "train_row_policy": row_policy.as_dict(),
            "fold_policy": _fold_report(boundaries, horizon),
            "model_capabilities": capabilities,
            "model_folds": model_metrics,
            "worker_process_isolation": True,
            "worker_status_count": len(worker_statuses),
            "interval_count": len(scratch.unique_intervals),
            "process_tree_peak_rss_bytes": peak_rss,
            "scratch_bytes": scratch.scratch_bytes,
            "group_index_row_count": scratch.group_index_row_count,
            "fragment_bytes": fragment_bytes,
            "source_chronological_order": scratch.chronological_source_order,
            "prediction_schema_validated": True,
        }
        report_path = run_path / "reports" / "model_matrix_report.json"
        atomic_write_json(report_path, _json_ready(report))
        diversity = build_low_memory_model_diversity_report(
            pred_path,
            target_column=target_column,
            n_rows=scratch.n_rows,
            unique_dates=scratch.unique_dates,
            unique_instruments=scratch.unique_instruments,
            unique_intervals=scratch.unique_intervals,
            date_code_path=scratch.date_code_path,
            instrument_code_path=scratch.instrument_code_path,
            interval_code_path=scratch.interval_code_path,
            target_path=scratch.target_path,
            random_seed=int(config.random_seed),
            expected_model_count=len(yaml_models),
            expected_fold_count=len(boundaries),
        )
        diversity_path = run_path / "reports" / "model_diversity_report.json"
        atomic_write_json(diversity_path, _json_ready(diversity))
        elastic_net_fit_diagnostics = _collect_elastic_net_worker_diagnostics(worker_statuses)
        if source.panel_path is None:
            raise ValueError("Canonical panel path required for streaming prediction sanity")
        target_lookup = build_panel_target_lookup(source.panel_path, target_column=target_column)
        prediction_sanity = build_streaming_prediction_sanity_report(
            pred_path,
            target_column=target_column,
            target_lookup=target_lookup,
            random_seed=int(config.random_seed),
        )
        validation_path = _emit_model_matrix_validation(
            config=config,
            source=source,
            run_path=run_path,
            target_column=target_column,
            target_metadata=target_metadata,
            unique_dates=list(scratch.unique_dates),
            boundaries=boundaries,
            source_chronological_order=scratch.chronological_source_order,
            fold_policy=report["fold_policy"],
            predictions=None,
            panel=None,
            yaml_models=yaml_models,
            elastic_net_fit_diagnostics=elastic_net_fit_diagnostics,
            diversity_report=diversity,
            prediction_sanity=prediction_sanity,
        )
        report["diversity"] = _diversity_report_summary(diversity)
        report["validation_report"] = str(validation_path)
        atomic_write_json(report_path, _json_ready(report))
        shutil.rmtree(fragment_dir, ignore_errors=True)
        LOG.info("model_matrix_complete", run_id=run_path.name, rows=total_rows)
        gc.collect()
        failed = False
        return {
            "run_dir": run_path,
            "model_prediction_panel": pred_path,
            "report": report_path,
            "model_diversity_report": diversity_path,
            "model_matrix_validation": validation_path,
        }
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_pred_path.exists():
            tmp_pred_path.unlink()
        if pred_path.exists():
            pred_path.unlink()
        raise
    finally:
        preserve_scratch = failed and config.panel_preserve_scratch_on_failure
        if scratch is not None and not preserve_scratch:
            shutil.rmtree(scratch.scratch_dir, ignore_errors=True)
        preserve_fragments = failed and config.panel_preserve_fragments_on_failure
        if failed and not preserve_fragments and fragment_dir.exists():
            shutil.rmtree(fragment_dir, ignore_errors=True)


def train_model_matrix(
    config: P2Config,
    experiment: MetaRouterExperimentSpec,
    *,
    run_dir: Path | None = None,
) -> dict[str, Path]:
    """Train YAML model matrix and persist model_prediction_panel."""

    run_token = run_dir or allocate_run_dir(
        lane="panel_model",
        run_id=f"model_matrix_{uuid4().hex[:12]}",
        smoke=config.smoke_test,
    )
    run_path = Path(run_token) if not isinstance(run_token, Path) else run_token
    model_entries = (
        [{"family": m.family, "params": m.params} for m in experiment.models]
        if experiment.models
        else [{"family": f} for f in config.panel_model_families]
    )
    yaml_models = model_entries_from_yaml(model_entries)
    source = require_canonical_panel_source_for_real_run(config, smoke_test=config.smoke_test)
    manifest = _resolve_source_manifest(source)
    target_column, features, _feature_report, target_metadata = resolve_schema_target_and_features(
        config,
        source.schema,
        manifest=manifest,
    )
    del target_column
    int(target_metadata.get("horizon_days", 0))
    approx_unique_dates = max(
        2, min(source.row_count, int(source.manifest.get("trading_day_count", 0) or 4))
    )
    fold_size = max(1, approx_unique_dates // (3 + 1))
    largest_train_rows = max(1, int(source.row_count * 0.75))
    largest_test_rows = max(1, int(source.row_count * (fold_size / max(approx_unique_dates, 1))))
    memory_decision = resolve_panel_memory_mode(
        config,
        panel_rows=source.row_count,
        feature_count=len(features),
        model_count=len(yaml_models),
        largest_train_rows=largest_train_rows,
        largest_test_rows=largest_test_rows,
        available_memory_bytes=_available_memory_bytes(),
    )
    row_policy = resolve_train_row_policy(config)
    _validate_quantile_backend_policy(yaml_models, config)
    if memory_decision.resolved_memory_mode == "in_memory":
        return _train_model_matrix_in_memory(
            config=config,
            source=source,
            run_path=run_path,
            yaml_models=yaml_models,
            memory_decision=memory_decision,
            row_policy=row_policy,
        )
    return _train_model_matrix_low_memory(
        config=config,
        source=source,
        run_path=run_path,
        yaml_models=yaml_models,
        memory_decision=memory_decision,
        row_policy=row_policy,
    )


def train_model_matrix_from_yaml(
    config_path: Path | str,
    *,
    smoke_test: bool = False,
    canary: bool = False,
    random_seed: int = 42,
    processed_data_root: str | None = None,
    max_train_rows_per_fold: int | None = None,
    quantile_max_train_rows_per_fold: int | None = None,
    panel_walk_forward_folds: int | None = None,
    panel_max_instruments: int | None = None,
    run_dir: Path | None = None,
) -> dict[str, Path]:
    resolved = Path(config_path)
    yaml_dict = load_p2_config(resolved)
    experiment = parse_meta_router_experiment(yaml_dict)
    overrides: dict[str, Any] = {"smoke_test": smoke_test, "random_seed": random_seed}
    if canary:
        overrides["panel_walk_forward_folds"] = (
            panel_walk_forward_folds if panel_walk_forward_folds is not None else 1
        )
        overrides["panel_max_instruments"] = (
            panel_max_instruments if panel_max_instruments is not None else 20
        )
        overrides["panel_train_memory_mode"] = "in_memory"
    elif panel_walk_forward_folds is not None:
        overrides["panel_walk_forward_folds"] = panel_walk_forward_folds
    if panel_max_instruments is not None and not canary:
        overrides["panel_max_instruments"] = panel_max_instruments
    if processed_data_root:
        overrides["processed_data_root"] = processed_data_root
    if max_train_rows_per_fold is not None:
        if max_train_rows_per_fold < 0:
            raise ValueError("--max-train-rows must be non-negative")
        overrides["panel_train_max_rows_per_fold"] = max_train_rows_per_fold
        overrides["panel_quantile_max_train_rows"] = max_train_rows_per_fold
    if quantile_max_train_rows_per_fold is not None:
        if quantile_max_train_rows_per_fold < 0:
            raise ValueError("--quantile-max-train-rows must be non-negative")
        overrides["panel_quantile_max_train_rows"] = quantile_max_train_rows_per_fold
    from pysrc.pipeline.p2_config_loader import yaml_to_p2_config

    config = yaml_to_p2_config(
        yaml_dict,
        cli_overrides=overrides,
        config_path=resolved,
    )
    return train_model_matrix(config, experiment, run_dir=run_dir)


__all__ = [
    "FoldBoundary",
    "TrainRowPolicy",
    "build_chronological_date_codes",
    "build_walk_forward_boundaries",
    "build_walk_forward_folds",
    "build_walk_forward_folds_from_codes",
    "fold_masks_from_boundaries",
    "initialize_train_matrix_scratch_dir",
    "rank_prediction_frame",
    "resolve_matrix_sklearn_n_jobs",
    "resolve_panel_memory_mode",
    "resolve_schema_target_and_features",
    "resolve_train_matrix_scratch_dir",
    "resolve_train_row_limit",
    "resolve_train_row_policy",
    "subsample_train_indices",
    "train_model_matrix",
    "train_model_matrix_from_yaml",
]

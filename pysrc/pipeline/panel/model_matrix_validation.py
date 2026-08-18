"""Gate 2 model-matrix validation audits before MetaRouter handoff."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pysrc.pipeline.panel.panel_keys import normalize_key_columns
from pysrc.pipeline.panel.panel_targets import (
    CANONICAL_PANEL_KEYS,
    PREDICTION_OUTPUT_KEYS,
    is_approved_forward_target_column,
)
from pysrc.pipeline.stages.market_data.sources.sip_panel import SIP_ADJUSTED_PANEL_SOURCE

if TYPE_CHECKING:
    from pysrc.pipeline.canonical_data import CanonicalPanelSource
    from pysrc.pipeline.contracts.p2 import P2Config

_CONTEMPORANEOUS_TARGET_COLUMNS = frozenset({"adjusted_return_1d", "raw_return_1d"})
_FORWARD_TARGET_COLUMNS = frozenset({"forward_return", "forward_return_horizon"})
_NUMERICAL_SAMPLE_SIZE = 5_000
_NUMERICAL_TOLERANCE = 1e-5
_STREAMING_RESERVOIR_SIZE = 10_000


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _quantile_summary(values: np.ndarray) -> dict[str, float | None]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            "min": None,
            "p01": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "min": float(np.min(clean)),
        "p01": float(np.quantile(clean, 0.01)),
        "p25": float(np.quantile(clean, 0.25)),
        "p50": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "p99": float(np.quantile(clean, 0.99)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
    }


def _deterministic_sample_indices(
    row_count: int, *, random_seed: int, sample_size: int
) -> np.ndarray:
    if row_count <= 0:
        return np.array([], dtype=np.int64)
    size = min(sample_size, row_count)
    digest = hashlib.sha256(f"{random_seed}:gate2:numerical".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(rng.choice(row_count, size=size, replace=False).astype(np.int64))


_PRICE_SUBSTRATE_READ_COLUMNS: tuple[str, ...] = (
    "date",
    "Date",
    "timestamp",
    "bar_date",
    "instrument",
    "ticker",
    "Ticker",
    "symbol",
    "Symbol",
    "adj_close",
)


def _price_substrate_read_columns(schema_names: set[str]) -> list[str]:
    return [name for name in _PRICE_SUBSTRATE_READ_COLUMNS if name in schema_names]


def _normalize_price_substrate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Align SIP hive columns with canonical date/instrument names."""
    from pysrc.pipeline.stages.market_data.sources.sip_panel import (
        _normalize_adjusted_panel_columns,
    )

    normalized = _normalize_adjusted_panel_columns(frame)
    if "adj_close" not in normalized.columns:
        raise ValueError("Price substrate missing adj_close after column normalization")
    return normalized.loc[:, ["date", "instrument", "adj_close"]]


def _load_price_frame(price_path: Path, instruments: set[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    if price_path.is_dir():
        parts = sorted(price_path.glob("year=*/month=*/*.parquet"))
        if not parts:
            parts = sorted(price_path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files under price substrate: {price_path}")
        frames = []
        for part in parts:
            parquet = pq.ParquetFile(part)
            schema_names = set(parquet.schema_arrow.names)
            columns = _price_substrate_read_columns(schema_names)
            if "adj_close" not in schema_names or not columns:
                continue
            batch = parquet.read(columns=columns).to_pandas()
            batch = _normalize_price_substrate_frame(batch)
            batch = batch.loc[batch["instrument"].astype(str).isin(instruments)]
            if len(batch):
                frames.append(batch)
        if not frames:
            return pd.DataFrame(columns=["date", "instrument", "adj_close"])
        return pd.concat(frames, ignore_index=True)

    parquet = pq.ParquetFile(price_path)
    schema_names = set(parquet.schema_arrow.names)
    columns = _price_substrate_read_columns(schema_names)
    if "adj_close" not in schema_names:
        raise ValueError(f"Price substrate missing adj_close: {price_path}")
    frames = []
    for batch in parquet.iter_batches(columns=columns, batch_size=250_000):
        part = batch.to_pandas()
        part = _normalize_price_substrate_frame(part)
        part = part.loc[part["instrument"].astype(str).isin(instruments)]
        if len(part):
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["date", "instrument", "adj_close"])
    return pd.concat(frames, ignore_index=True)


def _expected_forward_return(
    prices: pd.DataFrame,
    *,
    instrument: str,
    feature_date: str,
    horizon_days: int,
) -> tuple[float | None, str | None]:
    part = prices.loc[prices["instrument"].astype(str) == instrument].copy()
    if part.empty:
        return None, "missing_instrument"
    part["date"] = pd.to_datetime(part["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    part = part.sort_values("date", kind="mergesort")
    dates = part["date"].tolist()
    if feature_date not in dates:
        return None, "missing_feature_date"
    idx = dates.index(feature_date)
    end_idx = idx + horizon_days
    if end_idx >= len(dates):
        return None, "missing_future_price"
    base = float(part.iloc[idx]["adj_close"])
    future = float(part.iloc[end_idx]["adj_close"])
    if not np.isfinite(base) or base == 0.0 or not np.isfinite(future):
        return None, "non_finite_price"
    return future / base - 1.0, None


def _mark_forward_audit_not_performed(
    audit: dict[str, object], *, reason: str
) -> dict[str, object]:
    audit["verification_status"] = "not_performed"
    audit["skipped_reason"] = reason
    audit["rows_checked"] = 0
    audit["mismatch_count"] = None
    audit["maximum_absolute_error"] = None
    audit["missing_future_price_count"] = None
    audit["incorrect_terminal_label_count"] = None
    audit["sample_mismatches"] = []
    return audit


def build_forward_target_numerical_audit(
    *,
    panel_path: Path | None,
    target_column: str,
    horizon_days: int,
    price_substrate_path: str | Path | None = None,
    random_seed: int = 42,
    sample_size: int = _NUMERICAL_SAMPLE_SIZE,
    smoke_panel: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Numerically verify forward targets against adjusted close prices."""
    audit: dict[str, object] = {
        "schema_version": "forward_target_numerical_audit.v1",
        "target_column": target_column,
        "horizon_days": horizon_days,
        "verification_status": "not_performed",
        "rows_checked": 0,
        "mismatch_count": None,
        "maximum_absolute_error": None,
        "missing_future_price_count": None,
        "incorrect_terminal_label_count": None,
        "sample_mismatches": [],
    }
    if not is_approved_forward_target_column(target_column):
        return _mark_forward_audit_not_performed(audit, reason="target_not_forward_approved")

    if smoke_panel is not None:
        frame = smoke_panel.copy()
        if "adj_close" not in frame.columns:
            return _mark_forward_audit_not_performed(audit, reason="smoke_panel_missing_adj_close")
        prices = frame.loc[:, ["date", "instrument", "adj_close"]]
        sample = frame.dropna(subset=["date", "instrument", target_column]).head(sample_size)
    elif panel_path is None or not panel_path.is_file():
        return _mark_forward_audit_not_performed(audit, reason="panel_path_unavailable")
    else:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(panel_path)
        row_count = int(parquet.metadata.num_rows)
        sample_indices = _deterministic_sample_indices(
            row_count,
            random_seed=random_seed,
            sample_size=sample_size,
        )
        columns = ["date", "instrument", "interval", target_column]
        if "adj_close" in parquet.schema_arrow.names:
            columns.append("adj_close")
        collected: list[pd.DataFrame] = []
        offset = 0
        set(sample_indices.tolist())
        for batch in parquet.iter_batches(columns=columns, batch_size=250_000):
            length = int(batch.num_rows)
            next_offset = offset + length
            hit = (
                sample_indices[(sample_indices >= offset) & (sample_indices < next_offset)] - offset
            )
            if len(hit):
                part = batch.to_pandas().iloc[hit]
                collected.append(part)
            offset = next_offset
            if len(collected) and sum(len(part) for part in collected) >= len(sample_indices):
                break
        sample = (
            pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=columns)
        )
        substrate = (
            Path(price_substrate_path) if price_substrate_path else SIP_ADJUSTED_PANEL_SOURCE
        )
        instruments = set(sample["instrument"].astype(str).tolist())
        if not substrate.exists():
            return _mark_forward_audit_not_performed(
                audit,
                reason=f"price_substrate_missing:{substrate}",
            )
        try:
            prices = _load_price_frame(substrate, instruments)
        except FileNotFoundError:
            return _mark_forward_audit_not_performed(
                audit,
                reason=f"price_substrate_missing:{substrate}",
            )

    audit["verification_status"] = "performed"
    audit["mismatch_count"] = 0
    audit["maximum_absolute_error"] = 0.0
    audit["missing_future_price_count"] = 0
    audit["incorrect_terminal_label_count"] = 0

    mismatches: list[dict[str, object]] = []
    rows_checked = 0
    mismatch_count = 0
    missing_future = 0
    terminal_bad = 0
    max_error = 0.0
    for _, row in sample.iterrows():
        if pd.isna(row.get(target_column)):
            continue
        rows_checked += 1
        feature_date = str(pd.to_datetime(row["date"], errors="coerce").strftime("%Y-%m-%d"))
        instrument = str(row["instrument"])
        observed = float(row[target_column])
        expected, reason = _expected_forward_return(
            prices,
            instrument=instrument,
            feature_date=feature_date,
            horizon_days=horizon_days,
        )
        if reason == "missing_future_price":
            missing_future += 1
            if not np.isfinite(observed):
                continue
            terminal_bad += 1
            continue
        if expected is None:
            missing_future += 1
            continue
        error = abs(observed - expected)
        if error > _NUMERICAL_TOLERANCE:
            mismatch_count += 1
            max_error = max(max_error, error)
            if len(mismatches) < 12:
                mismatches.append(
                    {
                        "date": feature_date,
                        "instrument": instrument,
                        "interval": str(row.get("interval", "")),
                        "observed": observed,
                        "expected": expected,
                        "absolute_error": error,
                    }
                )
    audit.update(
        {
            "rows_checked": rows_checked,
            "mismatch_count": mismatch_count,
            "maximum_absolute_error": float(max_error),
            "missing_future_price_count": missing_future,
            "incorrect_terminal_label_count": terminal_bad,
            "sample_mismatches": mismatches,
        }
    )
    return audit


def build_diagnostic_findings_and_recommendations(
    *,
    validation_bundle: dict[str, object],
    diversity_report: dict[str, object],
) -> dict[str, object]:
    """Summarize non-blocking diagnostic findings and recommendations."""
    findings: list[dict[str, object]] = []
    recommendations: list[str] = []

    alignment = validation_bundle.get("target_alignment", {})
    if not alignment.get("forward_looking_satisfied"):
        findings.append(
            {
                "code": "target_alignment_not_forward_looking",
                "severity": "error",
                "message": "Resolved target is not forward-looking.",
            }
        )
        recommendations.append("Rebuild the panel with forward_return_horizon supervision.")

    numerical = validation_bundle.get("forward_target_numerical", {})
    if (
        numerical.get("verification_status") == "performed"
        and int(numerical.get("mismatch_count", 0) or 0) > 0
    ):
        findings.append(
            {
                "code": "forward_target_numerical_mismatch",
                "severity": "error",
                "message": "Forward target values disagree with adj_close recomputation.",
            }
        )
    elif numerical.get("verification_status") == "not_performed":
        findings.append(
            {
                "code": "forward_target_numerical_not_performed",
                "severity": "warning",
                "message": str(numerical.get("skipped_reason", "numerical audit skipped")),
            }
        )

    sanity = validation_bundle.get("prediction_sanity", {})
    if int(sanity.get("nonfinite_prediction_count", 0) or 0) > 0:
        findings.append(
            {
                "code": "nonfinite_predictions",
                "severity": "error",
                "message": "Prediction panel contains NaN or Inf values.",
            }
        )
    if int(sanity.get("duplicate_prediction_key_count", 0) or 0) > 0:
        findings.append(
            {
                "code": "duplicate_prediction_keys",
                "severity": "error",
                "message": "Duplicate prediction keys detected.",
            }
        )
    implausible = sanity.get("implausible_scale_models") or []
    if implausible:
        findings.append(
            {
                "code": "implausible_prediction_scale",
                "severity": "warning",
                "message": "Some model/fold predictions have implausible scale for a one-day return target.",
                "details": implausible,
            }
        )

    elastic = validation_bundle.get("elastic_net_degeneracy", {})
    if elastic.get("constant_prediction_flag"):
        findings.append(
            {
                "code": "elastic_net_constant_predictions",
                "severity": "warning",
                "message": "Elastic Net predictions are constant or near-constant.",
            }
        )

    coverage = diversity_report.get("diagnostic_coverage") or {}
    if coverage and not coverage.get("coverage_satisfied"):
        findings.append(
            {
                "code": "diagnostic_coverage_incomplete",
                "severity": "error",
                "message": "Stratified diagnostic sample did not cover all model/fold combinations evenly.",
                "details": coverage,
            }
        )

    if diversity_report.get("low_diversity_warning"):
        recommendations.append(
            "Review redundant model pairs before candidate-portfolio construction."
        )

    eligible_router = int(diversity_report.get("eligible_router_child_count", 0) or 0)
    if eligible_router <= 0 and int(diversity_report.get("model_count", 0) or 0) >= 2:
        recommendations.append(
            "No router-eligible children detected; inspect dispersion and fold IC."
        )

    return {
        "schema_version": "model_matrix_findings.v1",
        "findings": findings,
        "recommendations": recommendations,
        "router_representative_children": diversity_report.get(
            "router_representative_children", []
        ),
        "eligible_router_child_count": eligible_router,
    }


def _target_semantics(column: str) -> str:
    if column in _FORWARD_TARGET_COLUMNS:
        return "forward_looking"
    if column in _CONTEMPORANEOUS_TARGET_COLUMNS:
        return "contemporaneous_same_row"
    return "unknown"


def build_target_alignment_audit(
    *,
    panel_path: Path | None,
    target_column: str,
    target_metadata: dict[str, object],
    unique_dates: tuple[str, ...] | list[str] | None = None,
    sample_size: int = 12,
) -> dict[str, object]:
    """Prove (or flag) whether features at date t supervise a forward return."""
    horizon_days = int(target_metadata.get("horizon_days", 0) or 0)
    semantics = _target_semantics(target_column)
    audit: dict[str, object] = {
        "schema_version": "target_alignment_audit.v1",
        "resolved_target_column": target_column,
        "configured_target": target_metadata.get("configured_target"),
        "horizon_days": horizon_days,
        "target_semantics": semantics,
        "forward_looking_required": True,
        "forward_looking_satisfied": semantics == "forward_looking",
        "same_date_leakage_count": 0,
        "sample_rows": [],
        "warnings": [],
    }
    if panel_path is None or not panel_path.is_file():
        audit["warnings"].append("panel_path_unavailable_for_alignment_sample")
        return audit

    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(panel_path).schema_arrow.names)
    read_columns = ["date", "instrument"]
    for candidate in ("forward_return_horizon", "adjusted_return_1d", target_column):
        if candidate in schema_names and candidate not in read_columns:
            read_columns.append(candidate)

    frame = pq.read_table(panel_path, columns=read_columns).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["instrument"] = frame["instrument"].astype(str)
    if target_column in frame.columns:
        frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")

    date_index = {str(d): idx for idx, d in enumerate(sorted(set(frame["date"].dropna())))}
    samples: list[dict[str, object]] = []
    leakage_count = 0

    if (
        target_column in _CONTEMPORANEOUS_TARGET_COLUMNS
        and "forward_return_horizon" in frame.columns
    ):
        merged = frame.dropna(subset=[target_column, "forward_return_horizon"])
        if len(merged):
            corr = float(merged[target_column].corr(merged["forward_return_horizon"]))
            audit["contemporaneous_vs_forward_horizon_corr"] = corr
            audit["warnings"].append(
                "resolved_target_is_contemporaneous_but_forward_return_horizon_also_present"
            )

    sample_frame = frame.dropna(subset=["date", "instrument"]).head(
        max(sample_size * 50, sample_size)
    )
    for _, row in sample_frame.iterrows():
        if len(samples) >= sample_size:
            break
        feature_date = str(row["date"])
        code = date_index.get(feature_date)
        label_end_date: str | None = None
        if code is not None and horizon_days > 0:
            end_code = code + horizon_days
            ordered = sorted(date_index, key=lambda d: date_index[d])
            if end_code < len(ordered):
                label_end_date = ordered[end_code]
        entry = {
            "feature_date": feature_date,
            "instrument": str(row["instrument"]),
            "label_start_date": feature_date,
            "label_end_date": label_end_date,
            "horizon_days": horizon_days,
            "target_value": float(row[target_column])
            if target_column in row and pd.notna(row[target_column])
            else None,
        }
        if "forward_return_horizon" in row and pd.notna(row["forward_return_horizon"]):
            entry["forward_return_horizon"] = float(row["forward_return_horizon"])
        if "adjusted_return_1d" in row and pd.notna(row["adjusted_return_1d"]):
            entry["adjusted_return_1d"] = float(row["adjusted_return_1d"])
        samples.append(entry)
        if semantics == "contemporaneous_same_row":
            leakage_count += 1

    audit["same_date_leakage_count"] = (
        leakage_count if semantics == "contemporaneous_same_row" else 0
    )
    audit["sample_rows"] = samples
    if semantics != "forward_looking":
        audit["warnings"].append(
            "features_at_t_are_not_supervising_forward_return; prefer forward_return_horizon"
        )
    return audit


def build_chronological_split_audit(
    *,
    unique_dates: tuple[str, ...] | list[str],
    boundaries: list[Any],
    source_chronological_order: bool,
    panel_path: Path | None = None,
    instrument_sample: int = 5,
) -> dict[str, object]:
    """Validate fold construction assumptions when the source file is unsorted."""
    ordered_dates = tuple(sorted(str(d) for d in unique_dates))
    fold_overlap_count = 0
    train_after_test_violation_count = 0
    fold_checks: list[dict[str, object]] = []

    test_sets: list[set[int]] = []
    for boundary in boundaries:
        train_codes = set(range(int(boundary.train_start_code), int(boundary.train_end_code)))
        test_codes = set(range(int(boundary.test_start_code), int(boundary.test_end_code)))
        overlap = len(train_codes & test_codes)
        fold_overlap_count += overlap
        if int(boundary.train_end_code) > int(boundary.test_start_code):
            train_after_test_violation_count += 1
        test_sets.append(test_codes)
        fold_checks.append(
            {
                "fold_id": boundary.fold_id,
                "train_date_start": boundary.train_date_start,
                "train_date_end": boundary.train_date_end,
                "test_date_start": boundary.test_date_start,
                "test_date_end": boundary.test_date_end,
                "purge_dates": list(boundary.purge_dates),
                "train_test_code_overlap": overlap,
            }
        )

    for i, left in enumerate(test_sets):
        for right in test_sets[i + 1 :]:
            fold_overlap_count += len(left & right)

    monotonic_within_instrument = True
    monotonic_violation_count = 0
    if panel_path is not None and panel_path.is_file():
        import pyarrow.parquet as pq

        frame = pq.read_table(panel_path, columns=["date", "instrument"]).to_pandas().head(500_000)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        instruments = (
            frame["instrument"].astype(str).value_counts().head(instrument_sample).index.tolist()
        )
        for instrument in instruments:
            part = frame.loc[frame["instrument"].astype(str) == instrument, "date"].dropna()
            if len(part) > 1 and not part.is_monotonic_increasing:
                monotonic_within_instrument = False
                monotonic_violation_count += 1

    return {
        "schema_version": "chronological_split_audit.v1",
        "source_chronological_order": bool(source_chronological_order),
        "sorted_before_split": True,
        "split_uses_chronological_date_codes": True,
        "date_monotonic_within_instrument": monotonic_within_instrument,
        "monotonic_violation_instrument_count": monotonic_violation_count,
        "fold_overlap_count": int(fold_overlap_count),
        "train_after_test_violation_count": int(train_after_test_violation_count),
        "unique_date_count": len(ordered_dates),
        "fold_checks": fold_checks,
    }


def build_canonical_lineage_report(
    *,
    source: CanonicalPanelSource,
    config: P2Config,
    target_column: str,
    fold_policy: dict[str, object],
) -> dict[str, object]:
    """Document canonical product lineage without implying embedded semantics are owned by features."""
    return {
        "schema_version": "canonical_lineage_report.v1",
        "canonical_daily_panel": {
            "product_id": "sip_adjusted_day_panel_v1",
            "role": "upstream_price_substrate",
            "embedded_in_run": False,
        },
        "full_indicator_feature_panel": {
            "product_id": source.product_identity,
            "panel_path": str(source.panel_path) if source.panel_path else None,
            "manifest_path": str(source.manifest_path) if source.manifest_path else None,
            "row_count": source.row_count,
            "owns_features": True,
            "owns_supervision": False,
        },
        "target_label_panel": {
            "product_id": "target_label_panel",
            "physical_storage": "embedded_supervision_columns",
            "host_product": source.product_identity,
            "resolved_column": target_column,
            "note": "Supervision columns are joined onto the feature panel; not a separate parquet product.",
        },
        "fold_split_panel": {
            "product_id": "fold_split_panel",
            "physical_storage": "derived_at_train_time",
            "host_product": None,
            "policy": fold_policy,
            "note": "Walk-forward folds are computed from chronological date codes during train-matrix.",
        },
        "experiment_config_path": config.experiment_config_path,
        "experiment_config_hash": config.experiment_config_hash,
    }


def _calibration_slope(pred: np.ndarray, target: np.ndarray) -> float | None:
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 2:
        return None
    pred_clean = pred[mask]
    target_clean = target[mask]
    pred_var = float(np.var(pred_clean))
    if pred_var <= 0.0:
        return None
    cov = float(np.cov(pred_clean, target_clean, bias=True)[0, 1])
    return cov / pred_var


def _iter_prediction_batches(path: Path, *, batch_size: int = 250_000):
    import pyarrow.parquet as pq

    columns = ["date", "instrument", "interval", "model_id", "fold_id", "prediction"]
    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
        yield batch.to_pandas()


class _ReservoirQuantileTracker:
    def __init__(self, *, capacity: int, seed: int) -> None:
        self._capacity = capacity
        digest = hashlib.sha256(f"{seed}:reservoir".encode()).digest()
        self._rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        self._reservoir: list[float] = []
        self._seen = 0

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        for value in finite.astype(np.float64, copy=False):
            self._seen += 1
            if len(self._reservoir) < self._capacity:
                self._reservoir.append(float(value))
                continue
            replace_idx = int(self._rng.integers(0, self._seen))
            if replace_idx < self._capacity:
                self._reservoir[replace_idx] = float(value)

    def summary(self) -> dict[str, float | None]:
        if not self._reservoir:
            return _quantile_summary(np.array([], dtype=np.float64))
        return _quantile_summary(np.asarray(self._reservoir, dtype=np.float64))

    def extreme_count(self, threshold: float) -> int:
        return int(sum(1 for value in self._reservoir if abs(value) > threshold))


class _StreamingStratumStats:
    def __init__(self, *, seed: int) -> None:
        self.count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._cal_n = 0
        self._sum_pred = 0.0
        self._sum_target = 0.0
        self._sum_pred2 = 0.0
        self._sum_pred_target = 0.0
        self.prediction_reservoir = _ReservoirQuantileTracker(
            capacity=_STREAMING_RESERVOIR_SIZE, seed=seed
        )
        self.target_reservoir = _ReservoirQuantileTracker(
            capacity=_STREAMING_RESERVOIR_SIZE, seed=seed + 1
        )

    def update(self, preds: np.ndarray, targets: np.ndarray) -> None:
        preds = preds.astype(np.float64, copy=False)
        targets = targets.astype(np.float64, copy=False)
        self.count += int(preds.size)
        self.nan_count += int(np.isnan(preds).sum())
        self.inf_count += int(np.isinf(preds).sum())
        finite_pred = preds[np.isfinite(preds)]
        if finite_pred.size:
            self.min_val = float(min(self.min_val, float(np.min(finite_pred))))
            self.max_val = float(max(self.max_val, float(np.max(finite_pred))))
            self.prediction_reservoir.update(finite_pred)
            for value in finite_pred:
                self._n += 1
                delta = float(value) - self._mean
                self._mean += delta / self._n
                delta2 = float(value) - self._mean
                self._m2 += delta * delta2
        finite_target = targets[np.isfinite(targets)]
        if finite_target.size:
            self.target_reservoir.update(finite_target)
        mask = np.isfinite(preds) & np.isfinite(targets)
        if mask.any():
            p = preds[mask]
            t = targets[mask]
            self._cal_n += int(p.size)
            self._sum_pred += float(np.sum(p))
            self._sum_target += float(np.sum(t))
            self._sum_pred2 += float(np.sum(p * p))
            self._sum_pred_target += float(np.sum(p * t))

    def prediction_summary(self) -> dict[str, float | None]:
        summary = self.prediction_reservoir.summary()
        if self._n > 0:
            summary["mean"] = float(self._mean)
            summary["std"] = float(np.sqrt(self._m2 / self._n)) if self._n > 1 else 0.0
        if self.min_val != float("inf"):
            summary["min"] = self.min_val
        if self.max_val != float("-inf"):
            summary["max"] = self.max_val
        return summary

    def target_summary(self) -> dict[str, float | None]:
        return self.target_reservoir.summary()

    def calibration_slope(self) -> float | None:
        if self._cal_n < 2:
            return None
        n = float(self._cal_n)
        numerator = n * self._sum_pred_target - self._sum_pred * self._sum_target
        denominator = n * self._sum_pred2 - self._sum_pred * self._sum_pred
        if abs(denominator) <= 0.0:
            return None
        return float(numerator / denominator)


def build_streaming_prediction_sanity_report(
    predictions_path: Path,
    *,
    target_column: str,
    target_lookup: Callable[[pd.DataFrame], pd.Series],
    random_seed: int = 42,
    batch_size: int = 250_000,
    extreme_threshold: float = 1.0,
    implausible_std_threshold: float = 1.0,
) -> dict[str, object]:
    """Stream the full prediction panel for per-model/fold sanity without clipping."""
    by_model: dict[str, _StreamingStratumStats] = {}
    by_model_fold: dict[tuple[str, str], _StreamingStratumStats] = {}
    duplicate_keys: set[tuple[str, ...]] = set()
    duplicate_key_count = 0
    nonfinite_prediction_count = 0
    rows_scanned = 0
    target_stats = _StreamingStratumStats(seed=random_seed + 99)

    for batch in _iter_prediction_batches(predictions_path, batch_size=batch_size):
        batch = normalize_key_columns(batch.copy())
        rows_scanned += len(batch)
        batch["model_id"] = batch["model_id"].astype(str)
        batch["fold_id"] = batch["fold_id"].astype(str)
        batch["realized"] = target_lookup(batch)
        preds = batch["prediction"].to_numpy(dtype=np.float64)
        targets = batch["realized"].to_numpy(dtype=np.float64)
        nonfinite_prediction_count += int((~np.isfinite(preds)).sum())
        target_stats.update(preds, targets)

        for key_tuple in zip(
            batch["date"].to_numpy(),
            batch["instrument"].to_numpy(),
            batch["interval"].to_numpy(),
            batch["model_id"].to_numpy(),
            batch["fold_id"].to_numpy(),
            strict=True,
        ):
            key = tuple(str(part) for part in key_tuple)
            if key in duplicate_keys:
                duplicate_key_count += 1
            else:
                duplicate_keys.add(key)

        for model_id, model_frame in batch.groupby("model_id", sort=True):
            stats = by_model.setdefault(
                str(model_id), _StreamingStratumStats(seed=hash(str(model_id)) & 0xFFFF)
            )
            model_preds = model_frame["prediction"].to_numpy(dtype=np.float64)
            model_targets = model_frame["realized"].to_numpy(dtype=np.float64)
            stats.update(model_preds, model_targets)
            for fold_id, fold_frame in model_frame.groupby("fold_id", sort=True):
                fold_key = (str(model_id), str(fold_id))
                fold_stats = by_model_fold.setdefault(
                    fold_key,
                    _StreamingStratumStats(seed=(hash(fold_key) ^ random_seed) & 0xFFFF),
                )
                fold_preds = fold_frame["prediction"].to_numpy(dtype=np.float64)
                fold_targets = fold_frame["realized"].to_numpy(dtype=np.float64)
                fold_stats.update(fold_preds, fold_targets)

    def _serialize_stratum(
        model_id: str, stats: _StreamingStratumStats, fold_id: str | None = None
    ) -> dict[str, object]:
        pred_summary = stats.prediction_summary()
        target_summary = stats.target_summary()
        target_std = target_summary.get("std") or 0.0
        pred_std = pred_summary.get("std") or 0.0
        payload: dict[str, object] = {
            "model_id": model_id,
            "prediction": pred_summary,
            "target": target_summary,
            "prediction_to_target_std_ratio": (pred_std / target_std)
            if target_std and target_std > 0
            else None,
            "nan_count": stats.nan_count,
            "inf_count": stats.inf_count,
            "row_count": stats.count,
            "extreme_prediction_count": stats.prediction_reservoir.extreme_count(extreme_threshold),
            "calibration_slope": stats.calibration_slope(),
        }
        if fold_id is not None:
            payload["fold_id"] = fold_id
        return payload

    by_model_rows = [
        _serialize_stratum(model_id, stats) for model_id, stats in sorted(by_model.items())
    ]
    by_model_fold_rows = [
        _serialize_stratum(model_id, stats, fold_id=fold_id)
        for (model_id, fold_id), stats in sorted(by_model_fold.items())
    ]
    implausible: list[dict[str, object]] = []
    for row in by_model_fold_rows:
        pred = row.get("prediction") or {}
        pred_std = pred.get("std")
        pred_mean = pred.get("mean")
        if pred_std is not None and float(pred_std) > implausible_std_threshold:
            implausible.append(
                {
                    "model_id": row["model_id"],
                    "fold_id": row["fold_id"],
                    "prediction_mean": pred_mean,
                    "prediction_std": pred_std,
                }
            )

    return {
        "schema_version": "prediction_sanity_report.v2",
        "target_column": target_column,
        "target_summary": target_stats.target_summary(),
        "rows_scanned": rows_scanned,
        "streaming": True,
        "duplicate_prediction_key_count": duplicate_key_count,
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "implausible_scale_models": implausible,
        "by_model": by_model_rows,
        "by_model_fold": by_model_fold_rows,
    }


def build_prediction_sanity_report(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target_column: str,
    extreme_threshold: float = 1.0,
) -> dict[str, object]:
    """Per-model and per-fold prediction scale diagnostics."""
    keys = list(CANONICAL_PANEL_KEYS)
    panel_slice = panel.copy()
    if "interval" not in panel_slice.columns:
        panel_slice["interval"] = "1d"
    panel_slice = panel_slice[[*keys, target_column]].copy()
    panel_slice["date"] = panel_slice["date"].astype(str)
    panel_slice["instrument"] = panel_slice["instrument"].astype(str)
    panel_slice["interval"] = panel_slice["interval"].astype(str)
    pred = predictions.copy()
    pred["date"] = pred["date"].astype(str)
    pred["instrument"] = pred["instrument"].astype(str)
    pred["interval"] = pred["interval"].astype(str)
    merged = pred.merge(panel_slice, on=keys, how="left")
    target_stats = _quantile_summary(
        pd.to_numeric(merged[target_column], errors="coerce").to_numpy(dtype=np.float64)
    )

    by_model: list[dict[str, object]] = []
    by_model_fold: list[dict[str, object]] = []
    duplicate_key_count = int(merged.duplicated(list(PREDICTION_OUTPUT_KEYS)).sum())
    nonfinite_prediction_count = int((~np.isfinite(merged["prediction"].astype(float))).sum())

    for model_id, model_frame in merged.groupby("model_id", sort=True):
        preds = model_frame["prediction"].to_numpy(dtype=np.float64)
        targets = pd.to_numeric(model_frame[target_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        target_std = float(np.nanstd(targets)) if np.isfinite(targets).any() else 0.0
        pred_std = float(np.nanstd(preds)) if np.isfinite(preds).any() else 0.0
        by_model.append(
            {
                "model_id": str(model_id),
                "prediction": _quantile_summary(preds),
                "target": _quantile_summary(targets),
                "prediction_to_target_std_ratio": (pred_std / target_std)
                if target_std > 0
                else None,
                "nan_count": int(np.isnan(preds).sum()),
                "inf_count": int(np.isinf(preds).sum()),
                "extreme_prediction_count": int(np.sum(np.abs(preds) > extreme_threshold)),
                "calibration_slope": _calibration_slope(preds, targets),
            }
        )
        if "fold_id" not in model_frame.columns:
            continue
        for fold_id, fold_frame in model_frame.groupby("fold_id", sort=True):
            fold_preds = fold_frame["prediction"].to_numpy(dtype=np.float64)
            fold_targets = pd.to_numeric(fold_frame[target_column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            fold_target_std = (
                float(np.nanstd(fold_targets)) if np.isfinite(fold_targets).any() else 0.0
            )
            fold_pred_std = float(np.nanstd(fold_preds)) if np.isfinite(fold_preds).any() else 0.0
            by_model_fold.append(
                {
                    "model_id": str(model_id),
                    "fold_id": str(fold_id),
                    "prediction": _quantile_summary(fold_preds),
                    "target": _quantile_summary(fold_targets),
                    "prediction_to_target_std_ratio": (fold_pred_std / fold_target_std)
                    if fold_target_std > 0
                    else None,
                    "nan_count": int(np.isnan(fold_preds).sum()),
                    "inf_count": int(np.isinf(fold_preds).sum()),
                    "extreme_prediction_count": int(np.sum(np.abs(fold_preds) > extreme_threshold)),
                    "calibration_slope": _calibration_slope(fold_preds, fold_targets),
                }
            )

    return {
        "schema_version": "prediction_sanity_report.v1",
        "target_column": target_column,
        "target_summary": target_stats,
        "duplicate_prediction_key_count": duplicate_key_count,
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "by_model": by_model,
        "by_model_fold": by_model_fold,
    }


def extract_elastic_net_fit_diagnostics(model: Any) -> dict[str, object] | None:
    """Extract fitted Elastic Net diagnostics when the underlying estimator is available."""
    estimator = getattr(model, "_estimator", model)
    if estimator is None:
        return None
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError:
        return None
    if not isinstance(estimator, ElasticNet):
        return None
    coef = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    return {
        "nonzero_coefficient_count": int(np.count_nonzero(np.abs(coef) > 1e-12)),
        "selected_alpha": float(estimator.alpha),
        "selected_l1_ratio": float(estimator.l1_ratio),
    }


def _elastic_net_params_from_yaml(yaml_models: list[dict[str, Any]]) -> dict[str, object] | None:
    for entry in yaml_models:
        if str(entry.get("family")) == "elastic_net":
            params = entry.get("params") or {}
            if isinstance(params, dict):
                return dict(params)
    return None


def build_elastic_net_degeneracy_audit(
    *,
    predictions: pd.DataFrame | None,
    params: dict[str, object] | None,
    fit_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Audit Elastic Net for constant/degenerate predictions."""
    if predictions is None or predictions.empty or "model_id" not in predictions.columns:
        diagnostics = dict(fit_diagnostics or {})
        return {
            "schema_version": "elastic_net_degeneracy_audit.v1",
            "selected_alpha": params.get("alpha") if params else None,
            "selected_l1_ratio": params.get("l1_ratio") if params else None,
            "nonzero_coefficient_count": diagnostics.get("nonzero_coefficient_count"),
            "prediction_unique_count": 0,
            "prediction_range": 0.0,
            "constant_prediction_flag": False,
            "near_zero_std": True,
            "skipped": "predictions_unavailable",
        }
    frame = predictions.loc[predictions["model_id"].astype(str) == "elastic_net"]
    preds = (
        frame["prediction"].to_numpy(dtype=np.float64)
        if len(frame)
        else np.array([], dtype=np.float64)
    )
    unique_count = int(len(np.unique(preds[np.isfinite(preds)]))) if preds.size else 0
    pred_range = float(np.max(preds) - np.min(preds)) if preds.size else 0.0
    constant_flag = unique_count <= 1
    diagnostics = dict(fit_diagnostics or {})
    return {
        "schema_version": "elastic_net_degeneracy_audit.v1",
        "selected_alpha": params.get("alpha") if params else None,
        "selected_l1_ratio": params.get("l1_ratio") if params else None,
        "nonzero_coefficient_count": diagnostics.get("nonzero_coefficient_count"),
        "prediction_unique_count": unique_count,
        "prediction_range": pred_range,
        "constant_prediction_flag": constant_flag,
        "near_zero_std": bool(np.nanstd(preds) < 1e-8) if preds.size else True,
    }


def build_reproducibility_metadata(
    *,
    config: P2Config,
    yaml_models: list[dict[str, Any]],
    run_id: str,
) -> dict[str, object]:
    """Capture run reproducibility metadata for Gate 2 audits."""
    return {
        "schema_version": "model_matrix_reproducibility.v1",
        "run_id": run_id,
        "random_seed": config.random_seed,
        "git_revision": _git_revision(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(
            ("numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "marketmind")
        ),
        "model_hyperparameters": [
            {"family": str(entry.get("family")), "params": dict(entry.get("params") or {})}
            for entry in yaml_models
        ],
        "experiment_config_path": config.experiment_config_path,
        "experiment_config_hash": config.experiment_config_hash,
        "deterministic_rerun_comparison": "not_run",
        "model_artifact_roundtrip": "not_run",
    }


def build_model_matrix_validation_bundle(
    *,
    source: CanonicalPanelSource,
    config: P2Config,
    target_column: str,
    target_metadata: dict[str, object],
    unique_dates: tuple[str, ...] | list[str],
    boundaries: list[Any],
    source_chronological_order: bool,
    fold_policy: dict[str, object],
    predictions: pd.DataFrame | None,
    panel: pd.DataFrame | None,
    yaml_models: list[dict[str, Any]],
    run_id: str,
    elastic_net_params: dict[str, object] | None = None,
    elastic_net_fit_diagnostics: dict[str, object] | None = None,
    prediction_sanity: dict[str, object] | None = None,
    diversity_report: dict[str, object] | None = None,
    price_substrate_path: str | Path | None = None,
) -> dict[str, object]:
    """Aggregate model-matrix validation artifacts as non-blocking diagnostics."""
    target_metadata = {
        **target_metadata,
        "configured_target": str(config.panel_target),
    }
    horizon_days = int(target_metadata.get("horizon_days", 0) or 0)
    effective_price_substrate = price_substrate_path or config.panel_root
    forward_numerical = build_forward_target_numerical_audit(
        panel_path=source.panel_path,
        target_column=target_column,
        horizon_days=horizon_days,
        price_substrate_path=effective_price_substrate,
        random_seed=int(config.random_seed),
        smoke_panel=source.smoke_panel,
    )
    sanity = prediction_sanity
    if sanity is None and predictions is not None and panel is not None:
        sanity = build_prediction_sanity_report(predictions, panel, target_column=target_column)
    bundle = {
        "schema_version": "model_matrix_validation_bundle.v2",
        "target_alignment": build_target_alignment_audit(
            panel_path=source.panel_path,
            target_column=target_column,
            target_metadata=target_metadata,
            unique_dates=unique_dates,
        ),
        "forward_target_numerical": forward_numerical,
        "chronological_split": build_chronological_split_audit(
            unique_dates=unique_dates,
            boundaries=boundaries,
            source_chronological_order=source_chronological_order,
            panel_path=source.panel_path,
        ),
        "canonical_lineage": build_canonical_lineage_report(
            source=source,
            config=config,
            target_column=target_column,
            fold_policy=fold_policy,
        ),
        "prediction_sanity": sanity
        or {"schema_version": "prediction_sanity_report.v2", "skipped": True},
        "elastic_net_degeneracy": build_elastic_net_degeneracy_audit(
            predictions=predictions if predictions is not None else None,
            params=elastic_net_params,
            fit_diagnostics=elastic_net_fit_diagnostics,
        ),
        "reproducibility": build_reproducibility_metadata(
            config=config,
            yaml_models=yaml_models,
            run_id=run_id,
        ),
    }
    bundle["findings_and_recommendations"] = build_diagnostic_findings_and_recommendations(
        validation_bundle=bundle,
        diversity_report=diversity_report or {},
    )
    return bundle


__all__ = [
    "build_canonical_lineage_report",
    "build_chronological_split_audit",
    "build_diagnostic_findings_and_recommendations",
    "build_elastic_net_degeneracy_audit",
    "build_forward_target_numerical_audit",
    "build_model_matrix_validation_bundle",
    "build_prediction_sanity_report",
    "build_reproducibility_metadata",
    "build_streaming_prediction_sanity_report",
    "build_target_alignment_audit",
    "extract_elastic_net_fit_diagnostics",
]

"""Phase 3 persistence probe: verify raw → frame → parquet prediction fidelity."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.run_layout import allocate_run_dir
from pysrc.models.registry import model_entries_from_yaml
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.canonical_data import require_canonical_panel_source_for_real_run
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.p2_config_loader import (
    MetaRouterExperimentSpec,
    load_p2_config,
    parse_meta_router_experiment,
    yaml_to_p2_config,
)

if TYPE_CHECKING:
    from pysrc.pipeline.panel.train_model_matrix import FoldBoundary, ScratchPanel

LOG = get_logger(__name__)

PROBE_MODEL_FAMILIES: frozenset[str] = frozenset({"ridge", "random_forest", "quantile_regression"})
PROBE_FOLD_ID = "fold_0"
DEFAULT_KEY_SAMPLE_SIZE = 64
_GRAIN_COLUMNS = ("model_id", "fold_id", "date", "instrument", "interval")
_VALUE_COLUMNS = ("prediction", "confidence")
_RANK_COLUMN = "prediction_rank"


def _train_matrix():
    from pysrc.pipeline.panel import train_model_matrix as module

    return module


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _probe_key_rank(master_seed: int, fold_id: str, key: tuple[str, str, str]) -> bytes:
    date, interval, instrument = key
    return hashlib.sha256(
        f"{master_seed}:{fold_id}:{date}:{interval}:{instrument}".encode()
    ).digest()


def deterministic_probe_key_subset(
    keys: list[tuple[str, str, str]],
    *,
    master_seed: int,
    fold_id: str,
    max_keys: int,
) -> frozenset[tuple[str, str, str]]:
    """Select a stable subset of test keys for persistence comparison."""

    if max_keys <= 0:
        raise ValueError("max_keys must be positive")
    if not keys:
        return frozenset()
    unique_keys = sorted(set(keys))
    ranked = sorted(unique_keys, key=lambda key: _probe_key_rank(master_seed, fold_id, key))
    return frozenset(ranked[: min(max_keys, len(ranked))])


def _fold_boundary_from_report(
    report: dict[str, Any],
    fold_id: str,
    *,
    unique_dates: np.ndarray,
) -> FoldBoundary | None:
    tm = _train_matrix()
    fold_policy = report.get("fold_policy") or {}
    remapped = tm.boundaries_from_fold_date_policy(unique_dates, fold_policy)
    for boundary in remapped:
        if boundary.fold_id == fold_id:
            return boundary

    entries = fold_policy.get("fold_checks") or fold_policy.get("folds") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("fold_id")) != fold_id:
            continue
        test_start_code = int(entry.get("test_start_code", 0))
        test_end_code = int(entry.get("test_end_code", 0))
        if test_end_code <= test_start_code:
            continue
        return tm.FoldBoundary(
            fold_id=str(entry.get("fold_id", fold_id)),
            split=str(entry.get("split", "test")),
            train_start_code=int(entry.get("train_start_code", 0)),
            train_end_code=int(entry.get("train_end_code", 0)),
            test_start_code=test_start_code,
            test_end_code=test_end_code,
            purge_start_code=int(entry.get("purge_start_code", entry.get("test_start_code", 0))),
            train_date_start=str(entry.get("train_date_start", "")) or None,
            train_date_end=str(entry.get("train_date_end", "")) or None,
            test_date_start=str(entry.get("test_date_start", "")),
            test_date_end=str(entry.get("test_date_end", "")),
            purge_dates=tuple(entry.get("purge_dates") or ()),
        )
    return None


def _collect_fold_test_keys(
    *,
    scratch: ScratchPanel,
    boundary: FoldBoundary,
    instrument_codes: np.ndarray,
) -> list[tuple[str, str, str]]:
    tm = _train_matrix()
    keys: list[tuple[str, str, str]] = []
    index_dtype = tm._index_dtype(scratch.n_rows)
    for date_code in range(boundary.test_start_code, boundary.test_end_code):
        date_label = str(scratch.unique_dates[date_code])
        for interval_code, interval_label in enumerate(scratch.unique_intervals):
            group_indices = tm._read_group_indices(
                scratch=scratch,
                date_code=date_code,
                interval_code=interval_code,
                index_dtype=index_dtype,
            )
            if len(group_indices) == 0:
                continue
            order = np.argsort(instrument_codes[group_indices], kind="mergesort")
            ordered = group_indices[order]
            instruments = np.asarray(scratch.unique_instruments, dtype=object)[
                np.asarray(instrument_codes[ordered], dtype=np.intp)
            ]
            keys.extend(
                (date_label, str(interval_label), str(inst)) for inst in instruments.tolist()
            )
    return keys


def _filter_probe_keys(
    *,
    instruments: np.ndarray,
    date_label: str,
    interval_label: str,
    probe_keys: frozenset[tuple[str, str, str]],
) -> np.ndarray:
    keep = np.fromiter(
        ((date_label, interval_label, str(inst)) in probe_keys for inst in instruments.tolist()),
        dtype=np.bool_,
        count=len(instruments),
    )
    return keep


def _capture_model_fold_predictions(
    *,
    config: P2Config,
    scratch: ScratchPanel,
    entry: dict[str, Any],
    features: list[str],
    target_column: str,
    boundary: FoldBoundary,
    run_id: str,
    probe_keys: frozenset[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Minimal in-process extract of ``_low_memory_model_fold_worker`` predict path."""
    tm = _train_matrix()

    family = str(entry["family"])
    x_all = np.memmap(
        scratch.feature_path, dtype=np.float32, mode="r", shape=(scratch.n_rows, scratch.n_features)
    )
    y_all = np.memmap(scratch.target_path, dtype=np.float32, mode="r", shape=(scratch.n_rows,))
    instrument_codes = np.memmap(
        scratch.instrument_code_path, dtype=np.int32, mode="r", shape=(scratch.n_rows,)
    )
    finite_target = np.memmap(
        scratch.finite_target_path, dtype=np.bool_, mode="r", shape=(scratch.n_rows,)
    )
    index_dtype = tm._index_dtype(scratch.n_rows)

    train_indices = tm._finite_train_indices_from_groups(
        scratch=scratch,
        finite_target=finite_target,
        instrument_codes=instrument_codes,
        boundary=boundary,
        index_dtype=index_dtype,
    )
    row_limit = tm.resolve_train_row_limit(family=family, config=config)
    if row_limit > 0 and len(train_indices) > row_limit:
        train_indices = tm.subsample_train_indices(
            train_indices,
            max_rows=row_limit,
            master_seed=config.random_seed,
            family=family,
            fold_id=boundary.fold_id,
        ).astype(index_dtype, copy=False)
    if len(train_indices) == 0:
        raise ValueError(f"Fold {boundary.fold_id} has zero eligible training rows for {family}")

    model = tm._fit_model(
        family=family,
        entry=entry,
        features=features,
        config=config,
        train_rows=int(len(train_indices)),
        fold_id=boundary.fold_id,
    )
    x_train, y_train, train_path = tm._compact_training_matrix(
        x_all=x_all,
        y_all=y_all,
        train_indices=train_indices,
        scratch_dir=scratch.scratch_dir,
        family=family,
        fold_id=boundary.fold_id,
        chunk_rows=int(config.panel_train_chunk_rows),
    )
    model.fit(x_train, y_train, fold_id=boundary.fold_id)
    if train_path is not None:
        del x_train
        train_path.unlink(missing_ok=True)
    del y_train, train_indices
    gc.collect()

    raw_rows: list[dict[str, object]] = []
    frame_parts: list[pd.DataFrame] = []
    for date_code in range(boundary.test_start_code, boundary.test_end_code):
        date_label = str(scratch.unique_dates[date_code])
        for interval_code, interval_label in enumerate(scratch.unique_intervals):
            group_indices = tm._read_group_indices(
                scratch=scratch,
                date_code=date_code,
                interval_code=interval_code,
                index_dtype=index_dtype,
            )
            if len(group_indices) == 0:
                continue
            order = np.argsort(instrument_codes[group_indices], kind="mergesort")
            group_indices = group_indices[order]
            instruments = np.asarray(scratch.unique_instruments, dtype=object)[
                np.asarray(instrument_codes[group_indices], dtype=np.intp)
            ]
            keep = _filter_probe_keys(
                instruments=instruments,
                date_label=date_label,
                interval_label=str(interval_label),
                probe_keys=probe_keys,
            )
            if not keep.any():
                continue
            selected = group_indices[keep]
            selected_instruments = instruments[keep]
            x_chunk = np.asarray(x_all[selected, :], dtype=np.float32)
            if hasattr(model, "predict_with_confidence"):
                preds, conf = model.predict_with_confidence(x_chunk)
            else:
                preds = model.predict(x_chunk).reshape(-1)
                conf = model.predict_confidence(x_chunk).reshape(-1)
            preds = np.asarray(preds, dtype=np.float64).reshape(-1)
            conf = np.asarray(conf, dtype=np.float64).reshape(-1)
            if len(preds) != len(selected) or len(conf) != len(selected):
                raise ValueError("Prediction/confidence length mismatch")
            if not np.isfinite(preds).all() or not np.isfinite(conf).all():
                raise ValueError("Model emitted non-finite prediction or confidence")
            for inst, pred, confidence in zip(
                selected_instruments.tolist(), preds.tolist(), conf.tolist(), strict=True
            ):
                raw_rows.append(
                    {
                        "model_id": family,
                        "fold_id": boundary.fold_id,
                        "date": date_label,
                        "instrument": str(inst),
                        "interval": str(interval_label),
                        "prediction": float(pred),
                        "confidence": float(confidence),
                    }
                )
            frame = pd.DataFrame(
                {
                    "run_id": run_id,
                    "model_id": family,
                    "model_family": family,
                    "instrument": selected_instruments.astype(str),
                    "date": date_label,
                    "interval": str(interval_label),
                    "fold_id": boundary.fold_id,
                    "split": boundary.split,
                    "prediction": preds.astype(float),
                    "confidence": conf.astype(float),
                    "target_name": target_column,
                }
            )
            frame = tm.rank_prediction_frame(frame)
            if frame.duplicated(list(_GRAIN_COLUMNS)).any():
                raise ValueError("Duplicate prediction output keys in probe frame")
            frame_parts.append(frame)

    del x_all, y_all, instrument_codes, finite_target, model
    gc.collect()

    if not raw_rows:
        raise ValueError(f"No probe keys matched test rows for {family}/{boundary.fold_id}")

    raw_frame = pd.DataFrame(raw_rows)
    frame_prediction = pd.concat(frame_parts, ignore_index=True)
    parquet_path = scratch.scratch_dir / f"persistence_probe_{family}_{boundary.fold_id}.parquet"
    tm._write_prediction_fragment(frame_prediction, parquet_path)
    parquet_prediction = pd.read_parquet(parquet_path)
    parquet_path.unlink(missing_ok=True)
    return raw_frame, frame_prediction, parquet_prediction


def _compare_prediction_layers(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    compare_rank: bool,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, object]:
    left_keys = set(map(tuple, left[list(_GRAIN_COLUMNS)].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[list(_GRAIN_COLUMNS)].itertuples(index=False, name=None)))
    merged = left.merge(
        right,
        on=list(_GRAIN_COLUMNS),
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    missing_left = int((merged["_merge"] == "right_only").sum())
    missing_right = int((merged["_merge"] == "left_only").sum())
    both = merged[merged["_merge"] == "both"].copy()
    mismatches: list[dict[str, object]] = []
    value_mismatch_count = 0
    for column in _VALUE_COLUMNS:
        left_col = f"{column}_left"
        right_col = f"{column}_right"
        bad = ~np.isclose(
            both[left_col].astype(float), both[right_col].astype(float), rtol=rtol, atol=atol
        )
        value_mismatch_count += int(bad.sum())
        for row in (
            both.loc[bad, list(_GRAIN_COLUMNS) + [left_col, right_col]]
            .head(5)
            .itertuples(index=False)
        ):
            mismatches.append(
                {
                    "column": column,
                    "model_id": row.model_id,
                    "fold_id": row.fold_id,
                    "date": row.date,
                    "instrument": row.instrument,
                    "interval": row.interval,
                    left_label: getattr(row, left_col),
                    right_label: getattr(row, right_col),
                }
            )
    rank_mismatch_count = 0
    if compare_rank and _RANK_COLUMN in left.columns and _RANK_COLUMN in right.columns:
        rank_bad = both[f"{_RANK_COLUMN}_left"] != both[f"{_RANK_COLUMN}_right"]
        rank_mismatch_count = int(rank_bad.sum())
        for row in (
            both.loc[
                rank_bad, list(_GRAIN_COLUMNS) + [f"{_RANK_COLUMN}_left", f"{_RANK_COLUMN}_right"]
            ]
            .head(5)
            .itertuples(index=False)
        ):
            mismatches.append(
                {
                    "column": _RANK_COLUMN,
                    "model_id": row.model_id,
                    "fold_id": row.fold_id,
                    "date": row.date,
                    "instrument": row.instrument,
                    "interval": row.interval,
                    left_label: getattr(row, f"{_RANK_COLUMN}_left"),
                    right_label: getattr(row, f"{_RANK_COLUMN}_right"),
                }
            )

    return {
        "left": left_label,
        "right": right_label,
        "left_key_count": len(left_keys),
        "right_key_count": len(right_keys),
        "matched_key_count": int(len(both)),
        "missing_in_left": missing_left,
        "missing_in_right": missing_right,
        "value_mismatch_count": value_mismatch_count,
        "rank_mismatch_count": rank_mismatch_count,
        "match": missing_left == 0
        and missing_right == 0
        and value_mismatch_count == 0
        and rank_mismatch_count == 0,
        "sample_mismatches": mismatches,
    }


def _filter_probe_models(yaml_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [entry for entry in yaml_models if str(entry.get("family")) in PROBE_MODEL_FAMILIES]
    if not selected:
        raise ValueError(
            f"Persistence probe requires at least one of {sorted(PROBE_MODEL_FAMILIES)}"
        )
    return selected


def run_model_matrix_persistence_probe(
    config: P2Config,
    experiment: MetaRouterExperimentSpec,
    *,
    run_dir: Path | None = None,
    reference_run_dir: Path | None = None,
    key_sample_size: int = DEFAULT_KEY_SAMPLE_SIZE,
) -> dict[str, Path]:
    """Train fold_0 probe models and compare raw/frame/parquet prediction layers."""
    tm = _train_matrix()

    run_path = (
        Path(run_dir)
        if run_dir is not None
        else Path(
            allocate_run_dir(
                lane="panel_model",
                run_id=f"model_matrix_persistence_{uuid4().hex[:12]}",
                smoke=config.smoke_test,
            )
        )
    )
    run_path.mkdir(parents=True, exist_ok=True)

    model_entries = (
        [{"family": m.family, "params": m.params} for m in experiment.models]
        if experiment.models
        else [{"family": family} for family in config.panel_model_families]
    )
    yaml_models = _filter_probe_models(model_entries_from_yaml(model_entries))
    source = require_canonical_panel_source_for_real_run(config, smoke_test=config.smoke_test)
    manifest = tm._resolve_source_manifest(source)
    target_column, features, feature_report, target_metadata = (
        tm.resolve_schema_target_and_features(
            config,
            source.schema,
            manifest=manifest,
        )
    )
    horizon = int(target_metadata.get("horizon_days", 0))

    reference_report: dict[str, Any] | None = None
    if reference_run_dir is not None:
        report_path = reference_run_dir / "reports" / "model_matrix_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing reference model matrix report: {report_path}")
        reference_report = json.loads(report_path.read_text(encoding="utf-8"))

    scratch: ScratchPanel | None = None
    try:
        scratch = tm._prepare_scratch_panel(
            source=source,
            config=config,
            run_path=run_path,
            features=features,
            target_column=target_column,
            model_count=len(yaml_models),
        )
        boundary = None
        if reference_report is not None:
            boundary = _fold_boundary_from_report(
                reference_report,
                PROBE_FOLD_ID,
                unique_dates=np.asarray(scratch.unique_dates, dtype=object),
            )
        if boundary is None:
            boundaries = tm.build_walk_forward_boundaries(
                np.asarray(scratch.unique_dates, dtype=object),
                n_folds=3,
                target_horizon_days=horizon,
            )
            boundary = next((item for item in boundaries if item.fold_id == PROBE_FOLD_ID), None)
        if boundary is None:
            raise ValueError(f"Could not resolve fold boundary for {PROBE_FOLD_ID}")

        instrument_codes = np.memmap(
            scratch.instrument_code_path,
            dtype=np.int32,
            mode="r",
            shape=(scratch.n_rows,),
        )
        all_keys = _collect_fold_test_keys(
            scratch=scratch,
            boundary=boundary,
            instrument_codes=instrument_codes,
        )
        probe_keys = deterministic_probe_key_subset(
            all_keys,
            master_seed=int(config.random_seed),
            fold_id=boundary.fold_id,
            max_keys=int(key_sample_size),
        )
        if not probe_keys:
            raise ValueError(f"No test keys available for {boundary.fold_id}")

        model_reports: dict[str, object] = {}
        overall_ok = True
        for entry in yaml_models:
            family = str(entry["family"])
            raw_predict, frame_prediction, parquet_prediction = _capture_model_fold_predictions(
                config=config,
                scratch=scratch,
                entry=entry,
                features=features,
                target_column=target_column,
                boundary=boundary,
                run_id=run_path.name,
                probe_keys=probe_keys,
            )
            comparisons = {
                "raw_vs_frame": _compare_prediction_layers(
                    raw_predict,
                    frame_prediction[list(_GRAIN_COLUMNS) + list(_VALUE_COLUMNS)],
                    left_label="raw_predict",
                    right_label="frame_prediction",
                    compare_rank=False,
                ),
                "frame_vs_parquet": _compare_prediction_layers(
                    frame_prediction,
                    parquet_prediction,
                    left_label="frame_prediction",
                    right_label="parquet_prediction",
                    compare_rank=True,
                ),
                "raw_vs_parquet": _compare_prediction_layers(
                    raw_predict,
                    parquet_prediction[list(_GRAIN_COLUMNS) + list(_VALUE_COLUMNS)],
                    left_label="raw_predict",
                    right_label="parquet_prediction",
                    compare_rank=False,
                ),
            }
            model_ok = all(bool(section.get("match")) for section in comparisons.values())
            overall_ok = overall_ok and model_ok
            model_reports[family] = {
                "fold_id": boundary.fold_id,
                "probe_key_count": int(len(probe_keys)),
                "raw_row_count": int(len(raw_predict)),
                "frame_row_count": int(len(frame_prediction)),
                "parquet_row_count": int(len(parquet_prediction)),
                "comparisons": comparisons,
                "ok": model_ok,
            }

        report = {
            "schema_version": "model_matrix_persistence_probe.v1",
            "run_id": run_path.name,
            "fold_id": boundary.fold_id,
            "target_column": target_column,
            "target_metadata": target_metadata,
            "feature_count": len(features),
            "feature_usage": feature_report,
            "probe_model_families": [str(entry["family"]) for entry in yaml_models],
            "key_sample_size": int(key_sample_size),
            "probe_key_count": int(len(probe_keys)),
            "probe_keys": [
                {"date": key[0], "interval": key[1], "instrument": key[2]}
                for key in sorted(
                    probe_keys,
                    key=lambda item: _probe_key_rank(
                        int(config.random_seed), boundary.fold_id, item
                    ),
                )
            ],
            "reference_run_id": reference_report.get("run_id") if reference_report else None,
            "models": model_reports,
            "overall_ok": overall_ok,
        }
        report_path = run_path / "reports" / "persistence_probe.json"
        atomic_write_json(report_path, _json_ready(report))
        LOG.info(
            "model_matrix_persistence_probe_complete",
            run_id=run_path.name,
            overall_ok=overall_ok,
            probe_key_count=len(probe_keys),
        )
        return {"run_dir": run_path, "report": report_path}
    finally:
        if scratch is not None:
            shutil.rmtree(scratch.scratch_dir, ignore_errors=True)


def run_model_matrix_persistence_probe_from_yaml(
    config_path: Path | str,
    *,
    smoke_test: bool = False,
    random_seed: int = 42,
    processed_data_root: str | None = None,
    run_dir: Path | None = None,
    reference_run_dir: Path | None = None,
    key_sample_size: int = DEFAULT_KEY_SAMPLE_SIZE,
    max_train_rows_per_fold: int | None = None,
    quantile_max_train_rows_per_fold: int | None = None,
) -> dict[str, Path]:
    resolved = Path(config_path)
    yaml_dict = load_p2_config(resolved)
    experiment = parse_meta_router_experiment(yaml_dict)
    overrides: dict[str, Any] = {"smoke_test": smoke_test, "random_seed": random_seed}
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
    config = yaml_to_p2_config(
        yaml_dict,
        cli_overrides=overrides,
        config_path=resolved,
    )
    return run_model_matrix_persistence_probe(
        config,
        experiment,
        run_dir=run_dir,
        reference_run_dir=reference_run_dir,
        key_sample_size=key_sample_size,
    )


__all__ = [
    "DEFAULT_KEY_SAMPLE_SIZE",
    "PROBE_FOLD_ID",
    "PROBE_MODEL_FAMILIES",
    "deterministic_probe_key_subset",
    "run_model_matrix_persistence_probe",
    "run_model_matrix_persistence_probe_from_yaml",
]

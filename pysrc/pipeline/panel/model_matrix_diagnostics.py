"""Recompute model-matrix diagnostics from existing prediction artifacts without retraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.canonical_data import require_canonical_panel_source_for_real_run
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.p2_config_loader import load_p2_config, yaml_to_p2_config
from pysrc.pipeline.panel.model_diversity import (
    build_panel_target_lookup,
    build_streaming_model_diversity_report,
)
from pysrc.pipeline.panel.model_matrix_target_audit import (
    build_prediction_train_range_audit,
    build_train_target_fold_audit,
)
from pysrc.pipeline.panel.model_matrix_validation import (
    build_model_matrix_validation_bundle,
    build_streaming_prediction_sanity_report,
)

LOG = get_logger(__name__)


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


def _load_run_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "reports" / "model_matrix_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing model matrix report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _fold_boundaries_from_report(report: dict[str, Any]) -> list[Any]:
    from pysrc.pipeline.panel.train_model_matrix import FoldBoundary

    fold_policy = report.get("fold_policy") or {}
    checks = fold_policy.get("fold_checks") or fold_policy.get("folds") or []
    boundaries: list[FoldBoundary] = []
    for entry in checks:
        if not isinstance(entry, dict):
            continue
        boundaries.append(
            FoldBoundary(
                fold_id=str(entry.get("fold_id", "")),
                split=str(entry.get("split", "test")),
                train_start_code=int(entry.get("train_start_code", 0)),
                train_end_code=int(entry.get("train_end_code", 0)),
                test_start_code=int(entry.get("test_start_code", 0)),
                test_end_code=int(entry.get("test_end_code", 0)),
                purge_start_code=int(
                    entry.get("purge_start_code", entry.get("test_start_code", 0))
                ),
                train_date_start=str(entry.get("train_date_start", "")) or None,
                train_date_end=str(entry.get("train_date_end", "")) or None,
                test_date_start=str(entry.get("test_date_start", "")),
                test_date_end=str(entry.get("test_date_end", "")),
                purge_dates=tuple(entry.get("purge_dates") or ()),
            )
        )
    return boundaries


def _resolve_config_and_source(
    run_dir: Path,
    report: dict[str, Any],
    *,
    config: P2Config | None,
    config_path: Path | None,
    random_seed: int | None,
) -> tuple[P2Config, Any, str, dict[str, object], list[Any], int]:
    if config is None:
        yaml_path = config_path or Path(
            str(
                report.get("experiment_config_path")
                or "research/p2/configs/panel_model_matrix.yaml"
            )
        )
        yaml_dict = load_p2_config(yaml_path)
        overrides: dict[str, object] = {
            "random_seed": int(
                random_seed if random_seed is not None else report.get("random_seed", 42)
            ),
            "experiment_config_path": str(yaml_path),
        }
        config = yaml_to_p2_config(yaml_dict, cli_overrides=overrides, config_path=yaml_path)

    seed = int(random_seed if random_seed is not None else config.random_seed)
    source = require_canonical_panel_source_for_real_run(config, smoke_test=config.smoke_test)
    target_metadata = dict(report.get("target_metadata") or source.target_metadata or {})
    target_column = str(
        report.get("target_column") or target_metadata.get("column") or config.panel_target
    )
    target_metadata.setdefault("horizon_days", int(config.panel_target_horizon_days or 1))
    fold_policy = dict(report.get("fold_policy") or {})
    fold_entries = fold_policy.get("fold_checks") or fold_policy.get("folds") or []
    fold_count = int(fold_policy.get("fold_count") or len(fold_entries) or 3)
    boundaries: list[Any] = []
    if source.panel_path is not None:
        from pysrc.pipeline.panel.model_matrix_target_audit import _collect_unique_dates
        from pysrc.pipeline.panel.train_model_matrix import (
            boundaries_from_fold_date_policy,
            build_walk_forward_boundaries,
        )

        unique_dates = _collect_unique_dates(source.panel_path)
        boundaries = boundaries_from_fold_date_policy(unique_dates, fold_policy)
        if not boundaries:
            coded = _fold_boundaries_from_report(report)
            if coded and any(int(b.train_end_code) > 0 for b in coded):
                boundaries = coded
        if not boundaries:
            boundaries = build_walk_forward_boundaries(
                unique_dates,
                n_folds=fold_count,
                target_horizon_days=int(target_metadata.get("horizon_days", 1) or 1),
            )
    return config, source, target_column, target_metadata, boundaries, seed


def run_target_root_cause_investigation(
    run_dir: Path,
    *,
    config: P2Config | None = None,
    config_path: Path | None = None,
    random_seed: int | None = None,
    include_diversity: bool = True,
    max_sample_rows: int = 2_000_000,
) -> dict[str, Path]:
    """Run Phase 1-2 target audits and optionally refresh diversity/validation reports."""
    run_dir = run_dir.resolve()
    report = _load_run_report(run_dir)
    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction panel: {pred_path}")

    config, source, target_column, target_metadata, boundaries, seed = _resolve_config_and_source(
        run_dir,
        report,
        config=config,
        config_path=config_path,
        random_seed=random_seed,
    )
    if source.panel_path is None:
        raise ValueError("Canonical panel path required for target investigation")

    from pysrc.pipeline.panel.train_model_matrix import TrainRowPolicy

    train_policy_raw = report.get("train_row_policy") or {}
    train_row_policy = TrainRowPolicy(
        general_max_rows=train_policy_raw.get("general_max_rows"),
        quantile_max_rows=train_policy_raw.get("quantile_max_rows"),
    )

    train_audit = build_train_target_fold_audit(
        source.panel_path,
        target_column=target_column,
        boundaries=boundaries,
        train_row_policy=train_row_policy,
        random_seed=seed,
    )
    range_audit = build_prediction_train_range_audit(
        pred_path,
        source.panel_path,
        boundaries=boundaries,
        target_column=target_column,
        random_seed=seed,
    )

    train_path = run_dir / "reports" / "train_target_fold_audit.json"
    range_path = run_dir / "reports" / "prediction_train_range_audit.json"
    atomic_write_json(train_path, _json_ready(train_audit))
    atomic_write_json(range_path, _json_ready(range_audit))

    paths: dict[str, Path] = {
        "train_target_fold_audit": train_path,
        "prediction_train_range_audit": range_path,
    }

    if include_diversity:
        diag_paths = run_model_matrix_diagnostics(
            run_dir,
            config=config,
            max_sample_rows=max_sample_rows,
            random_seed=seed,
        )
        paths.update(diag_paths)

    report["target_investigation"] = {
        "train_target_fold_audit": str(train_path),
        "prediction_train_range_audit": str(range_path),
        "target_column": target_column,
    }
    atomic_write_json(run_dir / "reports" / "model_matrix_report.json", _json_ready(report))
    LOG.info(
        "target_root_cause_investigation_complete",
        run_id=run_dir.name,
        train_audit=str(train_path),
        range_audit=str(range_path),
    )
    return paths


def run_model_matrix_diagnostics(
    run_dir: Path,
    *,
    config: P2Config | None = None,
    config_path: Path | None = None,
    max_sample_rows: int = 2_000_000,
    random_seed: int | None = None,
) -> dict[str, Path]:
    """Recompute diversity and validation reports for an existing model-matrix run."""
    run_dir = run_dir.resolve()
    report = _load_run_report(run_dir)
    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction panel: {pred_path}")

    config, source, target_column, target_metadata, boundaries, seed = _resolve_config_and_source(
        run_dir,
        report,
        config=config,
        config_path=config_path,
        random_seed=random_seed,
    )
    fold_policy = dict(report.get("fold_policy") or {})
    fold_count = int(
        fold_policy.get("fold_count") or len(fold_policy.get("fold_checks") or []) or 3
    )
    n_models = int(report.get("n_models") or len(report.get("model_capabilities") or {}))

    if source.panel_path is None:
        raise ValueError("Canonical panel path required to recompute diagnostics")
    target_lookup = build_panel_target_lookup(source.panel_path, target_column=target_column)

    diversity = build_streaming_model_diversity_report(
        pred_path,
        target_column=target_column,
        target_lookup=target_lookup,
        top_k=20,
        random_seed=seed,
        max_sample_rows=max_sample_rows,
        expected_model_count=n_models,
        expected_fold_count=fold_count,
    )
    sanity = build_streaming_prediction_sanity_report(
        pred_path,
        target_column=target_column,
        target_lookup=target_lookup,
        random_seed=seed,
    )
    yaml_models = [
        {"family": str(entry.get("family")), "params": dict(entry.get("params") or {})}
        for entry in (report.get("model_capabilities") or {}).values()
        if isinstance(entry, dict)
    ]
    if not yaml_models:
        yaml_models = [
            {"family": model_id, "params": {}} for model_id in diversity.get("models", [])
        ]

    validation = build_model_matrix_validation_bundle(
        source=source,
        config=config,
        target_column=target_column,
        target_metadata=target_metadata,
        unique_dates=tuple(b.train_date_start for b in boundaries) or ("",),
        boundaries=boundaries,
        source_chronological_order=bool(report.get("source_chronological_order", False)),
        fold_policy=fold_policy,
        predictions=None,
        panel=None,
        yaml_models=yaml_models,
        run_id=run_dir.name,
        prediction_sanity=sanity,
        diversity_report=diversity,
    )

    diversity_path = run_dir / "reports" / "model_diversity_report.json"
    validation_path = run_dir / "reports" / "model_matrix_validation.json"
    atomic_write_json(diversity_path, _json_ready(diversity))
    atomic_write_json(validation_path, _json_ready(validation))
    report["diversity"] = {
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
    report["validation_report"] = str(validation_path)
    atomic_write_json(run_dir / "reports" / "model_matrix_report.json", _json_ready(report))
    LOG.info(
        "model_matrix_diagnostics_complete",
        run_id=run_dir.name,
        diversity=str(diversity_path),
        validation=str(validation_path),
    )
    return {
        "model_diversity_report": diversity_path,
        "model_matrix_validation": validation_path,
        "model_matrix_report": run_dir / "reports" / "model_matrix_report.json",
    }


__all__ = ["run_model_matrix_diagnostics", "run_target_root_cause_investigation"]

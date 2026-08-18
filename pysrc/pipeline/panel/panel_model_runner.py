"""P2-PANEL-MODEL-V1: train ticker-level models on the full eligible indicator universe."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.run_layout import allocate_run_dir
from pysrc.contracts.candidate_spec import CandidateSpec
from pysrc.models.tabular import train_and_predict
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.contracts.p2 import P2Config, RunMetadata, RunPhase
from pysrc.pipeline.panel.candidate_matrix import load_panel_candidate_matrix
from pysrc.pipeline.panel.indicator_universe_builder import (
    build_panel_supervision_frame,
    default_panel_model_output_dir,
    record_model_feature_usage,
    require_panel_grain_valid_for_training,
    write_feature_universe_artifacts,
)
from pysrc.pipeline.panel.panel_fold_slices import build_panel_fold_slices
from pysrc.pipeline.panel.panel_targets import resolve_panel_target_column
from pysrc.pipeline.panel.runtime import resolve_sklearn_n_jobs

LOG = get_logger(__name__)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, target)


def _evaluate_panel_candidate(
    candidate: CandidateSpec,
    panel_frame: object,
    config: P2Config,
    *,
    feature_names: tuple[str, ...],
    sklearn_n_jobs: int,
) -> dict[str, object]:
    from pysrc.pipeline.panel.indicator_universe_builder import PanelSupervisionFrame

    assert isinstance(panel_frame, PanelSupervisionFrame)
    if tuple(panel_frame.feature_names) != feature_names:
        raise ValueError("Panel candidate feature_names must match universe.eligible_features")

    fold_slices = build_panel_fold_slices(panel_frame, config)
    if not fold_slices:
        raise ValueError(f"No panel fold slices for candidate {candidate.candidate_id}")

    prediction_parts: list[pd.DataFrame] = []
    metrics: dict[str, float] = {}
    for fold in fold_slices:
        preds_test, _, train_diag = train_and_predict(
            candidate,
            fold.x_train,
            fold.y_train,
            fold.x_validation,
            fold.x_test,
            config.random_seed,
            sklearn_n_jobs=sklearn_n_jobs,
        )
        metrics = train_diag
        if int(train_diag.get("n_features", 0)) != len(feature_names):
            raise ValueError(
                f"Model used n_features={train_diag.get('n_features')} "
                f"but eligible universe has {len(feature_names)}"
            )
        pred_frame = fold.test_meta.copy()
        pred_frame["predicted_forward_return"] = preds_test.reshape(-1)
        pred_frame["realized_forward_return"] = fold.y_test.reshape(-1)
        pred_frame["fold_id"] = fold.fold_id
        pred_frame["candidate_id"] = candidate.candidate_id
        pred_frame["model_family"] = candidate.model_family
        prediction_parts.append(pred_frame)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    y_true = predictions["realized_forward_return"].to_numpy(dtype=np.float64)
    y_pred = predictions["predicted_forward_return"].to_numpy(dtype=np.float64)
    test_r2 = float(r2_score(y_true, y_pred))
    if not np.isfinite(test_r2):
        test_r2 = 0.0
    test_mse = float(mean_squared_error(y_true, y_pred))

    return {
        "candidate_id": candidate.candidate_id,
        "model_family": candidate.model_family,
        "feature_count": len(feature_names),
        "feature_policy": candidate.feature_policy,
        "train_r2": float(metrics.get("train_r2", 0.0)),
        "test_r2": test_r2,
        "test_mse": test_mse,
        "predictions": predictions,
    }


def run_p2_panel_model(config: P2Config | None = None) -> dict[str, Path]:
    """Run P2-PANEL-MODEL-V1 using universe.eligible_features for every candidate."""

    effective_config = config or P2Config()
    require_panel_grain_valid_for_training(effective_config)
    panel = build_panel_supervision_frame(effective_config)
    feature_names = panel.universe.eligible_features
    target_column = resolve_panel_target_column(panel.frame, effective_config)

    run_id = uuid4().hex[:12]
    run_dir = allocate_run_dir(
        lane="panel_model",
        run_id=run_id,
        smoke=bool(effective_config.smoke_test),
    )

    matrix = load_panel_candidate_matrix(effective_config)
    sklearn_n_jobs = resolve_sklearn_n_jobs(effective_config, parallel_workers=1)

    usage_rows: list[dict[str, object]] = []
    candidate_reports: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    for candidate in matrix.candidates:
        if candidate.feature_policy != "full_indicator_universe_v1":
            raise ValueError(
                f"Panel runner requires full_indicator_universe_v1, got {candidate.feature_policy}"
            )
        try:
            result = _evaluate_panel_candidate(
                candidate,
                panel,
                effective_config,
                feature_names=feature_names,
                sklearn_n_jobs=sklearn_n_jobs,
            )
            usage_rows.extend(
                record_model_feature_usage(
                    model_id=candidate.model_family,
                    candidate_id=candidate.candidate_id,
                    feature_names=feature_names,
                    selection_stage="fit",
                )
            )
            predictions = cast_predictions(result.pop("predictions"))
            all_predictions.append(predictions)
            candidate_reports.append(result)
        except Exception as exc:  # noqa: BLE001
            candidate_reports.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "model_family": candidate.model_family,
                    "status": "FAILED_TRAINING",
                    "error": str(exc),
                }
            )

    predictions_path = run_dir / "predictions" / "panel_predictions.parquet"
    if all_predictions:
        _atomic_write_parquet(pd.concat(all_predictions, ignore_index=True), predictions_path)

    meta = RunMetadata(phase=RunPhase.NARROW, random_seed=effective_config.random_seed)
    report = {
        "schema_version": "p2_panel_model_report.v1",
        "run_id": run_id,
        "meta": meta.model_dump(mode="json"),
        "target_column": target_column,
        "eligible_feature_count": len(feature_names),
        "uses_full_discovered_feature_universe": panel.universe.uses_full_discovered_feature_universe,
        "uses_full_indicator_universe": panel.universe.uses_full_indicator_universe,
        "candidates": candidate_reports,
    }
    report_path = run_dir / "reports" / "panel_model_report.json"
    atomic_write_json(report_path, report)

    audit_dir = default_panel_model_output_dir(effective_config)
    write_feature_universe_artifacts(
        panel.universe,
        audit_dir,
        model_feature_usage_rows=usage_rows,
    )

    LOG.info(
        "p2_panel_model_complete",
        run_dir=str(run_dir),
        candidates=len(candidate_reports),
        features=len(feature_names),
    )
    return {
        "run_dir": run_dir,
        "report": report_path,
        "predictions": predictions_path,
        "feature_universe_report": audit_dir / "feature_universe_report.json",
    }


def cast_predictions(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("Expected predictions DataFrame")
    return value

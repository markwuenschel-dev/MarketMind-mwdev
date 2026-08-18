"""Tests for panel train-target root-cause audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pysrc.pipeline.panel.model_matrix_target_audit import (
    build_prediction_train_range_audit,
    build_train_target_fold_audit,
)
from pysrc.pipeline.panel.train_model_matrix import (
    TrainRowPolicy,
    build_chronological_date_codes,
    build_walk_forward_boundaries,
)


def _write_panel(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(pa.Table.from_pandas(frame), path)


@pytest.mark.determinism("d1")
def test_train_target_fold_audit_reports_quantiles_and_extremes(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    dates = pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d").tolist()
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for inst_idx, instrument in enumerate(["AAA", "BBB"]):
            target = float(date_idx - inst_idx) / 10.0
            if date_idx == 0 and instrument == "AAA":
                target = 2.5
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "interval": "1d",
                    "f1": float(date_idx),
                    "f2": float(inst_idx),
                    "forward_return_horizon": target,
                    "adjusted_return_1d": target / 2.0,
                    "raw_return_1d": target / 3.0,
                }
            )
    panel_path = tmp_path / "panel.parquet"
    _write_panel(panel_path, pd.DataFrame(rows))

    _, unique_dates = build_chronological_date_codes([row["date"] for row in rows])
    boundaries = build_walk_forward_boundaries(unique_dates, n_folds=2, target_horizon_days=1)
    audit = build_train_target_fold_audit(
        panel_path=panel_path,
        target_column="forward_return_horizon",
        boundaries=boundaries,
        train_row_policy=TrainRowPolicy(general_max_rows=None, quantile_max_rows=None),
        random_seed=deterministic_seed,
    )

    assert audit["schema_version"] == "train_target_fold_audit.v1"
    assert audit["rows_scanned"] == len(rows)
    assert len(audit["folds"]) == 2
    fold_0 = audit["folds"][0]
    summary = fold_0["target_summary"]
    assert summary["count"] > 0
    assert summary["min"] is not None
    assert summary["max"] is not None
    for key in ("p0.001", "p0.01", "p0.1", "p1", "p50", "p99", "p99.9", "p99.99"):
        assert key in summary
    assert fold_0["count_abs_gt"]["abs_gt_1"] >= 0
    extreme_dates = {row["date"] for row in fold_0["extreme_rows"]}
    assert any(abs(row["forward_return_horizon"]) >= 0.5 for row in fold_0["extreme_rows"])
    assert "f1" in fold_0["extreme_rows"][0]
    assert extreme_dates


@pytest.mark.determinism("d1")
def test_train_target_fold_audit_excludes_non_finite_and_test_rows(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "instrument": ["AAA", "AAA", "AAA", "AAA"],
            "interval": ["1d", "1d", "1d", "1d"],
            "forward_return_horizon": [0.01, float("nan"), 0.03, 0.04],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    panel_path = tmp_path / "panel.parquet"
    _write_panel(panel_path, frame)
    unique_dates = np.asarray(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], dtype=object
    )
    boundaries = build_walk_forward_boundaries(unique_dates, n_folds=1, target_horizon_days=1)
    audit = build_train_target_fold_audit(
        panel_path=panel_path,
        target_column="forward_return_horizon",
        boundaries=boundaries,
        train_row_policy=TrainRowPolicy(general_max_rows=10, quantile_max_rows=5),
        random_seed=deterministic_seed,
    )
    fold = audit["folds"][0]
    assert fold["eligible_train_row_count"] == 1
    assert fold["target_summary"]["min"] == pytest.approx(0.01)
    assert fold["target_summary"]["max"] == pytest.approx(0.01)


@pytest.mark.determinism("d1")
def test_prediction_train_range_audit_counts_out_of_range_predictions(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    rows = [
        {
            "date": date,
            "instrument": "AAA",
            "interval": "1d",
            "forward_return_horizon": target,
            "f1": 1.0,
        }
        for date, target in zip(dates, [0.01, 0.02, 0.03, 0.04, 0.05], strict=True)
    ]
    panel_path = tmp_path / "panel.parquet"
    _write_panel(panel_path, pd.DataFrame(rows))
    unique_dates = np.asarray(dates, dtype=object)
    boundaries = build_walk_forward_boundaries(unique_dates, n_folds=1, target_horizon_days=1)

    train_audit = build_train_target_fold_audit(
        panel_path=panel_path,
        target_column="forward_return_horizon",
        boundaries=boundaries,
        train_row_policy=TrainRowPolicy(general_max_rows=None, quantile_max_rows=None),
        random_seed=deterministic_seed,
    )
    train_min = train_audit["folds"][0]["target_summary"]["min"]
    train_max = train_audit["folds"][0]["target_summary"]["max"]

    pred_rows = [
        {
            "date": "2024-01-05",
            "instrument": "AAA",
            "interval": "1d",
            "model_id": "ridge",
            "fold_id": "fold_0",
            "prediction": pred,
        }
        for pred in [train_min - 0.5, train_max + 0.5, (train_min + train_max) / 2.0]
    ]
    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(pred_rows).to_parquet(pred_path, index=False)

    audit = build_prediction_train_range_audit(
        pred_path=pred_path,
        panel_path=panel_path,
        boundaries=boundaries,
        target_column="forward_return_horizon",
        random_seed=deterministic_seed,
    )

    assert audit["schema_version"] == "prediction_train_range_audit.v1"
    assert len(audit["by_model_fold"]) == 1
    row = audit["by_model_fold"][0]
    assert row["model_id"] == "ridge"
    assert row["fold_id"] == "fold_0"
    assert row["train_target_min"] == pytest.approx(train_min)
    assert row["train_target_max"] == pytest.approx(train_max)
    assert row["below_train_min_count"] == 1
    assert row["above_train_max_count"] == 1
    assert row["prediction_min"] == pytest.approx(train_min - 0.5)
    assert row["prediction_max"] == pytest.approx(train_max + 0.5)

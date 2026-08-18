"""Low-memory panel train-matrix behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.contracts.meta_router import MODEL_PREDICTION_PANEL_COLUMNS
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.p2_config_loader import MetaRouterExperimentSpec, ModelEntrySpec
from pysrc.pipeline.panel.train_model_matrix import train_model_matrix


@pytest.mark.determinism("d1")
def test_low_memory_train_matrix_writes_prediction_panel(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    processed = tmp_path / "processed"
    panel_dir = processed / "full_indicator_feature_panel"
    panel_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d").tolist()
    for date_idx, date in enumerate(dates):
        for inst_idx, instrument in enumerate(["AAA", "BBB", "CCC"]):
            adj_close = 100.0 + float(date_idx) + float(inst_idx)
            next_adj = (
                100.0 + float(date_idx + 1) + float(inst_idx)
                if date_idx < len(dates) - 1
                else float("nan")
            )
            forward = next_adj / adj_close - 1.0 if np.isfinite(next_adj) else float("nan")
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "interval": "1d",
                    "adj_close": adj_close,
                    "f1": float(date_idx),
                    "f2": float(inst_idx),
                    "forward_return_horizon": forward,
                    "adjusted_return_1d": float(date_idx - inst_idx) / 100.0,
                }
            )
    panel = pd.DataFrame(rows)
    panel.sample(frac=1.0, random_state=7).to_parquet(panel_dir / "panel.parquet", index=False)
    (processed / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "processed_data_manifest.v1",
                "row_grain": "ticker_date_interval",
                "row_count": len(panel),
                "indicator_columns": ["f1", "f2"],
                "target_metadata": {
                    "forward_return": {
                        "column": "forward_return_horizon",
                        "horizon_days": 1,
                    }
                },
                "supervision_columns": ["forward_return_horizon", "adjusted_return_1d"],
            }
        ),
        encoding="utf-8",
    )

    config = P2Config(
        processed_data_root=str(processed),
        panel_train_memory_mode="low_memory",
        panel_train_chunk_rows=5,
        panel_target="forward_return",
        panel_target_horizon_days=1,
        panel_model_families=("ridge",),
        panel_preserve_scratch_on_failure=True,
        sklearn_n_jobs=1,
    )
    experiment = MetaRouterExperimentSpec(
        experiment="low_memory_panel_test",
        models=[ModelEntrySpec(family="ridge", params={"alpha": 1.0})],
    )

    result = train_model_matrix(config, experiment, run_dir=tmp_path / "run")
    predictions = pd.read_parquet(result["model_prediction_panel"])
    report = json.loads(result["report"].read_text(encoding="utf-8"))

    assert list(predictions.columns) == list(MODEL_PREDICTION_PANEL_COLUMNS)
    assert not predictions.empty
    assert report["memory_mode"]["resolved_memory_mode"] == "low_memory"
    assert report["target_metadata"]["horizon_days"] == 1
    assert report["fold_policy"]["folds"][0]["purge_dates"]
    assert report["canonical_data"]["duplicate_key_count"] == 0
    assert report["worker_process_isolation"] is True
    assert report["worker_status_count"] > 0
    assert report["prediction_schema_validated"] is True
    assert report["expected_prediction_rows"] == len(predictions)
    assert "downstream_compatibility" not in report
    assert not (tmp_path / "run" / "scratch" / "train_matrix").exists()

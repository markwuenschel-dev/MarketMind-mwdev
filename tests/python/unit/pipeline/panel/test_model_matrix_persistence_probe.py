"""Persistence probe tests for model-matrix prediction layers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.p2_config_loader import MetaRouterExperimentSpec, ModelEntrySpec
from pysrc.pipeline.panel.model_matrix_persistence_probe import (
    PROBE_FOLD_ID,
    deterministic_probe_key_subset,
    run_model_matrix_persistence_probe,
)
from pysrc.pipeline.panel.train_model_matrix import build_walk_forward_boundaries


def _write_synthetic_panel(tmp_path: Path) -> Path:
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
    return processed


@pytest.mark.determinism("d1")
def test_deterministic_probe_key_subset(deterministic_seed: int) -> None:
    _ = deterministic_seed
    keys = [("2024-01-02", "1d", "AAA"), ("2024-01-03", "1d", "BBB"), ("2024-01-04", "1d", "CCC")]
    first = deterministic_probe_key_subset(keys, master_seed=42, fold_id=PROBE_FOLD_ID, max_keys=2)
    second = deterministic_probe_key_subset(keys, master_seed=42, fold_id=PROBE_FOLD_ID, max_keys=2)
    assert first == second
    assert len(first) == 2


@pytest.mark.determinism("d1")
def test_persistence_probe_synthetic_scratch(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    processed = _write_synthetic_panel(tmp_path)
    config, experiment = _persistence_probe_config(processed)

    result = run_model_matrix_persistence_probe(
        config,
        experiment,
        run_dir=tmp_path / "probe_run",
        key_sample_size=6,
    )
    report = json.loads(result["report"].read_text(encoding="utf-8"))

    assert report["overall_ok"] is True
    assert report["fold_id"] == PROBE_FOLD_ID
    assert set(report["probe_model_families"]) == {"ridge", "random_forest", "quantile_regression"}
    assert report["probe_key_count"] == 6
    for family in ("ridge", "random_forest", "quantile_regression"):
        model_report = report["models"][family]
        assert model_report["ok"] is True
        assert model_report["raw_row_count"] == 6
        assert model_report["comparisons"]["raw_vs_frame"]["match"] is True
        assert model_report["comparisons"]["frame_vs_parquet"]["match"] is True
        assert model_report["comparisons"]["raw_vs_parquet"]["match"] is True


def _persistence_probe_config(processed: Path) -> tuple[P2Config, MetaRouterExperimentSpec]:
    config = P2Config(
        processed_data_root=str(processed),
        panel_train_memory_mode="low_memory",
        panel_train_chunk_rows=5,
        panel_target="forward_return",
        panel_target_horizon_days=1,
        panel_model_families=("ridge", "random_forest", "quantile_regression"),
        panel_preserve_scratch_on_failure=True,
        sklearn_n_jobs=1,
        random_seed=42,
    )
    experiment = MetaRouterExperimentSpec(
        experiment="persistence_probe_test",
        models=[
            ModelEntrySpec(family="ridge", params={"alpha": 1.0}),
            ModelEntrySpec(
                family="random_forest",
                params={"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 1},
            ),
            ModelEntrySpec(
                family="quantile_regression",
                params={
                    "backend": "hist_gradient_boosting",
                    "quantile": 0.5,
                    "learning_rate": 0.1,
                    "max_iter": 20,
                    "max_leaf_nodes": 8,
                },
            ),
        ],
    )
    return config, experiment


@pytest.mark.determinism("d1")
def test_persistence_probe_reference_run_date_label_folds(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    """Reference reports store date labels only; probe must remap to scratch date codes."""
    _ = deterministic_seed
    processed = _write_synthetic_panel(tmp_path)
    config, experiment = _persistence_probe_config(processed)
    dates = pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d").tolist()
    fold_0 = next(
        boundary
        for boundary in build_walk_forward_boundaries(
            np.asarray(dates, dtype=object),
            n_folds=3,
            target_horizon_days=1,
        )
        if boundary.fold_id == PROBE_FOLD_ID
    )
    ref_dir = tmp_path / "reference_run"
    reports = ref_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "model_matrix_report.json").write_text(
        json.dumps(
            {
                "run_id": "reference",
                "fold_policy": {
                    "target_horizon_days": 1,
                    "folds": [
                        {
                            "fold_id": fold_0.fold_id,
                            "train_date_start": fold_0.train_date_start,
                            "train_date_end": fold_0.train_date_end,
                            "test_date_start": fold_0.test_date_start,
                            "test_date_end": fold_0.test_date_end,
                            "purge_dates": list(fold_0.purge_dates),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_model_matrix_persistence_probe(
        config,
        experiment,
        run_dir=tmp_path / "probe_run_ref",
        reference_run_dir=ref_dir,
        key_sample_size=6,
    )
    report = json.loads(result["report"].read_text(encoding="utf-8"))
    assert report["probe_key_count"] == 6
    assert report["reference_run_id"] == "reference"
    assert report["overall_ok"] is True

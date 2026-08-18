"""Tests for Gate 2 model-matrix validation audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.panel.model_matrix_validation import (
    build_chronological_split_audit,
    build_model_matrix_validation_bundle,
    build_target_alignment_audit,
)
from pysrc.pipeline.panel.train_model_matrix import build_walk_forward_boundaries


@pytest.mark.determinism("d1")
def test_target_alignment_flags_contemporaneous_target(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "instrument": ["AAA", "AAA", "AAA"],
            "adjusted_return_1d": [0.01, -0.02, 0.03],
            "forward_return_horizon": [0.02, -0.01, 0.04],
        }
    )
    pq.write_table(pa.Table.from_pandas(frame), panel_path)
    audit = build_target_alignment_audit(
        panel_path=panel_path,
        target_column="adjusted_return_1d",
        target_metadata={"horizon_days": 1, "configured_target": "forward_return"},
    )
    assert audit["target_semantics"] == "contemporaneous_same_row"
    assert audit["forward_looking_satisfied"] is False
    assert audit["same_date_leakage_count"] > 0
    assert audit["sample_rows"]


@pytest.mark.determinism("d1")
def test_target_alignment_accepts_forward_horizon(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "instrument": ["AAA", "AAA"],
            "forward_return_horizon": [0.02, -0.01],
        }
    )
    pq.write_table(pa.Table.from_pandas(frame), panel_path)
    audit = build_target_alignment_audit(
        panel_path=panel_path,
        target_column="forward_return_horizon",
        target_metadata={"horizon_days": 1, "configured_target": "forward_return"},
    )
    assert audit["target_semantics"] == "forward_looking"
    assert audit["forward_looking_satisfied"] is True
    assert audit["same_date_leakage_count"] == 0


@pytest.mark.determinism("d1")
def test_chronological_split_audit_reports_zero_overlap(deterministic_seed: int) -> None:
    _ = deterministic_seed
    unique_dates = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08")
    boundaries = build_walk_forward_boundaries(
        np.asarray(unique_dates, dtype=object), n_folds=2, target_horizon_days=1
    )
    audit = build_chronological_split_audit(
        unique_dates=unique_dates,
        boundaries=boundaries,
        source_chronological_order=False,
    )
    assert audit["sorted_before_split"] is True
    assert audit["fold_overlap_count"] == 0
    assert audit["train_after_test_violation_count"] == 0


@pytest.mark.determinism("d1")
def test_forward_numerical_audit_skipped_reports_not_performed(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.panel.model_matrix_validation import build_forward_target_numerical_audit

    audit = build_forward_target_numerical_audit(
        panel_path=tmp_path / "missing.parquet",
        target_column="forward_return_horizon",
        horizon_days=1,
    )
    assert audit["verification_status"] == "not_performed"
    assert audit["rows_checked"] == 0
    assert audit["mismatch_count"] is None


@pytest.mark.determinism("d1")
def test_stratified_sampling_covers_all_models(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.panel.model_diversity import (
        build_streaming_model_diversity_report,
        select_aligned_prediction_keys,
    )

    rows: list[dict[str, object]] = []
    for model_id in ("ridge", "mlp", "elastic_net"):
        for fold_id in ("fold_0", "fold_1"):
            for idx in range(20):
                rows.append(
                    {
                        "date": f"2024-01-{idx + 2:02d}",
                        "instrument": f"T{idx % 5}",
                        "interval": "1d",
                        "model_id": model_id,
                        "fold_id": fold_id,
                        "prediction": float(idx) * 0.001,
                    }
                )
    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(pred_path, index=False)
    selection = select_aligned_prediction_keys(
        pred_path,
        random_seed=7,
        max_sample_rows=120,
        expected_model_count=3,
    )
    assert selection.total_aligned_keys == 40
    assert len(selection.selected_keys) > 0

    def target_lookup(batch: pd.DataFrame) -> pd.Series:
        return pd.Series(np.full(len(batch), 0.01), index=batch.index, dtype=float)

    report = build_streaming_model_diversity_report(
        pred_path,
        target_column="forward_return_horizon",
        target_lookup=target_lookup,
        random_seed=7,
        max_sample_rows=120,
        expected_model_count=3,
        expected_fold_count=2,
    )
    coverage = report["diagnostic_coverage"]
    assert coverage["expected_model_count"] == 3
    assert coverage["observed_model_count"] == 3
    assert coverage["expected_model_fold_combinations"] == 6
    assert coverage["coverage_satisfied"] is True
    assert len(set(coverage["per_model_sampled_row_count"].values())) == 1


@pytest.mark.determinism("d1")
def test_validation_bundle_contains_required_sections(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.canonical_data import CanonicalPanelSource

    predictions = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["AAA", "BBB"],
            "interval": ["1d", "1d"],
            "model_id": ["ridge", "ridge"],
            "fold_id": ["fold_0", "fold_0"],
            "prediction": [0.01, -0.02],
        }
    )
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["AAA", "BBB"],
            "interval": ["1d", "1d"],
            "forward_return_horizon": [0.02, -0.01],
        }
    )
    source = CanonicalPanelSource(
        panel_path=None,
        manifest_path=None,
        manifest={},
        schema={"forward_return_horizon": "double"},
        row_count=2,
        expected_row_grain="instrument_date_interval",
        product_identity="full_indicator_feature_panel",
        file_size_bytes=None,
        file_mtime_ns=None,
        config_hash=None,
        target_metadata={},
    )
    config = P2Config(panel_target="forward_return", panel_target_horizon_days=1)
    unique_dates = ("2024-01-02",)
    boundaries = build_walk_forward_boundaries(
        np.asarray(unique_dates, dtype=object),
        n_folds=1,
        target_horizon_days=1,
    )
    bundle = build_model_matrix_validation_bundle(
        source=source,
        config=config,
        target_column="forward_return_horizon",
        target_metadata={"horizon_days": 1},
        unique_dates=unique_dates,
        boundaries=boundaries,
        source_chronological_order=False,
        fold_policy={"fold_count": 1},
        predictions=predictions,
        panel=panel,
        yaml_models=[{"family": "ridge", "params": {"alpha": 1.0}}],
        run_id="model_matrix_test",
        diversity_report={
            "model_count": 1,
            "eligible_router_child_count": 1,
            "router_representative_children": ["ridge"],
        },
    )
    assert bundle["schema_version"] == "model_matrix_validation_bundle.v2"
    assert "findings_and_recommendations" in bundle
    assert "target_alignment" in bundle
    assert "chronological_split" in bundle
    assert "canonical_lineage" in bundle
    assert "prediction_sanity" in bundle
    assert "elastic_net_degeneracy" in bundle
    assert "reproducibility" in bundle

"""Integration test for the W1 ranking-vs-expression decomposition bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from pysrc.meta.w1_clean_rerun import run_default_w1_clean_rerun


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_run_default_w1_clean_rerun_emits_decomposition_bundle(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    out_dir = tmp_path / "w1_clean_rerun"
    run_default_w1_clean_rerun(output_dir=out_dir, seed=4242, timestamp_utc="2026-04-25T12:00:00Z")

    required = [
        "w1_ranking_vs_expression_decomposition.json",
        "w1_ic_decomposition_by_fold.csv",
        "w1_ic_decomposition_by_regime.csv",
        "w1_turnover_by_score_bucket.csv",
        "w1_expression_transform_comparison.csv",
        "w1_threshold_sweep.csv",
        "w1_top_k_sweep.csv",
        "w1_cost_penalty_sweep.csv",
        "w1_ranking_vs_expression_summary.md",
    ]
    for name in required:
        assert (out_dir / name).is_file(), name

    decomposition = json.loads(
        (out_dir / "w1_ranking_vs_expression_decomposition.json").read_text(encoding="utf-8")
    )
    assert decomposition["schema_version"] == "w1_ranking_vs_expression_decomposition.v1"
    assert decomposition["threshold_percentiles"] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    assert decomposition["top_k_values"] == [1, 3, 5, 10, 15, 20]
    assert decomposition["cost_penalty_lambdas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert decomposition["audit"]["status"] == "PASS"
    report = json.loads(
        (out_dir / "w1_clean_baseline_comparison_report.json").read_text(encoding="utf-8")
    )
    assert report["w1_gate_closure_eligible"] is True
    assert report["model_comparison_decision"] == "NO_CLEAR_WINNER"
    assert report["diagnostic_artifacts"]["ranking_vs_expression_decomposition"] == (
        "w1_ranking_vs_expression_decomposition.json"
    )

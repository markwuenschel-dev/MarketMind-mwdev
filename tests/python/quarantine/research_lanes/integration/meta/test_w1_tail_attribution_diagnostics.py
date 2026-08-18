"""Integration test for the W1 tail-attribution diagnostic bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from pysrc.meta.w1_clean_rerun import run_default_w1_clean_rerun


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_run_default_w1_clean_rerun_emits_tail_attribution_bundle(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    out_dir = tmp_path / "w1_clean_rerun"
    run_default_w1_clean_rerun(output_dir=out_dir, seed=4242, timestamp_utc="2026-04-25T12:00:00Z")

    required = [
        "w1_tail_attribution_diagnostics.json",
        "w1_decile_calibration.csv",
        "w1_top_k_hit_rate.csv",
        "w1_oracle_regret.csv",
        "w1_selected_task_overlap.csv",
        "w1_target_quantile_ic_decomposition.csv",
        "w1_tail_attribution_summary.md",
    ]
    for name in required:
        assert (out_dir / name).is_file(), name

    tail_doc = json.loads(
        (out_dir / "w1_tail_attribution_diagnostics.json").read_text(encoding="utf-8")
    )
    assert tail_doc["target_surfaces"] == ["gross", "net"]
    assert tail_doc["top_k_values"] == [1, 3, 5, 10, 15, 20]
    assert tail_doc["audit"]["status"] == "PASS"

    report = json.loads(
        (out_dir / "w1_clean_baseline_comparison_report.json").read_text(encoding="utf-8")
    )
    assert report["w1_gate_closure_eligible"] is True
    assert report["model_comparison_decision"] == "NO_CLEAR_WINNER"
    assert (
        report["diagnostic_artifacts"]["tail_attribution_diagnostics"]
        == "w1_tail_attribution_diagnostics.json"
    )
    assert report["tail_attribution_audit_status"] == "PASS"

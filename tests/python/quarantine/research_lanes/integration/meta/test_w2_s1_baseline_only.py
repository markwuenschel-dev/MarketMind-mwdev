"""Integration test for the W2-S1 baseline-only dry run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pysrc.meta.w2_s1_baseline_only import W2S1Config, run_w2_s1_baseline_only


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.integration
@pytest.mark.determinism("d1")
@pytest.mark.timeout(240)
def test_run_w2_s1_baseline_only_emits_governed_artifacts(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    pytest.importorskip("xgboost")
    _ = deterministic_seed

    out_dir = tmp_path / "w2_s1"
    result = run_w2_s1_baseline_only(
        W2S1Config(
            output_dir=out_dir,
            max_rebalance_dates=60,
            timestamp_utc="2026-04-26T00:00:00Z",
        )
    )

    assert result["w2_s1_status"] == "BASELINE_ONLY_DRY_RUN_PASS"
    assert result["w2_comparison_allowed"] is False

    required = (
        "w2_opportunity_manifest.json",
        "w2_universe_manifest.json",
        "w2_cost_assumptions.json",
        "w2_walk_forward_splits.json",
        "w2_baseline_predictions.task_level.v1.json",
        "w2_canonical_comparison_table.csv",
        "w2_s1_baseline_metric_report.json",
        "w2_s1_baseline_only_report.json",
        "w2_s1_baseline_only_summary.md",
        "w2_s1_gate_audit.json",
        "w2_s1_next_step_recommendation.md",
        "agent_outputs/agent_1_scope_guard.json",
        "agent_outputs/agent_2_opportunity_manifest_audit.json",
        "agent_outputs/agent_3_pit_feature_audit.json",
        "agent_outputs/agent_3_target_cost_audit.json",
        "agent_outputs/agent_4_split_audit.json",
        "agent_outputs/agent_4_train_eval_overlap_audit.csv",
        "agent_outputs/agent_5_baseline_scoring_audit.json",
        "agent_outputs/agent_6_comparison_table_audit.json",
        "agent_outputs/agent_6_join_audit.csv",
        "agent_outputs/agent_7_metric_audit.json",
    )
    for rel in required:
        assert (out_dir / rel).is_file(), rel

    report = _read_json(out_dir / "w2_s1_baseline_only_report.json")
    assert report["w2_s1_status"] == "BASELINE_ONLY_DRY_RUN_PASS"
    counts = report["counts"]
    assert isinstance(counts, dict)
    assert int(counts["n_instruments"]) == 8
    assert int(counts["n_folds"]) == 4
    assert int(counts["n_eval_opportunities"]) >= 160
    assert int(counts["n_missing_baseline_scores"]) == 0
    assert int(counts["n_missing_query_targets"]) == 0
    assert int(counts["n_join_expansion_alerts"]) == 0
    assert int(counts["n_train_eval_overlap"]) == 0


@pytest.mark.integration
@pytest.mark.determinism("d1")
@pytest.mark.timeout(240)
def test_run_w2_s1_baseline_only_is_deterministic_on_same_inputs(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    pytest.importorskip("xgboost")
    _ = deterministic_seed

    cfg = W2S1Config(
        max_rebalance_dates=40,
        timestamp_utc="2026-04-26T00:00:00Z",
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_w2_s1_baseline_only(replace(cfg, output_dir=first_dir))
    second = run_w2_s1_baseline_only(replace(cfg, output_dir=second_dir))

    assert first["w2_s1_status"] == "BASELINE_ONLY_DRY_RUN_PASS"
    assert second["w2_s1_status"] == "BASELINE_ONLY_DRY_RUN_PASS"

    first_report = _read_json(first_dir / "w2_s1_baseline_only_report.json")
    second_report = _read_json(second_dir / "w2_s1_baseline_only_report.json")
    assert first_report["counts"] == second_report["counts"]
    assert first_report["baseline_metrics"] == second_report["baseline_metrics"]

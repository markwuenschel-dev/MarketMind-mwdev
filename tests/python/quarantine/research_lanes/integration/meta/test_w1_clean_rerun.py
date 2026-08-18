"""Integration test for the clean W1 evaluation-surface rerun."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from pysrc.meta.w1_clean_rerun import run_default_w1_clean_rerun


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_run_default_w1_clean_rerun_emits_clean_surface(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    out_dir = tmp_path / "w1_clean_rerun"
    run_default_w1_clean_rerun(output_dir=out_dir, seed=4242, timestamp_utc="2026-04-25T12:00:00Z")

    required = [
        "w1_task_level_comparison_table.csv",
        "w1_walk_forward_splits.json",
        "w1_baseline_predictions.task_level.v1.json",
        "w1_challenger_predictions.task_level.v1.json",
        "w1_clean_task_level_metrics.json",
        "w1_clean_baseline_comparison_report.json",
        "w1_clean_baseline_comparison_summary.md",
        "w1_clean_gate_audit.json",
        "w1_clean_supersession_note.md",
    ]
    for name in required:
        assert (out_dir / name).is_file(), name

    agent_required = [
        "agent_1_task_universe.csv",
        "agent_1_task_universe_audit.json",
        "agent_2_walk_forward_splits.json",
        "agent_2_split_audit.json",
        "agent_3_baseline_predictions.task_level.v1.json",
        "agent_3_baseline_scoring_audit.json",
        "agent_4_challenger_predictions.task_level.v1.json",
        "agent_4_challenger_scoring_audit.json",
        "agent_5_comparison_table_audit.json",
        "agent_5_join_audit.csv",
        "agent_6_metric_audit.json",
    ]
    for name in agent_required:
        assert (out_dir / "agent_outputs" / name).is_file(), name

    report = json.loads(
        (out_dir / "w1_clean_baseline_comparison_report.json").read_text(encoding="utf-8")
    )
    assert report["supersedes"]["prior_run_disposition"] == "SUPERSEDED_INVALID_EVAL_ALIGNMENT"
    assert report["w1_gate_closure_eligible"] is True
    assert report["counts"]["n_final_eval_rows"] == 100
    assert report["counts"]["n_unique_eval_task_ids"] == 100
    assert report["counts"]["n_missing_baseline_scores"] == 0
    assert report["counts"]["n_missing_challenger_scores"] == 0
    assert report["counts"]["n_train_eval_overlap"] == 0
    assert report["counts"]["n_duplicate_eval_task_ids_within_fold"] == 0

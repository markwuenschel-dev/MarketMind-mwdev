"""Unit tests for OI-59 Branch B packing-semantics task-definition proposal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _config_with_geometry,
    _read_json,
    _write_fixture_bundle,
)
from tests.python.unit.meta.test_rg09_oi59_episode_construction import (
    _baseline_variant_b_payload,
    _multi_window_recovery_frame,
    _structural_failure_frame,
    _transition_anchor_recovery_frame,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_task_definition_always_runs_b1_and_b2(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_task_definition import build_oi59_task_definition_report

    frame = _transition_anchor_recovery_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=4,
            min_query_rows=4,
            label_horizon_bars=2,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=1,
        ),
    )
    baseline_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(baseline_report_path, _baseline_variant_b_payload())
    report = build_oi59_task_definition_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        experiment2_report_path=baseline_report_path,
        output_dir=tmp_path / "out",
    )
    assert report["proposal_class"] == "packing_semantics_redesign"
    assert len(report["variants"]) == 2
    assert {report["variants"][0]["variant_id"], report["variants"][1]["variant_id"]} == {
        "B1_tail_greedy_non_overlapping_pack",
        "B2_head_greedy_non_overlapping_pack",
    }
    b1 = next(
        v for v in report["variants"] if v["variant_id"] == "B1_tail_greedy_non_overlapping_pack"
    )
    assert b1["construction_manifest"]["construction_mode"] == "tail_greedy_non_overlapping_pack"
    assert b1["construction_manifest"]["selection_order"] == "descending_start"
    assert report["recommended_variant"] in (
        "B1_tail_greedy_non_overlapping_pack",
        "B2_head_greedy_non_overlapping_pack",
    )
    best_row = next(
        v for v in report["variants"] if v["variant_id"] == report["recommended_variant"]
    )
    assert report["overall_classification"] == best_row["classification"]
    best_id = str(report["experiment5_branch_b_evaluation"]["best_variant_by_metrics"])
    assert best_id == report["recommended_variant"]
    assert (
        report["branch_b_signal"]
        == report["experiment5_branch_b_evaluation"]["materiality_assessment"]
    )
    assert report["branch_b_signal"] in ("NO_EFFECT", "WEAK_EFFECT", "STRONG_EFFECT")
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert report["threshold_references"]["recovery_ratio_floor"]["state"] == "PROVISIONAL"
    assert "decision_output" not in report
    gap = report["experiment5_branch_b_evaluation"]["performance_gap_vs_baseline"]
    assert "feasible_task_ratio_delta" in gap
    assert "admissible_delta" in gap
    assert "horizon_overlap_delta" in gap
    assert (tmp_path / "out" / "oi59_task_definition_report.json").exists()
    assert (tmp_path / "out" / "oi59_task_definition_diagnostic.md").exists()
    assert (tmp_path / "out" / "task_manifest.json").exists()
    assert (tmp_path / "out" / "meta_validity_report.json").exists()
    assert (tmp_path / "out" / "execution_assumptions.json").exists()


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_branch_b_materiality_assessment_thresholds() -> None:
    from pysrc.meta.rg09_oi59_task_definition import _materiality_assessment

    assert (
        _materiality_assessment(
            delta_feasible_task_ratio=0.005,
            baseline_outcome="PARTIAL_RECOVERY",
            best_classification="PARTIAL_RECOVERY",
        )
        == "NO_EFFECT"
    )
    assert (
        _materiality_assessment(
            delta_feasible_task_ratio=0.02,
            baseline_outcome="PARTIAL_RECOVERY",
            best_classification="PARTIAL_RECOVERY",
        )
        == "WEAK_EFFECT"
    )
    assert (
        _materiality_assessment(
            delta_feasible_task_ratio=0.04,
            baseline_outcome="PARTIAL_RECOVERY",
            best_classification="PARTIAL_RECOVERY",
        )
        == "STRONG_EFFECT"
    )
    assert (
        _materiality_assessment(
            delta_feasible_task_ratio=0.0,
            baseline_outcome="PARTIAL_RECOVERY",
            best_classification="RECOVERED_FOR_NEXT_STAGE",
        )
        == "STRONG_EFFECT"
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_task_definition_partial_recovery_reports_experiment5(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_task_definition import build_oi59_task_definition_report

    frame = _multi_window_recovery_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=4,
            min_query_rows=4,
            label_horizon_bars=2,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=50,
        ),
    )
    baseline_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(baseline_report_path, _baseline_variant_b_payload())
    report = build_oi59_task_definition_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        experiment2_report_path=baseline_report_path,
        output_dir=tmp_path / "out",
    )
    assert report["overall_classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert len(report["variants"]) == 2
    assert report["variants"][0]["variant_id"] == "B1_tail_greedy_non_overlapping_pack"
    assert report["variants"][1]["variant_id"] == "B2_head_greedy_non_overlapping_pack"
    assert (
        report["variants"][1]["construction_manifest"]["construction_mode"]
        == "head_greedy_non_overlapping_pack"
    )
    assert report["branch_b_signal"] in ("NO_EFFECT", "WEAK_EFFECT", "STRONG_EFFECT")
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert "decision_output" not in report


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_task_definition_structural_failure(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_task_definition import build_oi59_task_definition_report

    frame = _structural_failure_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=4,
            min_query_rows=4,
            label_horizon_bars=2,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=2,
        ),
    )
    baseline_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(baseline_report_path, _baseline_variant_b_payload())
    report = build_oi59_task_definition_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        experiment2_report_path=baseline_report_path,
        output_dir=tmp_path / "out",
    )
    assert report["overall_classification"] == "FAIL_STRUCTURAL"
    assert report["phase2_result"] == "FAIL_STRUCTURAL"
    assert report["recommended_variant"] in (
        "B1_tail_greedy_non_overlapping_pack",
        "B2_head_greedy_non_overlapping_pack",
    )
    assert report["branch_b_signal"] in ("NO_EFFECT", "WEAK_EFFECT", "STRONG_EFFECT")
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert "decision_output" not in report
    written = _read_json(tmp_path / "out" / "oi59_task_definition_report.json")
    assert written["overall_classification"] == "FAIL_STRUCTURAL"

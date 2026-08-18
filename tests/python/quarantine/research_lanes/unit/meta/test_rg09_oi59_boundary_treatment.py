"""Unit tests for OI-59 Experiment 3 boundary-treatment redesign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _config_with_geometry,
    _read_json,
    _write_fixture_bundle,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _baseline_variant_b_payload() -> dict[str, Any]:
    return {
        "schema_version": "rg09.oi59_segmentation_redesign.v1",
        "surface": "corrected_multi_instrument",
        "recommended_variant": "B_regime_class_compatible_merge",
        "overall_classification": "PARTIAL_RECOVERY",
        "phase2_result": "INSUFFICIENT",
        "variants": [
            {
                "variant_id": "B_regime_class_compatible_merge",
                "classification": "PARTIAL_RECOVERY",
                "segment_stats": {
                    "count": 2638,
                    "feasible_count": 68,
                    "feasible_segment_ratio": 0.02577710386656558,
                },
                "episode_stats": {
                    "admissible": 54,
                    "total_candidates": 2638,
                    "rejection_breakdown": {
                        "HORIZON_OVERLAP": 2543,
                        "INSUFFICIENT_LENGTH": 35,
                        "OTHER": 6,
                    },
                },
                "delta_vs_baseline": {
                    "horizon_overlap_reduction_pct": 94.31439622599325,
                },
                "segment_window": {
                    "start_ts": "2024-01-01T00:00:00+00:00",
                    "end_ts": "2024-01-20T00:00:00+00:00",
                },
            }
        ],
    }


def _boundary_recovery_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-01T00:00:00+00:00")
    specs = [
        ("trend_hi__vol_lo__bocpd_stable", "bull", 4),
        ("trend_hi__vol_hi__bocpd_stable", "bull", 1),
        ("trend_hi__vol_lo__bocpd_stable", "bull", 4),
        ("trend_hi__vol_hi__bocpd_stable", "bull", 1),
        ("trend_hi__vol_lo__bocpd_stable", "bull", 4),
    ]
    day = 0
    state_idx = 0
    for regime_id, regime_class, length in specs:
        for _ in range(length):
            ts = base + pd.Timedelta(days=day)
            rows.append(
                {
                    "entity_id": "ES",
                    "decision_ts": ts,
                    "effective_at": ts,
                    "regime_id": regime_id,
                    "regime_label": regime_id,
                    "regime_class": regime_class,
                    "boundary_flag": "transition" if "transition" in regime_id else "stable",
                    "change_probability": 0.05,
                    "diag_regime_class_bocpd_gated": regime_class,
                    "run_length_mode": float(state_idx + 1),
                    "run_length_expectation": float(state_idx + 1),
                    "transition_probability": 0.05,
                    "posterior_entropy": 0.01,
                    "trend_score_raw": 0.2,
                    "vol_score_raw": 0.1 if regime_class == "bull" else 0.9,
                    "config_version": "rg09_v1.0.2",
                    "state_snapshot_id": f"sha256:state-{state_idx}",
                    "input_snapshot_id": "sha256:input",
                    "rg09_trading_day_ord": day,
                }
            )
            state_idx += 1
            day += 1
    return pd.DataFrame(rows)


def _incompatible_boundary_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-01T00:00:00+00:00")
    specs = [
        ("trend_hi__vol_lo__bocpd_stable", "bull", 2),
        ("trend_lo__vol_hi__bocpd_transition", "crisis", 2),
        ("trend_flat__vol_med__bocpd_transition", "sideways", 2),
    ]
    day = 0
    state_idx = 0
    for regime_id, regime_class, length in specs:
        for _ in range(length):
            ts = base + pd.Timedelta(days=day)
            rows.append(
                {
                    "entity_id": "ES",
                    "decision_ts": ts,
                    "effective_at": ts,
                    "regime_id": regime_id,
                    "regime_label": regime_id,
                    "regime_class": regime_class,
                    "boundary_flag": "transition",
                    "change_probability": 0.05,
                    "diag_regime_class_bocpd_gated": regime_class,
                    "run_length_mode": float(state_idx + 1),
                    "run_length_expectation": float(state_idx + 1),
                    "transition_probability": 0.05,
                    "posterior_entropy": 0.01,
                    "trend_score_raw": 0.2,
                    "vol_score_raw": 0.1 + (0.1 * state_idx),
                    "config_version": "rg09_v1.0.2",
                    "state_snapshot_id": f"sha256:state-{state_idx}",
                    "input_snapshot_id": "sha256:input",
                    "rg09_trading_day_ord": day,
                }
            )
            state_idx += 1
            day += 1
    return pd.DataFrame(rows)


def _same_class_incompatible_regime_id_boundary_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-01T00:00:00+00:00")
    specs = [
        ("trend_hi__vol_lo__bocpd_stable", "bull", 4),
        ("trend_hi__vol_hi__bocpd_transition", "bull", 2),
        ("trend_hi__vol_lo__other_suffix", "bull", 4),
    ]
    day = 0
    state_idx = 0
    for regime_id, regime_class, length in specs:
        for _ in range(length):
            ts = base + pd.Timedelta(days=day)
            rows.append(
                {
                    "entity_id": "ES",
                    "decision_ts": ts,
                    "effective_at": ts,
                    "regime_id": regime_id,
                    "regime_label": regime_id,
                    "regime_class": regime_class,
                    "boundary_flag": "transition" if "transition" in regime_id else "stable",
                    "change_probability": 0.05,
                    "diag_regime_class_bocpd_gated": regime_class,
                    "run_length_mode": float(state_idx + 1),
                    "run_length_expectation": float(state_idx + 1),
                    "transition_probability": 0.05,
                    "posterior_entropy": 0.01,
                    "trend_score_raw": 0.2,
                    "vol_score_raw": 0.1,
                    "config_version": "rg09_v1.0.2",
                    "state_snapshot_id": f"sha256:state-{state_idx}",
                    "input_snapshot_id": "sha256:input",
                    "rg09_trading_day_ord": day,
                }
            )
            state_idx += 1
            day += 1
    return pd.DataFrame(rows)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_boundary_treatment_reports_recovered_variant_from_variant_b_baseline(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_boundary_treatment import build_oi59_boundary_treatment_report

    # Low-threshold mechanics fixture: with THR-RG09-V20 still PROVISIONAL and gate-critical,
    # classification is held pending threshold governance instead of recovered.
    frame = _boundary_recovery_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=1,
        ),
    )
    baseline_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(baseline_report_path, _baseline_variant_b_payload())
    report = build_oi59_boundary_treatment_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        experiment2_report_path=baseline_report_path,
        output_dir=tmp_path / "out",
    )
    assert report["selected_baseline_variant"] == "B_regime_class_compatible_merge"
    assert report["overall_classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert report["phase2_result"] == "INSUFFICIENT"
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert report["decision_output"] == "hold_pending_threshold_review"
    assert [variant["variant_id"] for variant in report["variants"]] == [
        "BT-1_conservative_boundary_recovery",
        "BT-2_moderate_boundary_recovery",
        "BT-3_boundary_plus_label_plumbing_check",
    ]
    assert report["recommended_variant"] == "BT-1_conservative_boundary_recovery"
    bt1 = report["variants"][0]
    assert bt1["classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert (
        bt1["delta_vs_experiment2_variant_b"]["horizon_overlap_reduction_pct_vs_experiment1"]
        >= 50.0
    )
    assert (tmp_path / "out" / "task_manifest.json").exists()
    assert (tmp_path / "out" / "meta_validity_report.json").exists()
    assert (tmp_path / "out" / "execution_assumptions.json").exists()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_boundary_treatment_stops_when_boundary_variants_do_not_recover_constructibility(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_boundary_treatment import build_oi59_boundary_treatment_report

    frame = _incompatible_boundary_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=2,
        ),
    )
    baseline_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(baseline_report_path, _baseline_variant_b_payload())
    report = build_oi59_boundary_treatment_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        experiment2_report_path=baseline_report_path,
        output_dir=tmp_path / "out",
    )
    assert report["overall_classification"] == "FAIL_STRUCTURAL"
    assert report["phase2_result"] == "FAIL_STRUCTURAL"
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert report["recommended_variant"] is None
    assert (
        report["decision_output"]
        == "stop_corrected_branch_recovery_and_escalate_task_definition_change"
    )
    written = _read_json(tmp_path / "out" / "oi59_boundary_treatment_report.json")
    assert written["overall_classification"] == "FAIL_STRUCTURAL"
    diagnostic = (tmp_path / "out" / "oi59_boundary_treatment_diagnostic.md").read_text(
        encoding="utf-8"
    )
    assert "Overall classification: `FAIL_STRUCTURAL`" in diagnostic


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_boundary_treatment_does_not_merge_same_class_incompatible_regime_ids(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_boundary_treatment import _collapse_short_boundary_segments
    from pysrc.meta.rg09_oi59_segmentation_redesign import _variant_groups

    frame = _same_class_incompatible_regime_id_boundary_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=1,
        ),
    )
    from pysrc.meta.rg09_harness import (
        _load_and_validate_fixture,
        build_rg09_candidate_episode_groups,
        load_rg09_config,
    )

    config = load_rg09_config(paths["config"])
    bundle, _fixture_validation_fail_codes = _load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        expected_temporal_folds=config.min_temporal_folds,
    )
    baseline_groups = build_rg09_candidate_episode_groups(
        bundle.frame, config, boundary_recovery=None
    )
    variant_groups, _compatibility_rule = _variant_groups(baseline_groups)
    variant_b_groups = variant_groups["B_regime_class_compatible_merge"]
    collapsed_groups, manifest = _collapse_short_boundary_segments(
        variant_b_groups, max_boundary_bars=3
    )

    assert manifest["boundary_merge_events"] == 0
    assert len(collapsed_groups) == len(variant_b_groups)

"""Unit tests for OI-59 Experiment 4 episode-construction redesign."""

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


def _row(
    ts: pd.Timestamp,
    *,
    entity_id: str,
    regime_id: str,
    regime_class: str,
    boundary_flag: str = "stable",
    change_probability: float = 0.05,
    vol_score_raw: float = 0.1,
    trading_day_ord: int,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "decision_ts": ts,
        "effective_at": ts,
        "regime_id": regime_id,
        "regime_label": regime_id,
        "regime_class": regime_class,
        "boundary_flag": boundary_flag,
        "change_probability": change_probability,
        "diag_regime_class_bocpd_gated": regime_class,
        "run_length_mode": float(trading_day_ord + 1),
        "run_length_expectation": float(trading_day_ord + 1),
        "transition_probability": 0.05,
        "posterior_entropy": 0.01,
        "trend_score_raw": 0.2,
        "vol_score_raw": vol_score_raw,
        "config_version": "rg09_v1.0.2",
        "state_snapshot_id": f"sha256:state-{entity_id}-{trading_day_ord}",
        "input_snapshot_id": "sha256:input",
        "rg09_trading_day_ord": trading_day_ord,
    }


def _transition_anchor_recovery_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-01T00:00:00+00:00")
    specs = [
        ("trend_hi__vol_lo__bocpd_stable", "bull", 12, 0.10),
        ("trend_lo__vol_lo__bocpd_stable", "bear", 8, 0.30),
        ("trend_hi__vol_lo__bocpd_stable", "bull", 12, 0.12),
    ]
    day = 0
    for regime_id, regime_class, length, vol in specs:
        for _ in range(length):
            rows.append(
                _row(
                    base + pd.Timedelta(days=day),
                    entity_id="ES",
                    regime_id=regime_id,
                    regime_class=regime_class,
                    trading_day_ord=day,
                    vol_score_raw=vol,
                )
            )
            day += 1
    return pd.DataFrame(rows)


def _multi_window_recovery_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-02-01T00:00:00+00:00")
    for day in range(36):
        rows.append(
            _row(
                base + pd.Timedelta(days=day),
                entity_id="ES",
                regime_id="trend_hi__vol_lo__bocpd_stable",
                regime_class="bull",
                trading_day_ord=day,
                vol_score_raw=0.15,
            )
        )
    return pd.DataFrame(rows)


def _structural_failure_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-03-01T00:00:00+00:00")
    specs = [
        ("trend_hi__vol_lo__bocpd_stable", "bull", 3),
        ("trend_lo__vol_lo__bocpd_stable", "bear", 2),
        ("trend_hi__vol_lo__bocpd_stable", "bull", 3),
    ]
    day = 0
    for regime_id, regime_class, length in specs:
        for _ in range(length):
            rows.append(
                _row(
                    base + pd.Timedelta(days=day),
                    entity_id="ES",
                    regime_id=regime_id,
                    regime_class=regime_class,
                    trading_day_ord=day,
                    vol_score_raw=0.2,
                )
            )
            day += 1
    return pd.DataFrame(rows)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_episode_construction_prefers_transition_anchored_variant_when_it_is_least_permissive_success(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_episode_construction import build_oi59_episode_construction_report

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
    report = build_oi59_episode_construction_report(
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
    assert report["recommended_variant"] == "E4-A_transition_anchored_corrected_episodes"
    e4a = report["variants"][0]
    assert e4a["classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert e4a["task_stats"]["admissible"] >= 1
    assert e4a["task_stats"]["feasible_task_ratio"] == pytest.approx(1.0)
    assert (tmp_path / "out" / "task_manifest.json").exists()
    assert (tmp_path / "out" / "meta_validity_report.json").exists()
    assert (tmp_path / "out" / "execution_assumptions.json").exists()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_episode_construction_uses_multi_episode_packing_when_transition_anchors_cannot_recover(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_episode_construction import build_oi59_episode_construction_report

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
            min_admissible_episode_count=2,
        ),
    )
    baseline_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(baseline_report_path, _baseline_variant_b_payload())
    report = build_oi59_episode_construction_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        experiment2_report_path=baseline_report_path,
        output_dir=tmp_path / "out",
    )
    assert report["overall_classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert report["recommended_variant"] == "E4-B_multi_episode_packing_within_corrected_spans"
    e4a = report["variants"][0]
    e4b = report["variants"][1]
    assert e4a["task_stats"]["admissible"] == 0
    assert e4b["classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert e4b["task_stats"]["admissible"] >= 2
    assert e4b["task_stats"]["feasible_task_ratio"] >= 0.10


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_episode_construction_fails_structural_when_all_task_definition_variants_remain_infeasible(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_oi59_episode_construction import build_oi59_episode_construction_report

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
    report = build_oi59_episode_construction_report(
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
        == "stop_corrected_branch_recovery_and_escalate_program_assumption"
    )
    written = _read_json(tmp_path / "out" / "oi59_episode_construction_report.json")
    assert written["overall_classification"] == "FAIL_STRUCTURAL"
    diagnostic = (tmp_path / "out" / "oi59_episode_construction_diagnostic.md").read_text(
        encoding="utf-8"
    )
    assert "Overall classification: `FAIL_STRUCTURAL`" in diagnostic

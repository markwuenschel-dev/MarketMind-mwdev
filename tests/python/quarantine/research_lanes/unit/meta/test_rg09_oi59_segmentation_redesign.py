"""Unit tests for OI-59 Experiment 2 segmentation redesign."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _config_with_geometry,
    _read_json,
    _write_fixture_bundle,
)

from pysrc.meta.rg09_oi59_feasibility_audit import build_oi59_feasibility_audit
from pysrc.meta.rg09_oi59_segmentation_redesign import build_oi59_segmentation_redesign_report


def _interleaved_two_entity_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ts0 = pd.Timestamp("2024-01-01T00:00:00+00:00")
    for day in range(6):
        base = ts0 + pd.Timedelta(days=day)
        rows.append(
            {
                "entity_id": "ES",
                "decision_ts": base,
                "effective_at": base,
                "regime_id": "trend_hi__vol_lo__bocpd_stable",
                "regime_label": "trend_hi__vol_lo__bocpd_stable",
                "regime_class": "bull",
                "boundary_flag": "stable",
                "change_probability": 0.05,
                "diag_regime_class_bocpd_gated": "bull",
                "run_length_mode": float(day + 1),
                "run_length_expectation": float(day + 1),
                "transition_probability": 0.05,
                "posterior_entropy": 0.01,
                "trend_score_raw": 0.2,
                "vol_score_raw": 0.1 + (0.01 * day),
                "config_version": "rg09_v1.0.2",
                "state_snapshot_id": f"sha256:state-es-{day}",
                "input_snapshot_id": "sha256:input",
                "rg09_trading_day_ord": day,
            }
        )
        rows.append(
            {
                "entity_id": "NQ",
                "decision_ts": base + pd.Timedelta(nanoseconds=1),
                "effective_at": base + pd.Timedelta(nanoseconds=1),
                "regime_id": "trend_hi__vol_lo__bocpd_stable",
                "regime_label": "trend_hi__vol_lo__bocpd_stable",
                "regime_class": "bull",
                "boundary_flag": "stable",
                "change_probability": 0.05,
                "diag_regime_class_bocpd_gated": "bull",
                "run_length_mode": float(day + 1),
                "run_length_expectation": float(day + 1),
                "transition_probability": 0.05,
                "posterior_entropy": 0.01,
                "trend_score_raw": 0.2,
                "vol_score_raw": 0.2 + (0.01 * day),
                "config_version": "rg09_v1.0.2",
                "state_snapshot_id": f"sha256:state-nq-{day}",
                "input_snapshot_id": "sha256:input",
                "rg09_trading_day_ord": day,
            }
        )
    return pd.DataFrame(rows)


def _incompatible_single_bar_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ts0 = pd.Timestamp("2024-01-01T00:00:00+00:00")
    regime_ids = [
        ("trend_hi__vol_lo__bocpd_stable", "bull"),
        ("trend_lo__vol_hi__bocpd_stable", "crisis"),
        ("trend_flat__vol_med__bocpd_stable", "sideways"),
    ]
    for idx, (regime_id, regime_class) in enumerate(regime_ids):
        ts = ts0 + pd.Timedelta(days=idx)
        rows.append(
            {
                "entity_id": "ES",
                "decision_ts": ts,
                "effective_at": ts,
                "regime_id": regime_id,
                "regime_label": regime_id,
                "regime_class": regime_class,
                "boundary_flag": "stable",
                "change_probability": 0.05,
                "diag_regime_class_bocpd_gated": regime_class,
                "run_length_mode": 1.0,
                "run_length_expectation": 1.0,
                "transition_probability": 0.05,
                "posterior_entropy": 0.01,
                "trend_score_raw": 0.2,
                "vol_score_raw": 0.1,
                "config_version": "rg09_v1.0.2",
                "state_snapshot_id": f"sha256:state-{idx}",
                "input_snapshot_id": "sha256:input",
                "rg09_trading_day_ord": idx,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_segmentation_redesign_reports_partial_recovery_for_entity_local_exact_identity_merge(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _interleaved_two_entity_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=30,
        ),
    )
    baseline_json = tmp_path / "oi59_feasibility_audit.json"
    baseline_md = tmp_path / "oi59_feasibility_diagnostic.md"
    build_oi59_feasibility_audit(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_json_path=baseline_json,
        output_markdown_path=baseline_md,
    )
    report = build_oi59_segmentation_redesign_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        baseline_reference_path=baseline_json,
        output_dir=tmp_path / "out",
    )
    assert report["overall_classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert report["phase2_result"] == "INSUFFICIENT"
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert report["threshold_references"]["recovery_ratio_floor"]["state"] == "PROVISIONAL"
    variant_a = next(v for v in report["variants"] if v["variant_id"] == "A_exact_identity_merge")
    assert variant_a["classification"] == "HOLD_PENDING_THRESHOLD_REVIEW"
    assert variant_a["segment_stats"]["count"] == 2
    assert variant_a["segment_stats"]["feasible_segment_ratio"] == pytest.approx(1.0)
    assert variant_a["episode_stats"]["admissible"] == 2
    assert variant_a["delta_vs_baseline"]["horizon_overlap_reduction_pct"] == pytest.approx(100.0)
    assert (tmp_path / "out" / "task_manifest.json").exists()
    assert (tmp_path / "out" / "meta_validity_report.json").exists()
    assert (tmp_path / "out" / "execution_assumptions.json").exists()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_segmentation_redesign_classifies_fail_structural_when_no_variant_recovers_constructibility(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _incompatible_single_bar_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
            min_admissible_episode_count=30,
        ),
    )
    baseline_json = tmp_path / "oi59_feasibility_audit.json"
    baseline_md = tmp_path / "oi59_feasibility_diagnostic.md"
    build_oi59_feasibility_audit(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_json_path=baseline_json,
        output_markdown_path=baseline_md,
    )
    report = build_oi59_segmentation_redesign_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        baseline_reference_path=baseline_json,
        output_dir=tmp_path / "out",
    )
    assert report["overall_classification"] == "FAIL_STRUCTURAL"
    assert report["phase2_result"] == "FAIL_STRUCTURAL"
    assert report["threshold_references"]["recovery_ratio_floor"]["threshold_id"] == "THR-RG09-V20"
    assert report["recommended_variant"] is None
    written = _read_json(tmp_path / "out" / "oi59_segmentation_redesign_report.json")
    assert written["overall_classification"] == "FAIL_STRUCTURAL"
    diagnostic = (tmp_path / "out" / "oi59_segmentation_redesign_diagnostic.md").read_text(
        encoding="utf-8"
    )
    assert "Overall classification: `FAIL_STRUCTURAL`" in diagnostic

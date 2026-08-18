"""Unit tests for the OI-59 feasibility audit artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _config_with_geometry,
    _read_json,
    _single_episode_frame,
    _three_episode_frame,
    _write_fixture_bundle,
)

from pysrc.meta.rg09_oi59_feasibility_audit import build_oi59_feasibility_audit


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_audit_reports_feasible_and_admissible_ratios(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _single_episode_frame(length=8)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
        ),
    )
    json_path = tmp_path / "oi59_feasibility_audit.json"
    md_path = tmp_path / "oi59_feasibility_diagnostic.md"
    report = build_oi59_feasibility_audit(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_json_path=json_path,
        output_markdown_path=md_path,
    )
    assert report["Lmin"] == 6
    assert report["builder_geometry_floor"] == 5
    assert report["segment_stats"]["count"] == 1
    assert report["segment_stats"]["feasible_count"] == 1
    assert report["segment_stats"]["feasible_ratio"] == pytest.approx(1.0)
    assert report["episode_stats"]["total_candidates"] == 1
    assert report["episode_stats"]["admissible"] == 1
    assert report["episode_stats"]["rejection_breakdown"] == {
        "HORIZON_OVERLAP": 0,
        "INSUFFICIENT_LENGTH": 0,
        "OTHER": 0,
    }
    written = _read_json(json_path)
    assert written["ratio_summary"]["admissible_episode_ratio"] == pytest.approx(1.0)
    diagnostic_text = md_path.read_text(encoding="utf-8")
    assert "Classification: `UNEXPECTED`" in diagnostic_text


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_oi59_audit_classifies_structural_failure_when_segments_are_theoretically_infeasible(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _three_episode_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_temporal_folds=1,
        ),
    )
    json_path = tmp_path / "oi59_feasibility_audit.json"
    md_path = tmp_path / "oi59_feasibility_diagnostic.md"
    report = build_oi59_feasibility_audit(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_json_path=json_path,
        output_markdown_path=md_path,
    )
    assert report["segment_stats"]["count"] == 3
    assert report["segment_stats"]["feasible_count"] == 0
    assert report["segment_stats"]["feasible_ratio"] == pytest.approx(0.0)
    assert report["episode_stats"]["admissible"] == 0
    assert report["episode_stats"]["rejection_breakdown"]["HORIZON_OVERLAP"] == 3
    assert report["ratio_summary"]["horizon_overlap_rejection_pct"] == pytest.approx(100.0)
    assert report["classification"] == "FAIL_STRUCTURAL"
    diagnostic_text = md_path.read_text(encoding="utf-8")
    assert "Is admissibility = 0? `yes`" in diagnostic_text
    assert "Interpretation: segmentation is geometrically incompatible" in diagnostic_text

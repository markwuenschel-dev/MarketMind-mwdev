"""Integration test for the OI-59 Branch B task-definition CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


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


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_run_oi59_task_definition_emits_required_artifacts(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from scripts.run_oi59_task_definition import run_oi59_task_definition

    experiment2_report_path = tmp_path / "oi59_segmentation_redesign_report.json"
    _write_json(experiment2_report_path, _baseline_variant_b_payload())
    report = run_oi59_task_definition(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        experiment2_report_path=experiment2_report_path,
        output_dir=tmp_path / "out",
    )
    assert (tmp_path / "out" / "oi59_task_definition_report.json").exists()
    assert (tmp_path / "out" / "oi59_task_definition_diagnostic.md").exists()
    assert (tmp_path / "out" / "task_manifest.json").exists()
    assert (tmp_path / "out" / "meta_validity_report.json").exists()
    assert (tmp_path / "out" / "execution_assumptions.json").exists()
    written = _read_json(tmp_path / "out" / "oi59_task_definition_report.json")
    assert written["fixture_sha256"] == report["fixture_sha256"]
    assert written["proposal_class"] == "packing_semantics_redesign"
    assert written["schema_version"] == "rg09.oi59_task_definition.v2"
    assert written["branch_b_signal"] in ("NO_EFFECT", "WEAK_EFFECT", "STRONG_EFFECT")
    assert "experiment5_branch_b_evaluation" in written
    assert "decision_output" not in written

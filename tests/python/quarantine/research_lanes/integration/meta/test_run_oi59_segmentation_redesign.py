"""Integration test for the OI-59 Experiment 2 segmentation redesign CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_run_oi59_segmentation_redesign_emits_required_artifacts(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from scripts.run_oi59_feasibility_audit import run_oi59_feasibility_audit
    from scripts.run_oi59_segmentation_redesign import run_oi59_segmentation_redesign

    baseline_json = tmp_path / "oi59_feasibility_audit.json"
    baseline_md = tmp_path / "oi59_feasibility_diagnostic.md"
    run_oi59_feasibility_audit(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        output_json_path=baseline_json,
        output_markdown_path=baseline_md,
    )
    report = run_oi59_segmentation_redesign(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        baseline_reference_path=baseline_json,
        output_dir=tmp_path / "out",
    )
    assert (tmp_path / "out" / "oi59_segmentation_redesign_report.json").exists()
    assert (tmp_path / "out" / "oi59_segmentation_redesign_diagnostic.md").exists()
    assert (tmp_path / "out" / "task_manifest.json").exists()
    assert (tmp_path / "out" / "meta_validity_report.json").exists()
    assert (tmp_path / "out" / "execution_assumptions.json").exists()
    written = _read_json(tmp_path / "out" / "oi59_segmentation_redesign_report.json")
    assert written["fixture_sha256"] == report["fixture_sha256"]
    assert "variants" in written
    assert "overall_classification" in written

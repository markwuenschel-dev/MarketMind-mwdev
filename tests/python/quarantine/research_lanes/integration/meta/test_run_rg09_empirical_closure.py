"""Integration tests for scripts.run_rg09_empirical_closure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_cli_rg09_empirical_closure_fixture_v1(tmp_path: Path, deterministic_seed: int) -> None:
    from scripts.run_rg09_empirical_closure import run_rg09_empirical_closure_cli

    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "rg09_ec"
    code = run_rg09_empirical_closure_cli(
        fixture_path=root / "fixtures/rg09/v1/rg09_fixture_v1.parquet",
        fixture_summary_path=root / "fixtures/rg09/v1/rg09_fixture_summary.json",
        fixture_metadata_path=root / "fixtures/rg09/v1/rg09_fixture_metadata.json",
        config_path=root / "docs/rg09/rg09_pilot_config_v1.json",
        output_dir=out,
    )
    assert code in (0, 1)
    summary_path = out / "rg09_empirical_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["non_promotable"] is True
    assert (out / "rg09_empirical_report.md").exists()
    assert (out / "machine_manifest.json").exists()
    assert (out / "meta_validity_report_research.json").exists()
    assert (out / "task_manifest_research.json").exists()
    tm = json.loads((out / "rg09_surface_manifest.json").read_text(encoding="utf-8"))
    assert "research_schema_version" not in tm
    mm = json.loads((out / "machine_manifest.json").read_text(encoding="utf-8"))
    assert "machine_manifest.json" not in mm.get("empirical_research_output_filenames", [])
    assert mm.get("wrapper_manifest_filename") == "machine_manifest.json"

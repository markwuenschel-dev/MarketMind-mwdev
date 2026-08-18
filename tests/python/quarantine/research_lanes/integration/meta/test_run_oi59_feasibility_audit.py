"""Integration test for the OI-59 feasibility audit CLI."""

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
def test_run_oi59_feasibility_audit_emits_required_artifacts(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from scripts.run_oi59_feasibility_audit import run_oi59_feasibility_audit

    json_path = tmp_path / "oi59_feasibility_audit.json"
    md_path = tmp_path / "oi59_feasibility_diagnostic.md"
    report = run_oi59_feasibility_audit(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        output_json_path=json_path,
        output_markdown_path=md_path,
    )
    assert json_path.exists()
    assert md_path.exists()
    written = _read_json(json_path)
    assert written["fixture_sha256"] == report["fixture_sha256"]
    assert "segment_stats" in written
    assert "episode_stats" in written
    assert "ratio_summary" in written

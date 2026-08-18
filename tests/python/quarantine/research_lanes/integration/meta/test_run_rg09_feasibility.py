"""Integration tests for the RG-09 feasibility CLI."""

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
def test_run_rg09_feasibility_emits_required_artifact(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from scripts.run_rg09_feasibility import run_rg09_feasibility

    output_path = tmp_path / "rg09_episode_feasibility.json"
    report = run_rg09_feasibility(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        output_path=output_path,
    )
    assert output_path.exists()
    written = _read_json(output_path)
    assert written["fixture_sha256"] == report["fixture_sha256"]
    assert "candidate_episode_count" in written
    assert "precondition_fail_codes" in written

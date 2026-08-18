"""Integration tests for the RG-09 multi-ticker replay fixture v2 bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts.generate_rg09_fixture import FIXTURE_COLUMNS


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.integration
@pytest.mark.determinism("d2")
@pytest.mark.timeout(180)
def test_rg09_v2_fixture_bundle_schema_and_empirical_closure(
    tmp_path: Path, deterministic_seed: int
) -> None:
    """v2 parquet + sidecars must validate; empirical closure must emit gate artifacts."""
    _ = deterministic_seed
    from scripts.run_rg09_empirical_closure import run_rg09_empirical_closure_cli

    root = Path(__file__).resolve().parents[4]
    fixture_parquet = root / "fixtures/rg09/v2/rg09_fixture_v2.parquet"
    if not fixture_parquet.is_file():
        pytest.skip("fixtures/rg09/v2/rg09_fixture_v2.parquet not present in workspace")

    summary_path = root / "fixtures/rg09/v2/rg09_fixture_summary.json"
    metadata_path = root / "fixtures/rg09/v2/rg09_fixture_metadata.json"
    summary = _read_json(summary_path)
    metadata = _read_json(metadata_path)

    assert summary["amendment"] == "MLN-02-AMD-01"
    assert isinstance(summary.get("projection_rule"), str)
    assert summary["projection_rule"]
    assert summary["uniform_calendar_day_index"] is False
    assert summary["calendar_overlap_policy"] == "independent_instruments"
    assert summary["fold_construction"]["method"] == "calendar_time"
    sha = summary["fixture_sha256"]
    assert sha == metadata["fixture_sha256"]

    frame = pd.read_parquet(fixture_parquet)
    assert list(frame.columns) == list(FIXTURE_COLUMNS)
    assert frame["entity_id"].nunique() >= 2
    assert isinstance(summary["entity_id"], list)
    assert set(summary["entity_id"]) == set(frame["entity_id"].unique())

    out = tmp_path / "rg09_ec_v2"
    code = run_rg09_empirical_closure_cli(
        fixture_path=fixture_parquet,
        fixture_summary_path=summary_path,
        fixture_metadata_path=metadata_path,
        config_path=root / "docs/rg09/rg09_pilot_config_v1.json",
        output_dir=out,
    )
    assert code in (0, 1)
    gate_path = out / "rg09_gate_result.json"
    assert gate_path.exists()
    gate_result = _read_json(gate_path)
    assert gate_result["fixture_sha256"] == sha
    assert "successor_hypotheses" in gate_result
    sh = gate_result["successor_hypotheses"]
    assert isinstance(sh.get("eligible"), bool)
    if gate_result.get("gate_executed"):
        assert gate_result.get("decision") is not None
    else:
        assert sh["eligible"] is False
        assert sh.get("reason") == "preconditions_not_met"

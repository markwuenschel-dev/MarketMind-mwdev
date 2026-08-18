"""Integration tests for shuffled_label separability diagnostic CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.run_rg09_shuffled_label_separability_diagnostic import (
    SHUFFLED_LABEL_SEP_DIAG_FILENAME,
    run_rg09_shuffled_label_separability_diagnostic,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_run_rg09_shuffled_label_separability_diagnostic_emits_artifact(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    output_path = tmp_path / SHUFFLED_LABEL_SEP_DIAG_FILENAME
    report = run_rg09_shuffled_label_separability_diagnostic(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        output_path=output_path,
        boundary_recovery_mode="boundary_recovery_v1_hysteresis",
    )
    assert output_path.exists()
    written = _read_json(output_path)
    assert written["fixture_sha256"] == report["fixture_sha256"]
    assert written["schema_version"] == report["schema_version"]
    assert "folds" in written
    assert "candidate_segment_regime_class_purity" in written
    assert "closeout" in written

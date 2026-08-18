"""Tests for promotable-path meta validity scaffold emitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.meta.meta_validity_emitter import (
    SCHEMA_VERSION,
    build_meta_validity_report_document,
    emit_meta_validity_report,
)
from pysrc.meta.seed_policy import build_run_identity
from pysrc.meta.threshold_governance import clear_threshold_register_cache, default_register_path
from pysrc.meta_learning.confidence_contract import validate_confidence_calibration_artifact_block

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.determinism("d1")
def test_meta_validity_contract_fields_and_not_appendix_b1_name(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    out = tmp_path / "meta_validity_report.json"
    doc = emit_meta_validity_report(out, seed=3)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw == doc
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["non_promotable"] is True
    assert "harvey_t_statistic" in doc
    ht = doc["harvey_t_statistic"]
    assert ht["threshold_id"] == "THR-RG09-V03"
    assert ht["threshold_register_state"] == "VALIDATED"
    assert "value" not in ht
    assert "inner_loop_gain_harvey_t" not in json.dumps(doc)
    assert "inner_loop_gain_by_regime" in doc
    assert doc["baseline_comparison"]["status"] == "unavailable"
    assert doc["pit_compliance_flag"]["status"] == "unavailable"
    assert "confidence_calibration" in doc
    validate_confidence_calibration_artifact_block(doc["confidence_calibration"])
    assert doc["overall_result"] == "SCAFFOLD_INCOMPLETE"


@pytest.mark.determinism("d1")
def test_build_meta_validity_includes_run_identity() -> None:
    doc = build_meta_validity_report_document(run_identity=build_run_identity(5))
    assert doc["run_identity"]["seed"] == 5
    assert doc["harvey_t_statistic"]["threshold_id"] == "THR-RG09-V03"
    assert doc["harvey_t_statistic"]["threshold_register_state"] == "VALIDATED"
    validate_confidence_calibration_artifact_block(doc["confidence_calibration"])


@pytest.mark.determinism("d1")
def test_harvey_stub_shows_provisional_register_state(tmp_path: Path) -> None:
    payload = json.loads(default_register_path().read_text(encoding="utf-8"))
    for row in payload["records"]:
        if row["threshold_id"] == "THR-RG09-V03":
            row["state"] = "PROVISIONAL"
    reg_path = tmp_path / "register.json"
    reg_path.write_text(json.dumps(payload), encoding="utf-8")
    clear_threshold_register_cache()
    try:
        doc = build_meta_validity_report_document(
            run_identity=build_run_identity(1),
            register_path=reg_path,
        )
        assert doc["harvey_t_statistic"]["threshold_register_state"] == "PROVISIONAL"
    finally:
        clear_threshold_register_cache()

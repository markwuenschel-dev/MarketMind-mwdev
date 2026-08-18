"""Tests for ScreeningReportBuilder: schema, taxonomy, reason_family from REASON_CODE_TO_FAMILY."""

from __future__ import annotations

import pytest

from pysrc.registry.screening_report import ScreeningReportBuilder
from pysrc.registry.screening_taxonomy import (
    REASON_CODE_TO_FAMILY,
    ReasonCode,
    ReasonFamily,
    ScreeningStage,
    ScreeningStatus,
)


@pytest.mark.determinism("d1")
def test_builder_derives_reason_family_from_reason_code() -> None:
    """reason_family in serialized stages comes from REASON_CODE_TO_FAMILY, not caller."""
    builder = ScreeningReportBuilder(
        screening_run_id="run1",
        pit_boundary="2025-01-01T00:00:00Z",
        data_snapshot_hash="abc",
        seed=42,
    )
    builder.add_candidate(spec_hash="h1", signal_name="s1", slot_index=0, evaluation_ordinal=0)
    builder.add_stage(
        candidate_index=0,
        stage=ScreeningStage.LANE_0,
        status=ScreeningStatus.REJECTED,
        reason_code=ReasonCode.IC_BELOW_THRESHOLD,
        duration_ms=10,
    )
    payload = builder.serialize()
    assert payload["candidates"]
    stages = payload["candidates"][0]["stages"]
    assert len(stages) == 1
    assert stages[0]["reason_code"] == "IC_BELOW_THRESHOLD"
    assert stages[0]["reason_family"] == REASON_CODE_TO_FAMILY[ReasonCode.IC_BELOW_THRESHOLD].value


@pytest.mark.determinism("d1")
def test_reason_code_to_family_coverage() -> None:
    """Every ReasonCode has an entry in REASON_CODE_TO_FAMILY."""
    for code in ReasonCode:
        assert code in REASON_CODE_TO_FAMILY
        assert REASON_CODE_TO_FAMILY[code] in ReasonFamily


@pytest.mark.determinism("d0")
def test_candidate_run_id_deterministic() -> None:
    """Same (screening_run_id, spec_hash, ordinal) produces same candidate_run_id."""
    b1 = ScreeningReportBuilder("run1", "2025-01-01T00:00:00Z", "hash1", 42)
    b1.add_candidate(spec_hash="spec", signal_name="s", evaluation_ordinal=0)
    p1 = b1.serialize()
    b2 = ScreeningReportBuilder("run1", "2025-01-01T00:00:00Z", "hash1", 42)
    b2.add_candidate(spec_hash="spec", signal_name="s", evaluation_ordinal=0)
    p2 = b2.serialize()
    assert p1["candidates"][0]["candidate_run_id"] == p2["candidates"][0]["candidate_run_id"]


@pytest.mark.determinism("d0")
def test_candidate_run_id_unique_for_different_ordinal() -> None:
    """Different evaluation_ordinal yields different candidate_run_id."""
    b1 = ScreeningReportBuilder("run1", "2025-01-01T00:00:00Z", "hash1", 42)
    b1.add_candidate(spec_hash="spec", signal_name="s", evaluation_ordinal=0)
    b1.add_candidate(spec_hash="spec", signal_name="s", evaluation_ordinal=1)
    p = b1.serialize()
    ids = [c["candidate_run_id"] for c in p["candidates"]]
    assert len(ids) == len(set(ids))


@pytest.mark.determinism("d1")
def test_screening_report_schema_valid() -> None:
    """Serialized report has required top-level keys and structure."""
    builder = ScreeningReportBuilder("r1", "2025-01-01T00:00:00Z", "h1", 1)
    builder.add_candidate(spec_hash="s1", signal_name="n1", slot_index=None)
    builder.add_stage(0, ScreeningStage.INTAKE, ScreeningStatus.ACCEPTED, reason_code=None)
    builder.set_final(0, "REJECTED", "LANE_0", "IC_BELOW_THRESHOLD")
    payload = builder.serialize()
    assert "schema_version" in payload
    assert "screening_run_id" in payload
    assert "pit_boundary" in payload
    assert "data_snapshot_hash" in payload
    assert "seed" in payload
    assert "candidates" in payload
    assert "summary" in payload
    assert payload["summary"]["total_candidates"] == 1
    rd = payload["summary"]["rejection_distribution"]
    assert isinstance(rd, list)
    assert rd == [{"reason_code": "IC_BELOW_THRESHOLD", "count": 1}]


@pytest.mark.determinism("d1")
def test_reject_logged_with_reason_code() -> None:
    """Candidates that fail have correct stage, status, and reason_code in serialized output."""
    builder = ScreeningReportBuilder("r1", "2025-01-01T00:00:00Z", "h1", 1)
    builder.add_candidate(spec_hash="s1", signal_name="n1")
    builder.add_stage(
        0,
        ScreeningStage.INTAKE,
        ScreeningStatus.REJECTED,
        reason_code=ReasonCode.SPEC_INVALID,
        reason_detail="bad spec",
    )
    builder.set_final(0, "REJECTED", "INTAKE", "SPEC_INVALID")
    payload = builder.serialize()
    c = payload["candidates"][0]
    assert c["final_status"] == "REJECTED"
    assert c["final_reason_code"] == "SPEC_INVALID"
    assert any(s["reason_code"] == "SPEC_INVALID" for s in c["stages"])

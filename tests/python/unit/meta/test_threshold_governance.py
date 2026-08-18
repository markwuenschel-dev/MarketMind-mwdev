"""MLN-07 threshold governance: register, resolve_threshold, preflight."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pysrc.meta.threshold_catalog import provisional_threshold_record, threshold_value_record
from pysrc.meta.threshold_governance import (
    ConfiguredThresholdSpec,
    ThresholdGovernanceError,
    clear_threshold_register_cache,
    default_register_path,
    load_threshold_register,
    maybe_preflight_rg09_harness_gate_thresholds,
    preflight_configured_thresholds,
    preflight_threshold_references,
    require_gate_threshold_id,
    resolve_threshold,
    warn_hardcoded_threshold,
)


@pytest.fixture(autouse=True)
def _reset_register_cache() -> None:
    clear_threshold_register_cache()
    yield
    clear_threshold_register_cache()


def _base_register_payload() -> dict[str, object]:
    return json.loads(default_register_path().read_text(encoding="utf-8"))


def _write_register(tmp_path: Path, payload: dict[str, object]) -> Path:
    p = tmp_path / "register.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _row(
    tid: str,
    *,
    state: str = "VALIDATED",
    gate_critical: bool = True,
) -> dict[str, object]:
    return {
        "threshold_id": tid,
        "name": f"name-{tid}",
        "governing_surface": "g",
        "consumer_surface": "c",
        "state": state,
        "current_expression": "1",
        "evidence_required": "e",
        "evidence_location": "l",
        "authority": "a",
        "gate_critical": gate_critical,
        "supersedes": None,
        "superseded_by": None,
        "last_reviewed": "2026-04-06",
    }


@pytest.mark.determinism("d1")
def test_load_canonical_register() -> None:
    reg = load_threshold_register()
    assert "THR-RG09-V01" in reg
    assert "THR-RG09-V20" in reg
    assert reg["THR-RG09-V01"].state == "VALIDATED"
    assert reg["THR-RG09-V17"].state == "PROVISIONAL"
    assert reg["THR-RG09-V20"].state == "DEPRECATED"
    assert reg["THR-RG09-V20"].superseded_by == "THR-RG09-V21"
    assert reg["THR-RG09-V01"].supersedes is None
    assert reg["THR-RG09-V01"].superseded_by is None


@pytest.mark.determinism("d1")
def test_threshold_value_record_carries_id_and_validated_state() -> None:
    rec = threshold_value_record(3.0, "THR-RG09-V03")
    assert rec["threshold_id"] == "THR-RG09-V03"
    assert rec["value"] == 3.0
    assert rec["state"] == "VALIDATED"


@pytest.mark.determinism("d1")
def test_provisional_threshold_record_marks_provisional() -> None:
    rec = provisional_threshold_record(3.0, "THR-RG09-V03")
    assert rec["threshold_id"] == "THR-RG09-V03"
    assert rec["state"] == "PROVISIONAL"


@pytest.mark.determinism("d1")
def test_resolve_validated_gate_critical_passes() -> None:
    r = resolve_threshold("THR-RG09-V01", consumer="test", gate_critical=True)
    assert r.threshold_id == "THR-RG09-V01"
    assert r.current_expression


@pytest.mark.determinism("d1")
def test_resolve_provisional_non_gate_allowed() -> None:
    r = resolve_threshold("THR-RG09-V17", consumer="research", gate_critical=False)
    assert r.state == "PROVISIONAL"


@pytest.mark.determinism("d1")
def test_resolve_provisional_gate_critical_fails(tmp_path: Path) -> None:
    payload = _base_register_payload()
    assert isinstance(payload["records"], list)
    payload["records"] = list(payload["records"]) + [
        _row("THR-T-PROV", state="PROVISIONAL", gate_critical=True),
    ]
    p = _write_register(tmp_path, payload)
    clear_threshold_register_cache()
    with pytest.raises(ThresholdGovernanceError):
        resolve_threshold("THR-T-PROV", consumer="gate", gate_critical=True, register_path=p)


@pytest.mark.determinism("d1")
def test_resolve_unknown_id_fails() -> None:
    with pytest.raises(ThresholdGovernanceError):
        resolve_threshold("THR-DOES-NOT-EXIST", consumer="x", gate_critical=False)


@pytest.mark.determinism("d1")
def test_resolve_empty_id_fails() -> None:
    with pytest.raises(ThresholdGovernanceError):
        resolve_threshold("   ", consumer="x", gate_critical=True)


@pytest.mark.determinism("d1")
def test_require_gate_threshold_id_missing_fails() -> None:
    with pytest.raises(ThresholdGovernanceError):
        require_gate_threshold_id(None, consumer="harness")


@pytest.mark.determinism("d1")
def test_deprecated_fails_even_non_gate(tmp_path: Path) -> None:
    payload = _base_register_payload()
    assert isinstance(payload["records"], list)
    payload["records"] = list(payload["records"]) + [
        _row("THR-T-DEP", state="DEPRECATED", gate_critical=False),
    ]
    p = _write_register(tmp_path, payload)
    clear_threshold_register_cache()
    with pytest.raises(ThresholdGovernanceError):
        resolve_threshold("THR-T-DEP", consumer="x", gate_critical=False, register_path=p)


@pytest.mark.determinism("d1")
def test_rejected_fails(tmp_path: Path) -> None:
    payload = _base_register_payload()
    payload["records"] = list(payload["records"]) + [
        _row("THR-T-REJ", state="REJECTED", gate_critical=False),
    ]
    p = _write_register(tmp_path, payload)
    clear_threshold_register_cache()
    with pytest.raises(ThresholdGovernanceError):
        resolve_threshold("THR-T-REJ", consumer="x", gate_critical=False, register_path=p)


@pytest.mark.determinism("d1")
def test_preflight_provisional_as_gate_fails(tmp_path: Path) -> None:
    payload = _base_register_payload()
    assert isinstance(payload["records"], list)
    payload["records"] = list(payload["records"]) + [
        _row("THR-T-PROV-PREFLIGHT", state="PROVISIONAL", gate_critical=True),
    ]
    p = _write_register(tmp_path, payload)
    clear_threshold_register_cache()
    report = preflight_threshold_references(
        [("THR-T-PROV-PREFLIGHT", True)],
        consumer="audit",
        register_path=p,
    )
    assert not report.passed
    assert any(f.code == "PROVISIONAL_AS_VALIDATED" for f in report.findings)


@pytest.mark.determinism("d1")
def test_preflight_provisional_research_ok() -> None:
    report = preflight_threshold_references(
        [("THR-RG09-V17", False)],
        consumer="audit",
    )
    assert report.passed


@pytest.mark.determinism("d1")
def test_preflight_unknown_id_fails() -> None:
    report = preflight_threshold_references(
        [("THR-UNKNOWN-XYZ", True)],
        consumer="audit",
    )
    assert not report.passed
    assert any(f.code == "UNKNOWN_THRESHOLD_ID" for f in report.findings)


@pytest.mark.determinism("d1")
def test_maybe_preflight_rg09_raises_when_env_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pysrc.meta.threshold_governance as tg

    payload = _base_register_payload()
    assert isinstance(payload["records"], list)
    for row in payload["records"]:
        assert isinstance(row, dict)
        if row["threshold_id"] == "THR-RG09-V03":
            row["state"] = "PROVISIONAL"
    register_path = _write_register(tmp_path, payload)
    monkeypatch.setattr(tg, "default_register_path", lambda: register_path)
    clear_threshold_register_cache()
    monkeypatch.setenv("MARKETMIND_MLN07_RG09_PREFLIGHT", "1")
    with pytest.raises(ThresholdGovernanceError):
        maybe_preflight_rg09_harness_gate_thresholds()


@pytest.mark.determinism("d1")
def test_warn_hardcoded_threshold_logs() -> None:
    with patch("pysrc.meta.threshold_governance._LOG") as mock_log:
        warn_hardcoded_threshold(
            consumer="non_gate_surface", detail="bare 0.05 without threshold_id"
        )
        mock_log.warning.assert_called_once()


@pytest.mark.determinism("d1")
def test_preflight_configured_thresholds_warns_for_non_gate_numeric() -> None:
    report = preflight_configured_thresholds(
        {"embargo_gap_bars_daily": 0},
        consumer="cfg",
        field_specs={
            "embargo_gap_bars_daily": ConfiguredThresholdSpec("embargo_gap_bars_daily", False)
        },
    )
    assert report.passed
    assert any(f.code == "HARDCODED_THRESHOLD_WITHOUT_ID" for f in report.findings)


@pytest.mark.determinism("d1")
def test_preflight_configured_thresholds_fails_for_gate_numeric() -> None:
    report = preflight_configured_thresholds(
        {"p_value_threshold": 0.05},
        consumer="cfg",
        field_specs={"p_value_threshold": ConfiguredThresholdSpec("p_value_threshold", True)},
    )
    assert not report.passed
    assert any(f.code == "MISSING_THRESHOLD_ID" for f in report.findings)


@pytest.mark.determinism("d1")
def test_duplicate_id_in_register_file_fails(tmp_path: Path) -> None:
    row = _row("THR-DUP", state="VALIDATED", gate_critical=False)
    payload = {"schema_version": "threshold_register.mln07.v1", "records": [row, row]}
    p = _write_register(tmp_path, payload)
    with pytest.raises(ThresholdGovernanceError):
        load_threshold_register(p)

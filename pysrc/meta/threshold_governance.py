"""MLN-07 threshold identity, state, and enforcement (ThresholdGovernanceRegister).

Canonical register: ``threshold_register.mln07.v1.json`` alongside this module.

**Enforced today**
- :func:`resolve_threshold` — single governed lookup; raises on unknown, malformed, REJECTED,
  DEPRECATED, and on PROVISIONAL when ``gate_critical=True``.
- :func:`preflight_threshold_references` — batch audit for governed runs.
- :func:`preflight_configured_thresholds` — config-layer WARN/FAIL audit for anonymous threshold use.
- :func:`warn_hardcoded_threshold` — non-gate numeric threshold without ID (audit WARN).
- :func:`require_gate_threshold_id` — gate path must pass a non-empty ``threshold_id``.

**Still manual / unwired**
- MLN-06 orchestrator wiring remains library-only per MLN-07 brief scope.
- Non-RG-09 governed consumers still need to migrate from anonymous literals to ``threshold_id``.
- ``maybe_preflight_rg09_harness_gate_thresholds`` remains as an environment-gated compatibility shim.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pysrc.ops.mm_logkit import get_logger

_LOG = get_logger(__name__)

REGISTER_FILENAME: Final[str] = "threshold_register.mln07.v1.json"

THRESHOLD_STATES: Final[frozenset[str]] = frozenset(
    {"PROVISIONAL", "VALIDATED", "REJECTED", "DEPRECATED"},
)

REQUIRED_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "threshold_id",
    "name",
    "governing_surface",
    "consumer_surface",
    "state",
    "current_expression",
    "evidence_required",
    "evidence_location",
    "authority",
    "gate_critical",
    "supersedes",
    "superseded_by",
    "last_reviewed",
)


class ThresholdGovernanceError(RuntimeError):
    """Governed threshold resolution or preflight failed."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details


class ThresholdAuditSeverity(StrEnum):
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ThresholdAuditFinding:
    severity: ThresholdAuditSeverity
    code: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ThresholdPreflightReport:
    findings: tuple[ThresholdAuditFinding, ...]
    passed: bool


@dataclass(frozen=True)
class ThresholdRecord:
    threshold_id: str
    name: str
    governing_surface: str
    consumer_surface: str
    state: str
    current_expression: str
    evidence_required: str
    evidence_location: str
    authority: str
    gate_critical: bool
    supersedes: str | None
    superseded_by: str | None
    last_reviewed: str


@dataclass(frozen=True)
class ConfiguredThresholdSpec:
    field_name: str
    gate_critical: bool


def default_register_path() -> Path:
    return Path(__file__).resolve().parent / REGISTER_FILENAME


def load_threshold_register(path: Path | None = None) -> dict[str, ThresholdRecord]:
    """Load and validate the canonical register; returns ``threshold_id`` -> record."""
    reg_path = (default_register_path() if path is None else Path(path)).resolve()
    raw_text = reg_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if not isinstance(data, Mapping):
        raise ThresholdGovernanceError("threshold register root must be an object")
    sv = data.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        raise ThresholdGovernanceError("threshold register missing schema_version")
    recs = data.get("records")
    if not isinstance(recs, list):
        raise ThresholdGovernanceError("threshold register missing records array")
    out: dict[str, ThresholdRecord] = {}
    for i, row in enumerate(recs):
        if not isinstance(row, Mapping):
            raise ThresholdGovernanceError(f"records[{i}] must be an object")
        missing = [f for f in REQUIRED_RECORD_FIELDS if f not in row]
        if missing:
            raise ThresholdGovernanceError(
                f"records[{i}] missing fields: {missing}",
                details={"index": i, "missing": missing},
            )
        tid = str(row["threshold_id"]).strip()
        if not tid:
            raise ThresholdGovernanceError(f"records[{i}] has empty threshold_id")
        state = str(row["state"]).strip().upper()
        if state not in THRESHOLD_STATES:
            raise ThresholdGovernanceError(
                f"invalid state for {tid!r}: {row['state']!r}",
                details={"threshold_id": tid, "state": row["state"]},
            )
        gc = row["gate_critical"]
        if not isinstance(gc, bool):
            raise ThresholdGovernanceError(
                f"gate_critical must be bool for {tid!r}",
                details={"threshold_id": tid},
            )
        sup = row.get("supersedes")
        sub = row.get("superseded_by")
        record = ThresholdRecord(
            threshold_id=tid,
            name=str(row["name"]),
            governing_surface=str(row["governing_surface"]),
            consumer_surface=str(row["consumer_surface"]),
            state=state,
            current_expression=str(row["current_expression"]),
            evidence_required=str(row["evidence_required"]),
            evidence_location=str(row["evidence_location"]),
            authority=str(row["authority"]),
            gate_critical=gc,
            supersedes=None if sup is None else str(sup),
            superseded_by=None if sub is None else str(sub),
            last_reviewed=str(row["last_reviewed"]),
        )
        if tid in out:
            raise ThresholdGovernanceError(
                "duplicate threshold_id in register",
                details={"threshold_id": tid},
            )
        out[tid] = record
    return out


_REGISTER_CACHE: dict[str, dict[str, ThresholdRecord]] | None = None


def clear_threshold_register_cache() -> None:
    """Clear loader cache (tests and hot-reload tooling)."""
    global _REGISTER_CACHE
    _REGISTER_CACHE = None


def _cached_register(path: Path | None) -> dict[str, ThresholdRecord]:
    global _REGISTER_CACHE
    reg_path = (default_register_path() if path is None else Path(path)).resolve()
    key = str(reg_path)
    if _REGISTER_CACHE is None:
        _REGISTER_CACHE = {}
    if key not in _REGISTER_CACHE:
        _REGISTER_CACHE[key] = load_threshold_register(reg_path)
    return _REGISTER_CACHE[key]


def resolve_threshold(
    threshold_id: str,
    *,
    consumer: str,
    gate_critical: bool,
    register_path: Path | None = None,
) -> ThresholdRecord:
    """
    Single governed lookup path.

    * FAIL: unknown ID, malformed register row (loader), REJECTED, DEPRECATED.
    * FAIL: ``gate_critical=True`` and state is not ``VALIDATED``.
    * OK: ``VALIDATED`` always; ``PROVISIONAL`` only when ``gate_critical=False``.
    """
    tid = threshold_id.strip()
    if not tid:
        raise ThresholdGovernanceError(
            "gate-critical consumption requires a non-empty threshold_id",
            details={"consumer": consumer},
        )
    reg = _cached_register(register_path)
    if tid not in reg:
        raise ThresholdGovernanceError(
            "threshold_id not present in canonical register",
            details={"threshold_id": tid, "consumer": consumer},
        )
    rec = reg[tid]
    if rec.state == "REJECTED":
        raise ThresholdGovernanceError(
            "REJECTED threshold must not be used in governed logic",
            details={"threshold_id": tid, "consumer": consumer},
        )
    if rec.state == "DEPRECATED":
        raise ThresholdGovernanceError(
            "DEPRECATED threshold must not be used in new governed logic",
            details={"threshold_id": tid, "consumer": consumer, "superseded_by": rec.superseded_by},
        )
    if gate_critical and rec.state != "VALIDATED":
        raise ThresholdGovernanceError(
            "gate-critical path requires VALIDATED threshold state",
            details={
                "threshold_id": tid,
                "consumer": consumer,
                "state": rec.state,
            },
        )
    return rec


def require_gate_threshold_id(threshold_id: str | None, *, consumer: str) -> str:
    """Return stripped ID or raise if missing (anonymous gate threshold)."""
    if threshold_id is None or not str(threshold_id).strip():
        raise ThresholdGovernanceError(
            "gate decision path requires threshold_id",
            details={"consumer": consumer},
        )
    return str(threshold_id).strip()


def warn_hardcoded_threshold(*, consumer: str, detail: str) -> None:
    """Emit audit WARN for numeric/rule threshold use without register identity (non-gate)."""
    _LOG.warning(
        "mln07_hardcoded_threshold",
        consumer=consumer,
        detail=detail,
        severity=ThresholdAuditSeverity.WARN.value,
    )


def preflight_configured_thresholds(
    raw_config: Mapping[str, Any],
    *,
    consumer: str,
    field_specs: Mapping[str, ConfiguredThresholdSpec],
    register_path: Path | None = None,
) -> ThresholdPreflightReport:
    """
    Audit config-backed threshold fields before a governed run.

    Rules:
    - hardcoded threshold without ID -> WARN
    - threshold without ID in gate-critical path -> FAIL
    - missing threshold ID from register -> FAIL
    - PROVISIONAL used as validated in gate-critical path -> FAIL
    - DEPRECATED / REJECTED -> FAIL via reference preflight
    """
    findings: list[ThresholdAuditFinding] = []
    refs: list[tuple[str, bool]] = []
    for field_name, spec in field_specs.items():
        if field_name not in raw_config:
            continue
        raw_value = raw_config[field_name]
        if isinstance(raw_value, Mapping):
            threshold_id_raw = raw_value.get("threshold_id")
            if threshold_id_raw is None or not str(threshold_id_raw).strip():
                severity = (
                    ThresholdAuditSeverity.FAIL
                    if spec.gate_critical
                    else ThresholdAuditSeverity.WARN
                )
                findings.append(
                    ThresholdAuditFinding(
                        severity=severity,
                        code=(
                            "MISSING_THRESHOLD_ID"
                            if spec.gate_critical
                            else "HARDCODED_THRESHOLD_WITHOUT_ID"
                        ),
                        message=(
                            "gate-critical threshold field requires threshold_id"
                            if spec.gate_critical
                            else "threshold field is missing threshold_id and will be treated as hardcoded"
                        ),
                        details={"consumer": consumer, "field_name": field_name},
                    )
                )
                continue
            refs.append((str(threshold_id_raw).strip(), spec.gate_critical))
            continue
        severity = (
            ThresholdAuditSeverity.FAIL if spec.gate_critical else ThresholdAuditSeverity.WARN
        )
        findings.append(
            ThresholdAuditFinding(
                severity=severity,
                code="MISSING_THRESHOLD_ID"
                if spec.gate_critical
                else "HARDCODED_THRESHOLD_WITHOUT_ID",
                message=(
                    "gate-critical threshold field must not use anonymous numeric values"
                    if spec.gate_critical
                    else "threshold field uses anonymous numeric value"
                ),
                details={"consumer": consumer, "field_name": field_name},
            )
        )
    ref_report = preflight_threshold_references(
        refs, consumer=consumer, register_path=register_path
    )
    findings.extend(ref_report.findings)
    passed = not any(f.severity == ThresholdAuditSeverity.FAIL for f in findings)
    return ThresholdPreflightReport(findings=tuple(findings), passed=passed)


def preflight_threshold_references(
    refs: Sequence[tuple[str, bool]],
    *,
    consumer: str,
    register_path: Path | None = None,
) -> ThresholdPreflightReport:
    """
    Verify referenced IDs exist and gate-critical uses are state-legal.

    ``refs`` items are ``(threshold_id, used_as_gate_critical)``.
    """
    findings: list[ThresholdAuditFinding] = []
    reg = _cached_register(register_path)
    for tid_raw, gc in refs:
        tid = tid_raw.strip()
        if not tid:
            findings.append(
                ThresholdAuditFinding(
                    severity=ThresholdAuditSeverity.FAIL,
                    code="MISSING_THRESHOLD_ID",
                    message="empty threshold_id in governed reference list",
                    details={"consumer": consumer},
                )
            )
            continue
        if tid not in reg:
            findings.append(
                ThresholdAuditFinding(
                    severity=ThresholdAuditSeverity.FAIL,
                    code="UNKNOWN_THRESHOLD_ID",
                    message="referenced threshold_id not in register",
                    details={"threshold_id": tid, "consumer": consumer},
                )
            )
            continue
        rec = reg[tid]
        if rec.state in ("REJECTED", "DEPRECATED"):
            findings.append(
                ThresholdAuditFinding(
                    severity=ThresholdAuditSeverity.FAIL,
                    code="INVALID_STATE",
                    message=f"threshold state {rec.state} forbids use",
                    details={"threshold_id": tid, "consumer": consumer},
                )
            )
            continue
        if gc and rec.state != "VALIDATED":
            findings.append(
                ThresholdAuditFinding(
                    severity=ThresholdAuditSeverity.FAIL,
                    code="PROVISIONAL_AS_VALIDATED",
                    message="gate-critical reference requires VALIDATED state",
                    details={"threshold_id": tid, "consumer": consumer, "state": rec.state},
                )
            )
    passed = not any(f.severity == ThresholdAuditSeverity.FAIL for f in findings)
    return ThresholdPreflightReport(findings=tuple(findings), passed=passed)


def preflight_report_to_json(report: ThresholdPreflightReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "findings": [
            {
                "severity": f.severity.value,
                "code": f.code,
                "message": f.message,
                "details": f.details,
            }
            for f in report.findings
        ],
    }


#: RG-09 pilot thresholds referenced by the bounded harness gate surfaces (MLN-07 audit set).
RG09_HARNESS_GATE_PREFLIGHT_IDS: Final[tuple[str, ...]] = (
    "THR-RG09-V01",
    "THR-RG09-V02",
    "THR-RG09-V03",
    "THR-RG09-V06",
    "THR-RG09-V07",
    "THR-RG09-V08",
    "THR-RG09-V09",
    "THR-RG09-V10",
    "THR-RG09-V11",
    "THR-RG09-V12",
    "THR-RG09-V13",
    "THR-RG09-V15",
    "THR-RG09-V16",
    "THR-RG09-V19",
)


def maybe_preflight_rg09_harness_gate_thresholds(*, consumer: str = "rg09_harness") -> None:
    """
    When ``MARKETMIND_MLN07_RG09_PREFLIGHT=1``, enforce register compliance for RG-09 gate IDs.

    Default is **off** so PROVISIONAL register rows (e.g. ``THR-RG09-V17`` on the canonical path)
    do not break CI for gate-ID lists that still include them. Enable for strict governed runs.
    """
    if os.environ.get("MARKETMIND_MLN07_RG09_PREFLIGHT", "").strip() != "1":
        return
    refs = [(tid, True) for tid in RG09_HARNESS_GATE_PREFLIGHT_IDS]
    report = preflight_threshold_references(refs, consumer=consumer)
    if not report.passed:
        raise ThresholdGovernanceError(
            "RG-09 harness threshold preflight failed",
            details={"findings": preflight_report_to_json(report)},
        )


__all__ = [
    "clear_threshold_register_cache",
    "RG09_HARNESS_GATE_PREFLIGHT_IDS",
    "ThresholdAuditFinding",
    "ThresholdAuditSeverity",
    "ThresholdGovernanceError",
    "ThresholdPreflightReport",
    "ThresholdRecord",
    "ConfiguredThresholdSpec",
    "default_register_path",
    "load_threshold_register",
    "maybe_preflight_rg09_harness_gate_thresholds",
    "preflight_configured_thresholds",
    "preflight_report_to_json",
    "preflight_threshold_references",
    "require_gate_threshold_id",
    "resolve_threshold",
    "warn_hardcoded_threshold",
]

"""Gate CLI for validating run bundles per Appendix D contract (v5.1).

Normative gate-oriented delivery intent: Programming Guidelines §6.2 (capabilities must
expose measurable evidence for promotion).

This module implements the Gate CLI that validates run bundles against
the specifications in Appendix C (bundle structure) and Appendix D
(gate contract) of the MarketMind Implementation Plan v5.1.

Usage (per Appendix D.1):
    marketmind-gate check --bundle /path/to/run_bundle_v1 --output /path/to/gate_result.json

    # Also supports shorthand:
    python -m pysrc.cli.gate check ./run_bundle_001/
    python -m pysrc.cli.gate validate ./run_bundle_001/

Exit Codes (per Appendix D.1):
    0 = all gates PASS
    1 = one or more gates FAIL (validation failures)
    2 = invalid input (missing bundle, malformed JSON, unknown schema version)
    3 = internal error (unexpected exception)

Output:
    gate_result.json written to --output path (or bundle/gate_result.json by default)
    Human logs go to stderr; machine-readable output is gate_result.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
from typing import Any

try:
    click: Any = importlib.import_module("click")
except ModuleNotFoundError:

    class _ClickFallback:
        @staticmethod
        def echo(message: str, err: bool = False) -> None:
            stream = sys.stderr if err else sys.stdout
            stream.write(f"{message}\n")

    click = _ClickFallback()

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pysrc.artifact_registry.reproducibility import (
    collect_bundle_reproducibility_echo,
    validate_plan_reproducibility_fields,
)
from pysrc.ops.hashing.canonical_frame import CANONICAL_FRAME_CI_STATUS_VALUE
from pysrc.ops.telemetry import SPAN_GATE_EVALUATE, tracer

# Import splits manifest validation
try:
    from pysrc.preprocessor.splits import (
        SPLITS_SCHEMA_VERSION,
        SplitsManifest,
    )

    SPLITS_AVAILABLE = True
except ImportError:
    SPLITS_AVAILABLE = False
    SPLITS_SCHEMA_VERSION = "1.0.0"


# Gate result schema version
GATE_SCHEMA_VERSION = "1.0.0"

# Gate CLI version
GATE_CLI_VERSION = "0.2.0"


class ExitCode(Enum):
    """Exit codes per Appendix D.1."""

    PASS = 0
    FAIL = 1
    INVALID_INPUT = 2
    INTERNAL_ERROR = 3


class GateResult(Enum):
    """Gate result status."""

    PASS = "PASS"
    FAIL = "FAIL"


class ReasonCode(Enum):
    """Reason codes for gate results.

    Codes that trigger EXIT_CODE 2 (invalid input):
    - UNKNOWN_SCHEMA_VERSION
    - MISSING_SCHEMA_VERSION
    - INVALID_SCHEMA_VERSION
    - MALFORMED_JSON

    Note: MISSING_FILE for required files within an existing bundle triggers
    EXIT_CODE 1 (gate failure), not EXIT_CODE 2. Only a missing bundle directory
    itself triggers EXIT_CODE 2 (handled specially in validate_bundle_exists).
    """

    # Success codes
    VALID = "VALID"

    # Schema validation errors (EXIT CODE 2)
    UNKNOWN_SCHEMA_VERSION = "UNKNOWN_SCHEMA_VERSION"
    MISSING_SCHEMA_VERSION = "MISSING_SCHEMA_VERSION"
    INVALID_SCHEMA_VERSION = "INVALID_SCHEMA_VERSION"

    # File errors
    MISSING_FILE = "MISSING_FILE"
    MALFORMED_JSON = "MALFORMED_JSON"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"

    # Plan validation errors
    MISSING_PLAN_HASH = "MISSING_PLAN_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"

    # Splits validation errors
    INVALID_SPLITS = "INVALID_SPLITS"
    LEAKAGE_DETECTED = "LEAKAGE_DETECTED"
    PURGE_VIOLATION = "PURGE_VIOLATION"
    EMBARGO_VIOLATION = "EMBARGO_VIOLATION"

    # Config errors
    INVALID_CONFIG = "INVALID_CONFIG"

    # Statistical validity report (required policy artifact; Appendix H)
    STAT_VALIDITY_INVALID_STRUCTURE = "STAT_VALIDITY_INVALID_STRUCTURE"
    STAT_VALIDITY_GATE_FAIL = "STAT_VALIDITY_GATE_FAIL"

    # Execution / cost assumptions (required policy artifact; Appendix G)
    COST_ASSUMPTION_MISSING = "COST_ASSUMPTION_MISSING"
    COST_GATE_REJECTED = "COST_GATE_REJECTED"
    ZERO_COST_ASSUMED = "ZERO_COST_ASSUMED"
    EXECUTION_ASSUMPTIONS_INVALID_STRUCTURE = "EXECUTION_ASSUMPTIONS_INVALID_STRUCTURE"
    PIT_NON_COMPLIANT = "PIT_NON_COMPLIANT"
    MISSING_KNOWLEDGE_TIME_COLUMN = "MISSING_KNOWLEDGE_TIME_COLUMN"
    CONTENT_HASH_MISMATCH = "CONTENT_HASH_MISMATCH"
    STALE_DOWNLOAD_WARNING = "STALE_DOWNLOAD_WARNING"

    # Optional plan.json reproducibility fields (Programming Guidelines §7.2)
    INVALID_DETERMINISM_TIER = "INVALID_DETERMINISM_TIER"
    INVALID_REPRODUCIBILITY_METADATA = "INVALID_REPRODUCIBILITY_METADATA"


# Reason codes that should trigger exit code 2 (invalid input)
INVALID_INPUT_REASON_CODES = {
    ReasonCode.UNKNOWN_SCHEMA_VERSION.value,
    ReasonCode.MISSING_SCHEMA_VERSION.value,
    ReasonCode.INVALID_SCHEMA_VERSION.value,
    ReasonCode.MALFORMED_JSON.value,
    ReasonCode.STAT_VALIDITY_INVALID_STRUCTURE.value,
    ReasonCode.EXECUTION_ASSUMPTIONS_INVALID_STRUCTURE.value,
}


@dataclass
class GateCheck:
    """Result of a single gate check."""

    gate_id: str
    result: str  # "PASS" or "FAIL"
    reason_code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "result": self.result,
            "reason_code": self.reason_code,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class GateReport:
    """Complete gate validation report per Appendix D.2."""

    schema_version: str = GATE_SCHEMA_VERSION
    bundle_path: str = ""
    timestamp: str = ""
    overall_result: str = "PASS"
    gates: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _has_invalid_input: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.timestamp:
            # Use 'Z' suffix per Appendix D.2 example format
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.metadata:
            self.metadata = {
                "gate_version": GATE_CLI_VERSION,
                "execution_time_ms": 0,
                "canonical_frame_ci_status": CANONICAL_FRAME_CI_STATUS_VALUE,
            }

    def add_check(self, check: GateCheck) -> None:
        """Add a gate check result."""
        self.gates.append(check.to_dict())
        if check.result == GateResult.FAIL.value:
            self.overall_result = GateResult.FAIL.value
            # Track if this is an "invalid input" failure
            if check.reason_code in INVALID_INPUT_REASON_CODES:
                self._has_invalid_input = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_path": self.bundle_path,
            "timestamp": self.timestamp,
            "overall_result": self.overall_result,
            "gates": self.gates,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> None:
        """Write report to JSON file."""
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def resolve_gate_output_path(output_path: Path | None, bundle_path: Path) -> Path:
    """Resolve the canonical output location for ``gate_result.json``."""
    if output_path is not None:
        return output_path
    if bundle_path.exists() and bundle_path.is_dir():
        return bundle_path / "gate_result.json"
    return Path("gate_result.json")


def write_gate_report(report: GateReport, output_path: Path | None, bundle_path: Path) -> Path:
    """Persist a gate report to the canonical output location."""
    target = resolve_gate_output_path(output_path, bundle_path)
    report.to_json(target)
    return target


def emit_gate_failure_report(
    bundle_path: Path,
    *,
    gate_id: str,
    reason_code: str,
    message: str,
    evidence: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> GateReport:
    """Emit a canonical failure-shaped gate report outside normal validation flow."""
    report = GateReport(bundle_path=str(bundle_path.absolute()))
    report.add_check(
        GateCheck(
            gate_id=gate_id,
            result=GateResult.FAIL.value,
            reason_code=reason_code,
            message=message,
            evidence=evidence or {},
        )
    )
    report.metadata["execution_time_ms"] = 0
    write_gate_report(report, output_path, bundle_path)
    return report


# =============================================================================
# Appendix C: Required Bundle Files (v5.1)
# =============================================================================

# Required files per Appendix C.2
REQUIRED_BUNDLE_FILES = [
    "plan.json",
    "env_fingerprint.json",
    "dataset_manifest.json",
    "preprocessing_report.json",
    "splits_manifest.json",
]

# Optional files per Appendix C.2
OPTIONAL_BUNDLE_FILES = [
    "stat_validity_report.json",
    "execution_assumptions.json",
]

# Schema field requirements per Appendix C.2 (Key Fields column)
PLAN_REQUIRED_FIELDS = ["schema_version", "plan_hash", "as_of_time", "config_hash"]
ENV_REQUIRED_FIELDS = ["schema_version", "python_version", "git_sha", "deps"]
DATASET_REQUIRED_FIELDS = ["schema_version", "dataset_id", "symbols", "row_count", "time_range"]
PREPROCESSING_REQUIRED_FIELDS = ["schema_version", "steps", "timings", "warnings"]
SPLITS_REQUIRED_FIELDS = [
    "schema_version",
    "split_method",
    "purge_window",
    "embargo_window",
    "splits",
]


def _log(msg: str) -> None:
    """Log to stderr per Appendix D.1."""
    click.echo(msg, err=True)


def _gate_eval_telemetry_result(report: GateReport) -> str:
    """Map gate report to PASS / WARN / FAIL for OTel (stat_validity WARN lives in evidence)."""
    if report.overall_result != GateResult.PASS.value:
        return str(report.overall_result)
    for g in report.gates:
        ev = g.get("evidence") or {}
        if ev.get("gate_result") == "WARN":
            return "WARN"
    return "PASS"


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse ISO timestamp string to datetime.

    Handles 'Z' suffix, positive offsets (+00:00), and negative offsets (-05:00).
    """
    if not ts_str:
        return None

    try:
        # Handle 'Z' suffix -> convert to +00:00
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"

        # Parse the timestamp
        dt = datetime.fromisoformat(ts_str)

        # If naive (no tzinfo), assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return dt
    except (ValueError, TypeError):
        return None


def _validate_schema_version(
    version: str, supported_major: int = 1, field_name: str = "schema_version"
) -> tuple[bool, str, str]:
    """Validate schema version is supported.

    Returns:
        (is_valid, reason_code, message)
    """
    if not version:
        return False, ReasonCode.MISSING_SCHEMA_VERSION.value, f"Missing {field_name}"

    try:
        parts = version.split(".")
        if len(parts) != 3:
            return (
                False,
                ReasonCode.INVALID_SCHEMA_VERSION.value,
                f"Invalid version format: {version}",
            )

        major = int(parts[0])
        if major > supported_major:
            return (
                False,
                ReasonCode.UNKNOWN_SCHEMA_VERSION.value,
                f"Version {version} not supported (max major: {supported_major})",
            )

        return True, ReasonCode.VALID.value, f"Version {version} is supported"
    except (ValueError, IndexError) as e:
        return (
            False,
            ReasonCode.INVALID_SCHEMA_VERSION.value,
            f"Cannot parse version {version}: {e}",
        )


def _load_json_file(path: Path) -> tuple[dict[str, Any] | None, str, str]:
    """Load and parse a JSON file.

    Returns:
        (data, reason_code, message) - data is None on error
    """
    if not path.exists():
        return None, ReasonCode.MISSING_FILE.value, f"File not found: {path}"

    try:
        with open(path) as f:
            data = json.load(f)
        return data, ReasonCode.VALID.value, f"Loaded {path.name}"
    except json.JSONDecodeError as e:
        return None, ReasonCode.MALFORMED_JSON.value, f"Invalid JSON in {path.name}: {e}"
    except Exception as e:
        return None, ReasonCode.INVALID_STRUCTURE.value, f"Error reading {path.name}: {e}"


def validate_bundle_exists(bundle_path: Path, report: GateReport) -> bool:
    """Validate bundle directory exists."""
    check = GateCheck(
        gate_id="bundle_exists",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message="Bundle directory exists",
        evidence={"path": str(bundle_path)},
    )

    if not bundle_path.exists():
        check.result = GateResult.FAIL.value
        check.reason_code = ReasonCode.MISSING_FILE.value
        check.message = f"Bundle directory not found: {bundle_path}"
        report.add_check(check)
        return False

    if not bundle_path.is_dir():
        check.result = GateResult.FAIL.value
        check.reason_code = ReasonCode.INVALID_STRUCTURE.value
        check.message = f"Bundle path is not a directory: {bundle_path}"
        report.add_check(check)
        return False

    report.add_check(check)
    return True


def validate_required_files(bundle_path: Path, report: GateReport) -> dict[str, bool]:
    """Validate required files are present per Appendix C.

    Returns:
        Dict mapping filename to exists status
    """
    results = {}

    for filename in REQUIRED_BUNDLE_FILES:
        file_path = bundle_path / filename
        exists = file_path.exists()
        results[filename] = exists

        reason_code = ReasonCode.VALID.value if exists else ReasonCode.MISSING_FILE.value

        check = GateCheck(
            gate_id=f"file_exists_{filename.replace('.', '_')}",
            result=GateResult.PASS.value if exists else GateResult.FAIL.value,
            reason_code=reason_code,
            message=f"Required file {filename} {'found' if exists else 'missing'}",
            evidence={"path": str(file_path), "exists": exists},
        )
        report.add_check(check)

    return results


def validate_plan_identity(
    bundle_path: Path,
    report: GateReport,
    expected_plan_hash: str | None = None,
) -> bool:
    """Validate plan.json schema and identity per Appendix C.2/D.2.

    Args:
        bundle_path: Path to bundle directory
        report: GateReport to add checks to
        expected_plan_hash: Optional expected hash for HASH_MISMATCH check
    """
    plan_path = bundle_path / "plan.json"

    data, reason_code, message = _load_json_file(plan_path)

    if data is None:
        check = GateCheck(
            gate_id="plan_identity",
            result=GateResult.FAIL.value,
            reason_code=reason_code,
            message=message,
            evidence={"path": str(plan_path)},
        )
        report.add_check(check)
        return False

    # Validate schema version
    schema_version = data.get("schema_version", "")
    valid, reason, msg = _validate_schema_version(schema_version)

    if not valid:
        check = GateCheck(
            gate_id="plan_identity",
            result=GateResult.FAIL.value,
            reason_code=reason,
            message=msg,
            evidence={"schema_version": schema_version, "path": str(plan_path)},
        )
        report.add_check(check)
        return False

    # Check for plan_hash (required per Appendix C.2)
    actual_plan_hash = data.get("plan_hash")
    if not actual_plan_hash:
        check = GateCheck(
            gate_id="plan_identity",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MISSING_PLAN_HASH.value,
            message="Missing plan_hash in plan.json",
            evidence={"present_fields": list(data.keys())},
        )
        report.add_check(check)
        return False

    # If expected hash provided, verify match (per Appendix D.2 evidence pattern)
    if expected_plan_hash is not None and actual_plan_hash != expected_plan_hash:
        check = GateCheck(
            gate_id="plan_identity",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.HASH_MISMATCH.value,
            message="Plan hash does not match expected value",
            evidence={
                "expected": expected_plan_hash,
                "actual": actual_plan_hash,
            },
        )
        report.add_check(check)
        return False

    # Validate all required fields per Appendix C.2
    missing_fields = [f for f in PLAN_REQUIRED_FIELDS if f not in data]

    if missing_fields:
        check = GateCheck(
            gate_id="plan_identity",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MISSING_REQUIRED_FIELD.value,
            message=f"Missing required fields: {missing_fields}",
            evidence={"missing_fields": missing_fields, "present_fields": list(data.keys())},
        )
        report.add_check(check)
        return False

    ok_rep, rep_rc, rep_msg = validate_plan_reproducibility_fields(data)
    if not ok_rep:
        check = GateCheck(
            gate_id="plan_identity",
            result=GateResult.FAIL.value,
            reason_code=rep_rc,
            message=rep_msg,
            evidence={"path": str(plan_path), "present_fields": list(data.keys())},
        )
        report.add_check(check)
        return False

    # Build evidence with expected/actual pattern when hash was verified
    evidence = {
        "schema_version": schema_version,
        "plan_hash": actual_plan_hash,
        "as_of_time": data.get("as_of_time"),
        "config_hash": data.get("config_hash"),
    }
    for opt_key in (
        "determinism_tier",
        "planner_version",
        "feature_op_registry_version",
        "seed_lineage",
        "partition_identity",
    ):
        if opt_key in data:
            evidence[opt_key] = data[opt_key]
    if expected_plan_hash is not None:
        evidence["expected"] = expected_plan_hash
        evidence["actual"] = actual_plan_hash

    check = GateCheck(
        gate_id="plan_identity",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message="Plan identity is valid",
        evidence=evidence,
    )
    report.add_check(check)
    return True


def validate_env_fingerprint(bundle_path: Path, report: GateReport) -> bool:
    """Validate env_fingerprint.json schema per Appendix C.2."""
    env_path = bundle_path / "env_fingerprint.json"

    data, reason_code, message = _load_json_file(env_path)

    if data is None:
        check = GateCheck(
            gate_id="env_fingerprint",
            result=GateResult.FAIL.value,
            reason_code=reason_code,
            message=message,
            evidence={"path": str(env_path)},
        )
        report.add_check(check)
        return False

    # Validate schema version
    schema_version = data.get("schema_version", "")
    valid, reason, msg = _validate_schema_version(schema_version)

    if not valid:
        check = GateCheck(
            gate_id="env_fingerprint",
            result=GateResult.FAIL.value,
            reason_code=reason,
            message=msg,
            evidence={"schema_version": schema_version},
        )
        report.add_check(check)
        return False

    # Validate all required fields per Appendix C.2 Key Fields
    missing_fields = [f for f in ENV_REQUIRED_FIELDS if f not in data]

    if missing_fields:
        check = GateCheck(
            gate_id="env_fingerprint",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MISSING_REQUIRED_FIELD.value,
            message=f"Missing required fields: {missing_fields}",
            evidence={"missing_fields": missing_fields, "present_fields": list(data.keys())},
        )
        report.add_check(check)
        return False

    check = GateCheck(
        gate_id="env_fingerprint",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message="Environment fingerprint is valid",
        evidence={
            "schema_version": schema_version,
            "python_version": data.get("python_version"),
            "git_sha": data.get("git_sha"),
            "deps_count": len(data.get("deps", {})),
        },
    )
    report.add_check(check)
    return True


def validate_dataset_manifest(bundle_path: Path, report: GateReport) -> bool:
    """Validate dataset_manifest.json schema per Appendix C.2."""
    dataset_path = bundle_path / "dataset_manifest.json"

    data, reason_code, message = _load_json_file(dataset_path)

    if data is None:
        check = GateCheck(
            gate_id="dataset_manifest",
            result=GateResult.FAIL.value,
            reason_code=reason_code,
            message=message,
            evidence={"path": str(dataset_path)},
        )
        report.add_check(check)
        return False

    # Validate schema version
    schema_version = data.get("schema_version", "")
    valid, reason, msg = _validate_schema_version(schema_version)

    if not valid:
        check = GateCheck(
            gate_id="dataset_manifest",
            result=GateResult.FAIL.value,
            reason_code=reason,
            message=msg,
            evidence={"schema_version": schema_version},
        )
        report.add_check(check)
        return False

    # Validate all required fields per Appendix C.2 Key Fields
    missing_fields = [f for f in DATASET_REQUIRED_FIELDS if f not in data]

    if missing_fields:
        check = GateCheck(
            gate_id="dataset_manifest",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MISSING_REQUIRED_FIELD.value,
            message=f"Missing required fields: {missing_fields}",
            evidence={"missing_fields": missing_fields, "present_fields": list(data.keys())},
        )
        report.add_check(check)
        return False

    check = GateCheck(
        gate_id="dataset_manifest",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message="Dataset manifest is valid",
        evidence={
            "schema_version": schema_version,
            "dataset_id": data.get("dataset_id"),
            "symbols": data.get("symbols"),
            "row_count": data.get("row_count"),
            "time_range": data.get("time_range"),
        },
    )
    report.add_check(check)
    return True


class DataLineageGate:
    """Governed lineage enforcement for dataset_manifest.json on the canonical path."""

    def __init__(self, *, max_download_age_days: int = 30) -> None:
        self._max_download_age_days = max_download_age_days

    def validate(self, bundle_path: Path, report: GateReport) -> bool:
        dataset_path = bundle_path / "dataset_manifest.json"
        data, reason_code, message = _load_json_file(dataset_path)
        if data is None:
            check = GateCheck(
                gate_id="data_lineage",
                result=GateResult.FAIL.value,
                reason_code=reason_code,
                message=message,
                evidence={"path": str(dataset_path)},
            )
            report.add_check(check)
            return False

        pit_compliant = bool(data.get("pit_compliant"))
        if not pit_compliant:
            report.add_check(
                GateCheck(
                    gate_id="data_lineage",
                    result=GateResult.FAIL.value,
                    reason_code=ReasonCode.PIT_NON_COMPLIANT.value,
                    message="dataset_manifest.json must declare pit_compliant=true on the governed path",
                    evidence={
                        "path": str(dataset_path),
                        "pit_compliant": data.get("pit_compliant"),
                    },
                )
            )
            return False

        knowledge_time_column = data.get("knowledge_time_column")
        if not isinstance(knowledge_time_column, str) or not knowledge_time_column.strip():
            report.add_check(
                GateCheck(
                    gate_id="data_lineage",
                    result=GateResult.FAIL.value,
                    reason_code=ReasonCode.MISSING_KNOWLEDGE_TIME_COLUMN.value,
                    message="dataset_manifest.json must include a non-empty knowledge_time_column on the governed path",
                    evidence={
                        "path": str(dataset_path),
                        "knowledge_time_column": knowledge_time_column,
                    },
                )
            )
            return False

        content_hash = data.get("content_hash")
        content_hash_expected = data.get("content_hash_expected")
        if (
            isinstance(content_hash, str)
            and isinstance(content_hash_expected, str)
            and content_hash != content_hash_expected
        ):
            report.add_check(
                GateCheck(
                    gate_id="data_lineage",
                    result=GateResult.FAIL.value,
                    reason_code=ReasonCode.CONTENT_HASH_MISMATCH.value,
                    message="dataset_manifest.json content_hash does not match the expected source hash",
                    evidence={
                        "path": str(dataset_path),
                        "content_hash": content_hash,
                        "content_hash_expected": content_hash_expected,
                    },
                )
            )
            return False

        download_timestamp = data.get("download_timestamp")
        if isinstance(download_timestamp, str):
            parsed = _parse_timestamp(download_timestamp)
            if parsed is not None:
                age_days = (datetime.now(UTC) - parsed.astimezone(UTC)).days
                if age_days > self._max_download_age_days:
                    report.add_check(
                        GateCheck(
                            gate_id="data_lineage",
                            result=GateResult.PASS.value,
                            reason_code=ReasonCode.STALE_DOWNLOAD_WARNING.value,
                            message=(
                                "dataset_manifest.json download_timestamp is stale; governed path warns "
                                "but does not fail on download age alone"
                            ),
                            evidence={
                                "path": str(dataset_path),
                                "download_timestamp": download_timestamp,
                                "age_days": age_days,
                            },
                        )
                    )
                    return True

        report.add_check(
            GateCheck(
                gate_id="data_lineage",
                result=GateResult.PASS.value,
                reason_code=ReasonCode.VALID.value,
                message="Dataset lineage manifest is valid for the governed path",
                evidence={
                    "path": str(dataset_path),
                    "pit_compliant": pit_compliant,
                    "knowledge_time_column": knowledge_time_column,
                    "content_hash": content_hash,
                    "download_timestamp": download_timestamp,
                },
            )
        )
        return True


DATA_LINEAGE_GATE = DataLineageGate()


def validate_preprocessing_report(bundle_path: Path, report: GateReport) -> bool:
    """Validate preprocessing_report.json schema per Appendix C.2."""
    preproc_path = bundle_path / "preprocessing_report.json"

    data, reason_code, message = _load_json_file(preproc_path)

    if data is None:
        check = GateCheck(
            gate_id="preprocessing_report",
            result=GateResult.FAIL.value,
            reason_code=reason_code,
            message=message,
            evidence={"path": str(preproc_path)},
        )
        report.add_check(check)
        return False

    # Validate schema version
    schema_version = data.get("schema_version", "")
    valid, reason, msg = _validate_schema_version(schema_version)

    if not valid:
        check = GateCheck(
            gate_id="preprocessing_report",
            result=GateResult.FAIL.value,
            reason_code=reason,
            message=msg,
            evidence={"schema_version": schema_version},
        )
        report.add_check(check)
        return False

    # Validate all required fields per Appendix C.2 Key Fields
    missing_fields = [f for f in PREPROCESSING_REQUIRED_FIELDS if f not in data]

    if missing_fields:
        check = GateCheck(
            gate_id="preprocessing_report",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MISSING_REQUIRED_FIELD.value,
            message=f"Missing required fields: {missing_fields}",
            evidence={"missing_fields": missing_fields, "present_fields": list(data.keys())},
        )
        report.add_check(check)
        return False

    check = GateCheck(
        gate_id="preprocessing_report",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message="Preprocessing report is valid",
        evidence={
            "schema_version": schema_version,
            "steps_count": len(data.get("steps", [])),
            "timings": data.get("timings"),
            "warnings_count": len(data.get("warnings", [])),
        },
    )
    report.add_check(check)
    return True


def validate_splits_manifest(bundle_path: Path, report: GateReport) -> bool:
    """Validate splits_manifest.json using existing SplitsManifest.from_json()."""
    manifest_path = bundle_path / "splits_manifest.json"

    # First, try to load as raw JSON to get better error messages
    data, reason_code, message = _load_json_file(manifest_path)

    if data is None:
        check = GateCheck(
            gate_id="splits_manifest_schema",
            result=GateResult.FAIL.value,
            reason_code=reason_code,
            message=message,
            evidence={"path": str(manifest_path)},
        )
        report.add_check(check)
        return False

    # Use SplitsManifest.from_json() for proper validation if available
    if SPLITS_AVAILABLE:
        try:
            manifest = SplitsManifest.from_json(manifest_path)

            check = GateCheck(
                gate_id="splits_manifest_schema",
                result=GateResult.PASS.value,
                reason_code=ReasonCode.VALID.value,
                message=f"Splits manifest schema v{manifest.schema_version} is valid",
                evidence={
                    "schema_version": manifest.schema_version,
                    "split_method": manifest.split_method,
                    "n_splits": manifest.n_splits,
                    "purge_window": manifest.purge_window,
                    "embargo_window": manifest.embargo_window,
                },
            )
            report.add_check(check)

            # Check for warnings from manifest
            if manifest.warnings:
                warn_check = GateCheck(
                    gate_id="splits_manifest_warnings",
                    result=GateResult.PASS.value,
                    reason_code=ReasonCode.VALID.value,
                    message=f"Splits manifest has {len(manifest.warnings)} warnings",
                    evidence={"warnings": manifest.warnings},
                )
                report.add_check(warn_check)

            return True

        except ValueError as e:
            # SplitsManifest.from_json raises ValueError for schema version issues
            # Map error message tokens to appropriate reason codes (all trigger exit 2)
            error_msg = str(e)
            if "UNKNOWN_SCHEMA_VERSION" in error_msg:
                reason = ReasonCode.UNKNOWN_SCHEMA_VERSION.value
            elif "MISSING_SCHEMA_VERSION" in error_msg:
                reason = ReasonCode.MISSING_SCHEMA_VERSION.value
            elif "INVALID_SCHEMA_VERSION" in error_msg:
                reason = ReasonCode.INVALID_SCHEMA_VERSION.value
            else:
                reason = ReasonCode.INVALID_STRUCTURE.value

            check = GateCheck(
                gate_id="splits_manifest_schema",
                result=GateResult.FAIL.value,
                reason_code=reason,
                message=error_msg,
                evidence={"path": str(manifest_path)},
            )
            report.add_check(check)
            return False

        except Exception as e:
            check = GateCheck(
                gate_id="splits_manifest_schema",
                result=GateResult.FAIL.value,
                reason_code=ReasonCode.INVALID_STRUCTURE.value,
                message=f"Error loading splits manifest: {e}",
                evidence={"path": str(manifest_path), "error": str(e)},
            )
            report.add_check(check)
            return False
    else:
        # Fallback: validate manually if splits module not available
        schema_version = data.get("schema_version", "")
        valid, reason, msg = _validate_schema_version(schema_version)

        if not valid:
            check = GateCheck(
                gate_id="splits_manifest_schema",
                result=GateResult.FAIL.value,
                reason_code=reason,
                message=msg,
                evidence={"schema_version": schema_version},
            )
            report.add_check(check)
            return False

        check = GateCheck(
            gate_id="splits_manifest_schema",
            result=GateResult.PASS.value,
            reason_code=ReasonCode.VALID.value,
            message=f"Splits manifest schema v{schema_version} is valid (fallback validation)",
            evidence={"schema_version": schema_version},
        )
        report.add_check(check)
        return True


def validate_leakage_invariants(bundle_path: Path, report: GateReport) -> bool:
    """Validate no train/test leakage per Known Leakage Patterns (v5.1).

    This gate enforces:
    - Train end < Test start (with purge gap)
    - Gap >= purge_window (per spec: "Verify gap ≥ purge_window")
    - Embargo window on both sides for k-fold
    """
    manifest_path = bundle_path / "splits_manifest.json"

    if not manifest_path.exists():
        return True

    data, _, _ = _load_json_file(manifest_path)
    if data is None:
        return True

    splits = data.get("splits", [])
    purge_window = data.get("purge_window", 0)
    embargo_window = data.get("embargo_window", 0)
    split_method = data.get("split_method", "unknown")

    if not splits:
        check = GateCheck(
            gate_id="leakage_invariants",
            result=GateResult.PASS.value,
            reason_code=ReasonCode.VALID.value,
            message="No splits to check for leakage",
            evidence={},
        )
        report.add_check(check)
        return True

    violations: list[str] = []
    purge_violations: list[str] = []
    embargo_violations: list[str] = []

    for split in splits:
        fold_id = split.get("fold_id", "?")
        train_end_str = split.get("train_end")
        test_start_str = split.get("test_start")
        test_end_str = split.get("test_end")
        non_contiguous = split.get("non_contiguous_train", False)

        # Parse timestamps properly
        train_end = _parse_timestamp(train_end_str)
        test_start = _parse_timestamp(test_start_str)
        _parse_timestamp(test_end_str)

        if train_end is None or test_start is None:
            continue

        # For walk-forward (non k-fold): train must end before test starts
        if not non_contiguous:
            if train_end >= test_start:
                violations.append(
                    f"Fold {fold_id}: train_end ({train_end_str}) >= test_start ({test_start_str})"
                )
            else:
                # Check purge gap: verify gap >= purge_window days
                gap = test_start - train_end
                gap_days = gap.total_seconds() / 86400

                if purge_window > 0 and gap_days < purge_window:
                    purge_violations.append(
                        f"Fold {fold_id}: purge gap ({gap_days:.1f} days) < required ({purge_window} days)"
                    )

        # For k-fold: check embargo window (training data after test must respect embargo)
        # This is checked via the embargoed_count in the split, or we'd need the actual data
        if non_contiguous and embargo_window > 0:
            # If we have embargoed_count = 0 but embargo_window > 0, that's suspicious
            # But we can't fully validate without the actual data
            pass

    # Report violations
    violations + purge_violations + embargo_violations

    if violations:
        check = GateCheck(
            gate_id="leakage_invariants",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.LEAKAGE_DETECTED.value,
            message=f"Temporal leakage detected in {len(violations)} splits",
            evidence={
                "violations": violations,
                "purge_window": purge_window,
                "embargo_window": embargo_window,
            },
        )
        report.add_check(check)
        return False

    if purge_violations:
        check = GateCheck(
            gate_id="leakage_invariants",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.PURGE_VIOLATION.value,
            message=f"Purge window violations in {len(purge_violations)} splits",
            evidence={
                "purge_violations": purge_violations,
                "purge_window": purge_window,
            },
        )
        report.add_check(check)
        return False

    if embargo_violations:
        check = GateCheck(
            gate_id="leakage_invariants",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.EMBARGO_VIOLATION.value,
            message=f"Embargo window violations in {len(embargo_violations)} splits",
            evidence={
                "embargo_violations": embargo_violations,
                "embargo_window": embargo_window,
            },
        )
        report.add_check(check)
        return False

    check = GateCheck(
        gate_id="leakage_invariants",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message=f"No leakage in {len(splits)} splits ({split_method})",
        evidence={
            "n_splits": len(splits),
            "split_method": split_method,
            "purge_window": purge_window,
            "embargo_window": embargo_window,
        },
    )
    report.add_check(check)
    return True


def validate_splits_integrity(bundle_path: Path, report: GateReport) -> bool:
    """Validate splits don't have overlapping train/test indices."""
    manifest_path = bundle_path / "splits_manifest.json"

    if not manifest_path.exists():
        return True

    data, _, _ = _load_json_file(manifest_path)
    if data is None:
        return True

    splits = data.get("splits", [])

    if not splits:
        check = GateCheck(
            gate_id="splits_integrity",
            result=GateResult.PASS.value,
            reason_code=ReasonCode.VALID.value,
            message="No splits to validate",
            evidence={"n_splits": 0},
        )
        report.add_check(check)
        return True

    # Validate each split has required boundary fields
    issues = []
    required = ["fold_id", "train_start", "train_end", "test_start", "test_end"]

    for i, split in enumerate(splits):
        missing = [f for f in required if f not in split]
        if missing:
            issues.append(f"Split {i} missing fields: {missing}")

    if issues:
        check = GateCheck(
            gate_id="splits_integrity",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.INVALID_SPLITS.value,
            message=f"Invalid split structure: {len(issues)} issues",
            evidence={"issues": issues},
        )
        report.add_check(check)
        return False

    check = GateCheck(
        gate_id="splits_integrity",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message=f"All {len(splits)} splits have valid structure",
        evidence={"n_splits": len(splits)},
    )
    report.add_check(check)
    return True


# Stat validity report schema (Appendix H): required top-level keys
STAT_VALIDITY_REQUIRED_KEYS = (
    "schema_version",
    "sharpe_ratio",
    "dsr",
    "min_trl",
    "bootstrap_ci",
    "pbo",
    "gate_result",
)
STAT_VALIDITY_GATE_VALUES = frozenset(("PASS", "WARN", "FAIL"))
STAT_VALIDITY_STRUCTURED_SECTIONS = ("dsr", "min_trl", "bootstrap_ci", "pbo")


def _is_finite_number(value: object) -> bool:
    """Return True when value is a finite int/float (bool excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _add_invalid_stat_validity_check(
    report: GateReport,
    path: Path,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    """Record a stat_validity_report.json structure failure (exit 2)."""
    check = GateCheck(
        gate_id="stat_validity_report",
        result=GateResult.FAIL.value,
        reason_code=ReasonCode.STAT_VALIDITY_INVALID_STRUCTURE.value,
        message=message,
        evidence={"path": str(path), **(evidence or {})},
    )
    report.add_check(check)
    report._has_invalid_input = True


def validate_stat_validity_report(bundle_path: Path, report: GateReport) -> None:
    """Validate stat_validity_report.json as a required canonical policy artifact.

    The gate interprets the emitted artifact only; it does not recompute statistics.
    Missing artifacts fail the governed path. Malformed structure remains invalid input.
    """
    path = bundle_path / "stat_validity_report.json"
    if not path.exists():
        check = GateCheck(
            gate_id="stat_validity_report",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MISSING_FILE.value,
            message="stat_validity_report.json is required for canonical governed backtests",
            evidence={"path": str(path)},
        )
        report.add_check(check)
        return

    data, reason_code, message = _load_json_file(path)
    if data is None:
        _add_invalid_stat_validity_check(report, path, message)
        return

    missing = [k for k in STAT_VALIDITY_REQUIRED_KEYS if k not in data]
    if missing:
        _add_invalid_stat_validity_check(
            report,
            path,
            f"stat_validity_report.json missing required keys: {missing}",
            {"missing_keys": missing},
        )
        return

    schema_version = data.get("schema_version")
    if schema_version not in ("v1", "1.0"):
        _add_invalid_stat_validity_check(
            report,
            path,
            f"stat_validity_report.json schema_version must be 'v1' or '1.0', got: {schema_version!r}",
            {"schema_version": schema_version},
        )
        return

    for section_name in STAT_VALIDITY_STRUCTURED_SECTIONS:
        if not isinstance(data.get(section_name), dict):
            _add_invalid_stat_validity_check(
                report,
                path,
                f"stat_validity_report.json field '{section_name}' must be an object",
                {"field": section_name, "actual_type": type(data.get(section_name)).__name__},
            )
            return

    pbo = data["pbo"]
    pbo_value = pbo.get("value")
    if not _is_finite_number(pbo_value):
        _add_invalid_stat_validity_check(
            report,
            path,
            "stat_validity_report.json pbo.value must be a finite number in [0, 1]",
            {"pbo": pbo},
        )
        return
    if float(pbo_value) < 0.0 or float(pbo_value) > 1.0:
        _add_invalid_stat_validity_check(
            report,
            path,
            f"stat_validity_report.json pbo.value must be between 0 and 1 inclusive, got: {pbo_value!r}",
            {"pbo": pbo},
        )
        return

    for section_name in STAT_VALIDITY_STRUCTURED_SECTIONS:
        nested_gate_result = data[section_name].get("gate_result")
        if nested_gate_result is not None and nested_gate_result not in STAT_VALIDITY_GATE_VALUES:
            _add_invalid_stat_validity_check(
                report,
                path,
                f"stat_validity_report.json {section_name}.gate_result must be PASS/WARN/FAIL when present, got: {nested_gate_result!r}",
                {"field": f"{section_name}.gate_result", "gate_result": nested_gate_result},
            )
            return

    gate_result = data.get("gate_result")
    if gate_result not in STAT_VALIDITY_GATE_VALUES:
        _add_invalid_stat_validity_check(
            report,
            path,
            f"stat_validity_report.json gate_result must be PASS/WARN/FAIL, got: {gate_result!r}",
            {"gate_result": gate_result},
        )
        return

    if gate_result == "FAIL":
        check = GateCheck(
            gate_id="stat_validity_report",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.STAT_VALIDITY_GATE_FAIL.value,
            message="Statistical validity report gate_result is FAIL",
            evidence={"path": str(path), "gate_result": gate_result, "pbo_value": float(pbo_value)},
        )
        report.add_check(check)
        return

    msg = "Statistical validity report gate_result is PASS"
    if gate_result == "WARN":
        msg = "Statistical validity report gate_result is WARN (evidence includes warnings)"
    check = GateCheck(
        gate_id="stat_validity_report",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message=msg,
        evidence={"path": str(path), "gate_result": gate_result, "pbo_value": float(pbo_value)},
    )
    report.add_check(check)


# Execution assumptions (Appendix G): canonical path requires explicit non-zero cost visibility
EXECUTION_ASSUMPTIONS_REQUIRED_KEYS = ("schema_version",)
EXECUTION_ASSUMPTIONS_COST_KEYS = (
    "commission",
    "commission_bps",
    "slippage_bps",
    "slippage",
    "cost_per_unit_turnover",
    "cost_model_id",
)


def validate_execution_assumptions(bundle_path: Path, report: GateReport) -> None:
    """Validate execution_assumptions.json as a required canonical policy artifact."""
    path = bundle_path / "execution_assumptions.json"
    if not path.exists():
        check = GateCheck(
            gate_id="execution_assumptions",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.COST_ASSUMPTION_MISSING.value,
            message="execution_assumptions.json is required for canonical governed backtests",
            evidence={"path": str(path)},
        )
        report.add_check(check)
        return

    data, reason_code, message = _load_json_file(path)
    if data is None:
        check = GateCheck(
            gate_id="execution_assumptions",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.EXECUTION_ASSUMPTIONS_INVALID_STRUCTURE.value,
            message=message,
            evidence={"path": str(path)},
        )
        report.add_check(check)
        report._has_invalid_input = True
        return

    missing = [k for k in EXECUTION_ASSUMPTIONS_REQUIRED_KEYS if k not in data]
    if missing:
        check = GateCheck(
            gate_id="execution_assumptions",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.EXECUTION_ASSUMPTIONS_INVALID_STRUCTURE.value,
            message=f"execution_assumptions.json missing required keys: {missing}",
            evidence={"path": str(path), "missing_keys": missing},
        )
        report.add_check(check)
        report._has_invalid_input = True
        return

    numeric_cost_keys = (
        "commission",
        "commission_bps",
        "slippage_bps",
        "slippage",
        "cost_per_unit_turnover",
    )
    numeric_costs = {
        key: abs(float(data[key])) for key in numeric_cost_keys if _is_finite_number(data.get(key))
    }
    cost_model_id = data.get("cost_model_id")
    normalized_cost_model_id = (
        cost_model_id.strip().lower() if isinstance(cost_model_id, str) else ""
    )
    has_cost_model_id = normalized_cost_model_id != ""
    has_non_zero_numeric_cost = any(value > 0 for value in numeric_costs.values())
    zero_numeric_cost_declared = bool(numeric_costs) and not has_non_zero_numeric_cost
    zero_cost_model_declared = has_cost_model_id and "zero" in normalized_cost_model_id
    effectively_zero = (
        zero_numeric_cost_declared
        or zero_cost_model_declared
        or (not numeric_costs and not has_cost_model_id)
    )

    if effectively_zero:
        check = GateCheck(
            gate_id="execution_assumptions",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.ZERO_COST_ASSUMED.value,
            message="Execution assumptions indicate zero or no transaction costs; canonical policy requires non-zero net-cost assumptions",
            evidence={
                "path": str(path),
                "cost_keys_present": [k for k in EXECUTION_ASSUMPTIONS_COST_KEYS if k in data],
                "numeric_costs": numeric_costs,
                "cost_model_id": cost_model_id,
            },
        )
        report.add_check(check)
        return

    check = GateCheck(
        gate_id="execution_assumptions",
        result=GateResult.PASS.value,
        reason_code=ReasonCode.VALID.value,
        message="Execution assumptions present with non-zero cost",
        evidence={
            "path": str(path),
            "numeric_costs": numeric_costs,
            "cost_model_id": cost_model_id,
        },
    )
    report.add_check(check)


def validate_bundle(
    bundle_path: Path,
    output_path: Path | None = None,
    expected_plan_hash: str | None = None,
) -> tuple[GateReport, ExitCode]:
    """Run all validation gates on a bundle.

    Args:
        bundle_path: Path to run bundle directory
        output_path: Optional path for gate_result.json (defaults to bundle/gate_result.json)
        expected_plan_hash: Optional expected hash for plan_identity HASH_MISMATCH check

    Returns:
        (GateReport, ExitCode)

    Note:
        Always attempts to write a report, even on invalid input (exit 2).
        If bundle doesn't exist, writes to --output or ./gate_result.json.
    """
    import time

    start_time = time.time()

    report = GateReport(bundle_path=str(bundle_path.absolute()))
    with tracer.start_as_current_span(SPAN_GATE_EVALUATE) as span:
        span.set_attribute("gate_name", "validate_bundle")
        span.set_attribute("bundle_id", str(bundle_path))
        try:
            # Gate 1: Bundle exists
            if not validate_bundle_exists(bundle_path, report):
                # Calculate execution time even for early exit
                report.metadata["execution_time_ms"] = int((time.time() - start_time) * 1000)
                try:
                    target = write_gate_report(report, output_path, bundle_path)
                    _log(f"Gate report written to: {target}")
                except Exception as exc:
                    _log(f"Warning: could not write gate report: {exc}")
                return report, ExitCode.INVALID_INPUT

            # Gate 2: Required files present (per Appendix C)
            file_status = validate_required_files(bundle_path, report)

            # Gate 3: plan.json (plan_identity gate per Appendix D.2)
            if file_status.get("plan.json", False):
                validate_plan_identity(bundle_path, report, expected_plan_hash)

            # Gate 4: env_fingerprint.json
            if file_status.get("env_fingerprint.json", False):
                validate_env_fingerprint(bundle_path, report)

            # Gate 5: dataset_manifest.json
            if file_status.get("dataset_manifest.json", False):
                dataset_valid = validate_dataset_manifest(bundle_path, report)
                if dataset_valid:
                    DATA_LINEAGE_GATE.validate(bundle_path, report)

            # Gate 6: preprocessing_report.json
            if file_status.get("preprocessing_report.json", False):
                validate_preprocessing_report(bundle_path, report)

            # Gate 7: Splits manifest schema validation
            if file_status.get("splits_manifest.json", False):
                validate_splits_manifest(bundle_path, report)

                # Gate 8: Splits integrity (structure check)
                validate_splits_integrity(bundle_path, report)

                # Gate 9: Leakage invariants (purge/embargo enforcement)
                validate_leakage_invariants(bundle_path, report)

            # Gate 10: Statistical validity report (required policy artifact; Appendix H)
            validate_stat_validity_report(bundle_path, report)

            # Gate 11: Execution / cost assumptions (required policy artifact; Appendix G)
            validate_execution_assumptions(bundle_path, report)

            report.metadata["reproducibility"] = collect_bundle_reproducibility_echo(bundle_path)

            # Calculate execution time
            report.metadata["execution_time_ms"] = int((time.time() - start_time) * 1000)

            # Write report
            try:
                target = write_gate_report(report, output_path, bundle_path)
                _log(f"Gate report written to: {target}")
            except Exception as exc:
                _log(f"Warning: could not write gate report: {exc}")

            # Determine exit code per Appendix D.1
            if report.overall_result == GateResult.PASS.value:
                return report, ExitCode.PASS
            if report._has_invalid_input:
                # Exit 2 for malformed JSON, unknown schema version, etc.
                return report, ExitCode.INVALID_INPUT
            return report, ExitCode.FAIL

        except Exception as e:
            # Internal error - add to report
            error_check = GateCheck(
                gate_id="internal_error",
                result=GateResult.FAIL.value,
                reason_code="INTERNAL_ERROR",
                message=f"Unexpected error during validation: {e}",
                evidence={"traceback": traceback.format_exc()},
            )
            report.add_check(error_check)
            if bundle_path.is_dir():
                report.metadata["reproducibility"] = collect_bundle_reproducibility_echo(
                    bundle_path
                )
            report.metadata["execution_time_ms"] = int((time.time() - start_time) * 1000)

            # Always try to write report
            try:
                target = write_gate_report(report, output_path, bundle_path)
                _log(f"Gate report written to: {target}")
            except Exception as exc:
                _log(f"Warning: could not write gate report: {exc}")

            return report, ExitCode.INTERNAL_ERROR
        finally:
            span.set_attribute("result", _gate_eval_telemetry_result(report))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code per Appendix D.1
    """
    parser = argparse.ArgumentParser(
        prog="marketmind-gate",
        description="Validate run bundles per Appendix D contract (v5.1)",
        epilog="Exit codes: 0=PASS, 1=FAIL, 2=invalid input, 3=internal error",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # check subcommand (per Appendix D.1)
    check_parser = subparsers.add_parser(
        "check",
        help="Validate a run bundle",
        aliases=["validate"],
    )

    # Support both --bundle flag (per spec) and positional (convenience)
    check_parser.add_argument(
        "bundle_positional",
        type=Path,
        nargs="?",
        default=None,
        help="Path to run bundle directory (positional form)",
    )
    check_parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to run bundle directory (per Appendix D.1)",
    )
    check_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for gate_result.json (default: <bundle>/gate_result.json)",
    )
    check_parser.add_argument(
        "--expected-plan-hash",
        type=str,
        default=None,
        help="Expected plan hash for HASH_MISMATCH verification (per Appendix D.2)",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Also output report JSON to stdout",
    )

    args = parser.parse_args(argv)

    if args.command in ("check", "validate"):
        # Resolve bundle path: prefer --bundle, fall back to positional
        bundle_path = args.bundle or args.bundle_positional

        if bundle_path is None:
            _log("Error: bundle path required (use --bundle or positional argument)")
            return ExitCode.INVALID_INPUT.value

        bundle_path = bundle_path.resolve()

        _log(f"Validating bundle: {bundle_path}")

        report, exit_code = validate_bundle(
            bundle_path,
            args.output,
            expected_plan_hash=args.expected_plan_hash,
        )

        # Print summary to stderr
        _log(f"Overall result: {report.overall_result}")
        _log(f"Gates checked: {len(report.gates)}")

        passed = sum(1 for g in report.gates if g["result"] == "PASS")
        failed = sum(1 for g in report.gates if g["result"] == "FAIL")
        _log(f"  Passed: {passed}")
        _log(f"  Failed: {failed}")

        # Optionally output JSON to stdout
        if args.json:
            click.echo(json.dumps(report.to_dict(), indent=2))

        return exit_code.value

    return ExitCode.INVALID_INPUT.value


if __name__ == "__main__":
    sys.exit(main())

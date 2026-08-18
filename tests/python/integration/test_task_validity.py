"""Integration coverage for the task-validity diagnostic controls."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from pysrc.validation.task_validity import (
    check_episode_construction_validity,
    check_leakage_geometry,
    check_task_non_exchangeability,
    validate_task,
)

pytestmark = [pytest.mark.integration, pytest.mark.determinism("d1")]

REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "bundles" / "valid_synthetic"
LEAKAGE_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "bundles" / "injected_leakage"
SCHEMA_PATH = REPO_ROOT / "schemas" / "task_validity_report.schema.json"


def _check_by_id(report_checks: list[object], check_id: str) -> dict[str, object]:
    for check in report_checks:
        if isinstance(check, dict) and check.get("check_id") == check_id:
            return check
    raise AssertionError(f"missing check {check_id}")


def test_positive_control_passes(deterministic_seed: int) -> None:
    """Diagnostic returns all_pass=true on clean synthetic bundle."""
    _ = deterministic_seed
    report = validate_task(VALID_BUNDLE)
    payload = report.to_json_dict()

    assert payload["all_pass"] is True
    assert all(check["passed"] for check in payload["checks"])


def test_negative_control_catches_leakage(deterministic_seed: int) -> None:
    """Diagnostic returns all_pass=false and TV-02 names the leakage vector."""
    _ = deterministic_seed
    report = validate_task(LEAKAGE_BUNDLE)
    payload = report.to_json_dict()
    tv02 = _check_by_id(payload["checks"], "TV-02")
    evidence = tv02["evidence"]

    assert payload["all_pass"] is False
    assert tv02["passed"] is False
    assert isinstance(evidence, dict)
    assert evidence["leaking_feature_count"] == 1
    assert evidence["leaking_features"][0]["feature_name"] == "tomorrow_return"
    assert evidence["leaking_features"][0]["lookahead_bars"] == 1


def test_output_validates_against_schema(deterministic_seed: int) -> None:
    """Diagnostic output passes JSON Schema validation for both controls."""
    _ = deterministic_seed
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    for bundle in (VALID_BUNDLE, LEAKAGE_BUNDLE):
        jsonschema.validate(validate_task(bundle).to_json_dict(), schema)


def test_missing_plan_timestamp_fails_closed(tmp_path: Path, deterministic_seed: int) -> None:
    """Missing plan timestamp provenance is not converted to a fake epoch timestamp."""
    _ = deterministic_seed
    bundle = tmp_path / "missing_plan"
    shutil.copytree(VALID_BUNDLE, bundle)
    (bundle / "plan.json").unlink()

    with pytest.raises(ValueError, match="plan.json.*as_of_time"):
        validate_task(bundle)


def test_tv03_reports_seconds_and_day_equivalents(deterministic_seed: int) -> None:
    """TV-03 evidence names raw seconds separately from day-equivalent values."""
    _ = deterministic_seed
    check = check_episode_construction_validity(VALID_BUNDLE)
    evidence = check.evidence

    assert evidence["minimum_train_test_gap_seconds"] == 259200
    assert evidence["minimum_train_test_gap_days_equivalent"] == 3
    assert "minimum_train_test_gap_days" not in evidence
    assert "interpreted as days" in str(evidence["threshold_source"])


def test_individual_checks_are_independently_callable(deterministic_seed: int) -> None:
    """Each TV-XX check can be invoked alone and returns valid evidence."""
    _ = deterministic_seed
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    checks = [
        check_task_non_exchangeability(VALID_BUNDLE),
        check_leakage_geometry(VALID_BUNDLE),
        check_episode_construction_validity(VALID_BUNDLE),
    ]

    for check in checks:
        payload = {
            "bundle_hash": "sha256:" + ("0" * 64),
            "checks": [check.to_json_dict()],
            "all_pass": check.passed,
            "diagnostic_version": "1.0.0",
            "timestamp": "2026-04-10T12:00:00Z",
        }
        jsonschema.validate(payload, schema)
        assert isinstance(check.evidence, dict)
        assert "threshold_source" in check.evidence

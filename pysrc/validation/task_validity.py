"""Task-validity diagnostic for governed run bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import jsonschema

DIAGNOSTIC_VERSION = "1.0.0"
TASK_VALIDITY_INPUTS = "task_validity_inputs.json"
SCHEMA_RELATIVE_PATH = Path("schemas/task_validity_report.schema.json")

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class TaskValidityCheck:
    """Result for one independently callable task-validity check."""

    check_id: str
    check_name: str
    passed: bool
    evidence: JsonObject

    def to_json_dict(self) -> JsonObject:
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "passed": self.passed,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TaskValidityReport:
    """Schema-validated aggregate task-validity report."""

    bundle_hash: str
    checks: tuple[TaskValidityCheck, ...]
    all_pass: bool
    diagnostic_version: str
    timestamp: str

    def to_json_dict(self) -> JsonObject:
        return {
            "bundle_hash": self.bundle_hash,
            "checks": [check.to_json_dict() for check in self.checks],
            "all_pass": self.all_pass,
            "diagnostic_version": self.diagnostic_version,
            "timestamp": self.timestamp,
        }


def validate_task(bundle_path: Path) -> TaskValidityReport:
    """
    Run all task-validity checks against a run bundle.

    The function is a pure function of bundle contents: hash, timestamp, checks,
    and threshold sources are all read from the bundle or repo schemas.
    """
    bundle = Path(bundle_path)
    checks = (
        check_task_non_exchangeability(bundle),
        check_leakage_geometry(bundle),
        check_episode_construction_validity(bundle),
    )
    report = TaskValidityReport(
        bundle_hash=_bundle_sha256(bundle),
        checks=checks,
        all_pass=all(check.passed for check in checks),
        diagnostic_version=DIAGNOSTIC_VERSION,
        timestamp=_report_timestamp(bundle),
    )
    _validate_report_schema(report)
    return report


def check_task_non_exchangeability(bundle_path: Path) -> TaskValidityCheck:
    """TV-01: right-tail empirical permutation test for non-exchangeability."""
    inputs = _load_task_validity_inputs(bundle_path)
    section = _require_object(inputs.get("task_non_exchangeability"), "task_non_exchangeability")
    observed = _require_number(section.get("observed_statistic"), "observed_statistic")
    null_values = _require_number_list(section.get("null_statistics"), "null_statistics")
    threshold = _require_number(section.get("p_value_threshold"), "p_value_threshold")
    threshold_source = _require_text(section.get("threshold_source"), "threshold_source")

    exceedances = sum(1 for value in null_values if value >= observed)
    p_value = (exceedances + 1.0) / (len(null_values) + 1.0)
    evidence: JsonObject = {
        "observed_statistic": observed,
        "null_draw_count": len(null_values),
        "null_exceedance_count": exceedances,
        "permutation_test_p_value": p_value,
        "threshold": threshold,
        "threshold_source": threshold_source,
    }
    return TaskValidityCheck(
        check_id="TV-01",
        check_name="Task non-exchangeability",
        passed=p_value < threshold,
        evidence=evidence,
    )


def check_leakage_geometry(bundle_path: Path) -> TaskValidityCheck:
    """TV-02: feature-level lookahead check relative to episode boundary."""
    features = _load_feature_lookahead_records(bundle_path)
    leaking: list[JsonValue] = []
    max_lookahead = 0.0
    for feature in features:
        lookahead = _require_number(feature.get("lookahead_bars"), "lookahead_bars")
        max_lookahead = max(max_lookahead, lookahead)
        if lookahead > 0:
            leaking.append(
                {
                    "feature_name": _require_text(feature.get("feature_name"), "feature_name"),
                    "lookahead_bars": lookahead,
                    "boundary": str(feature.get("boundary", "episode_query_start")),
                }
            )

    evidence: JsonObject = {
        "feature_count": len(features),
        "leaking_feature_count": len(leaking),
        "max_lookahead_bars": max_lookahead,
        "leaking_features": leaking,
        "threshold": 0,
        "threshold_source": "docs/rg09/rg09_gate_spec.md Sections 1-2; feature vectors must be admissible under DataView.as_of(decision_ts)",
    }
    return TaskValidityCheck(
        check_id="TV-02",
        check_name="Leakage geometry",
        passed=len(leaking) == 0,
        evidence=evidence,
    )


def check_episode_construction_validity(bundle_path: Path) -> TaskValidityCheck:
    """TV-03: split boundaries respect purge/embargo values from manifest."""
    manifest = _load_json_object(Path(bundle_path) / "splits_manifest.json")
    splits = _require_object_list(manifest.get("splits"), "splits")
    purge_window = _require_number(manifest.get("purge_window"), "purge_window")
    embargo_window = _require_number(manifest.get("embargo_window"), "embargo_window")
    split_method = _require_text(manifest.get("split_method"), "split_method")

    violations: list[JsonValue] = []
    minimum_gap_seconds: float | None = None
    for index, split in enumerate(splits):
        fold_id = split.get("fold", split.get("fold_id", index))
        train_end = _parse_timestamp(_require_text(split.get("train_end"), "train_end"))
        test_start = _parse_timestamp(_require_text(split.get("test_start"), "test_start"))
        test_end_raw = split.get("test_end")
        test_end = (
            _parse_timestamp(_require_text(test_end_raw, "test_end"))
            if test_end_raw is not None
            else None
        )

        if train_end >= test_start:
            violations.append(
                {
                    "fold": fold_id,
                    "violation": "train_end_not_before_test_start",
                    "train_end": train_end.isoformat(),
                    "test_start": test_start.isoformat(),
                }
            )
            continue

        gap_seconds = (test_start - train_end).total_seconds()
        gap_days_equivalent = _seconds_to_days(gap_seconds)
        minimum_gap_seconds = (
            gap_seconds if minimum_gap_seconds is None else min(minimum_gap_seconds, gap_seconds)
        )
        if gap_days_equivalent < purge_window:
            violations.append(
                {
                    "fold": fold_id,
                    "violation": "purge_gap_below_manifest",
                    "actual_gap_seconds": gap_seconds,
                    "actual_gap_days_equivalent": gap_days_equivalent,
                    "required_purge_seconds": _days_to_seconds(purge_window),
                    "required_purge_days_equivalent": purge_window,
                }
            )

        post_test_train = split.get("post_test_train_start")
        if post_test_train is not None and test_end is not None:
            post_train_start = _parse_timestamp(
                _require_text(post_test_train, "post_test_train_start")
            )
            embargo_gap_seconds = (post_train_start - test_end).total_seconds()
            embargo_gap_days_equivalent = _seconds_to_days(embargo_gap_seconds)
            if embargo_gap_days_equivalent < embargo_window:
                violations.append(
                    {
                        "fold": fold_id,
                        "violation": "embargo_gap_below_manifest",
                        "actual_gap_seconds": embargo_gap_seconds,
                        "actual_gap_days_equivalent": embargo_gap_days_equivalent,
                        "required_embargo_seconds": _days_to_seconds(embargo_window),
                        "required_embargo_days_equivalent": embargo_window,
                    }
                )

    minimum_gap_days_equivalent = (
        None if minimum_gap_seconds is None else _seconds_to_days(minimum_gap_seconds)
    )
    evidence: JsonObject = {
        "split_method": split_method,
        "split_count": len(splits),
        "purge_window": purge_window,
        "embargo_window": embargo_window,
        "minimum_train_test_gap_seconds": minimum_gap_seconds,
        "minimum_train_test_gap_days_equivalent": minimum_gap_days_equivalent,
        "violation_count": len(violations),
        "violations": violations,
        "threshold_source": (
            "splits_manifest.json purge_window and embargo_window interpreted as days "
            "under the current Appendix C run-bundle contract"
        ),
    }
    return TaskValidityCheck(
        check_id="TV-03",
        check_name="Episode construction validity",
        passed=len(violations) == 0,
        evidence=evidence,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _repo_root() / SCHEMA_RELATIVE_PATH


def _validate_report_schema(report: TaskValidityReport) -> None:
    schema = _load_json_object(_schema_path())
    jsonschema.validate(report.to_json_dict(), schema)


def _bundle_sha256(bundle_path: Path) -> str:
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"bundle_path must be a directory: {bundle_path}")
    digest = hashlib.sha256()
    for path in sorted(item for item in bundle_path.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle_path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _report_timestamp(bundle_path: Path) -> str:
    plan_path = bundle_path / "plan.json"
    if not plan_path.exists():
        raise ValueError("plan.json with as_of_time is required for task-validity report timestamp")
    try:
        plan = _load_json_object(plan_path)
    except json.JSONDecodeError as exc:
        raise ValueError("plan.json with as_of_time must be valid JSON") from exc
    as_of = plan.get("as_of_time")
    if not isinstance(as_of, str) or not as_of.strip():
        raise ValueError("plan.json as_of_time is required for task-validity report timestamp")
    return _format_utc_timestamp(_parse_timestamp(as_of))


def _format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _days_to_seconds(days: float) -> float:
    return days * 86400.0


def _seconds_to_days(seconds: float) -> float:
    return seconds / 86400.0


def _load_task_validity_inputs(bundle_path: Path) -> JsonObject:
    return _load_json_object(bundle_path / TASK_VALIDITY_INPUTS)


def _load_feature_lookahead_records(bundle_path: Path) -> list[JsonObject]:
    inputs_path = bundle_path / TASK_VALIDITY_INPUTS
    if inputs_path.exists():
        inputs = _load_json_object(inputs_path)
        features = inputs.get("features")
        if features is not None:
            return _require_object_list(features, "features")

    report = _load_json_object(bundle_path / "preprocessing_report.json")
    steps = _require_object_list(report.get("steps"), "steps")
    records: list[JsonObject] = []
    for step in steps:
        name = step.get("output", step.get("name"))
        records.append(
            {
                "feature_name": _require_text(name, "feature_name"),
                "lookahead_bars": _require_number(step.get("lookahead_bars", 0), "lookahead_bars"),
                "boundary": "episode_query_start",
            }
        )
    return records


def _load_json_object(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _require_object(cast(JsonValue, data), str(path))


def _require_object(value: JsonValue | object, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return cast(JsonObject, value)


def _require_object_list(value: JsonValue | object, field: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result: list[JsonObject] = []
    for index, item in enumerate(value):
        result.append(_require_object(item, f"{field}[{index}]"))
    return result


def _require_number_list(value: JsonValue | object, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON array")
    return [_require_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _require_number(value: JsonValue | object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _require_text(value: JsonValue | object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)

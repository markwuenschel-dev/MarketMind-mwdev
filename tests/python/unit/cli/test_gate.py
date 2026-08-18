from __future__ import annotations

import json
from pathlib import Path

import pytest

import marketmind_gate.cli as compatibility_cli
from pysrc.cli.gate import (
    ExitCode,
    ReasonCode,
    emit_gate_failure_report,
    main,
    validate_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _write_minimal_bundle(bundle_dir: Path) -> None:
    payloads = {
        "plan.json": {
            "schema_version": "1.0.0",
            "plan_hash": "sha256:plan",
            "as_of_time": "2026-03-21T00:00:00Z",
            "config_hash": "sha256:cfg",
            "determinism_tier": "D1",
            "planner_version": "pipeline_strategy/1",
            "feature_op_registry_version": "1.0.0",
        },
        "env_fingerprint.json": {
            "schema_version": "1.0.0",
            "python_version": "3.12.0",
            "git_sha": "abc123",
            "deps": {},
        },
        "dataset_manifest.json": {
            "schema_version": "1.0.0",
            "dataset_id": "fixture",
            "symbols": ["SPY"],
            "row_count": 10,
            "time_range": {"start": "2024-01-01", "end": "2024-01-10"},
            "pit_compliant": True,
            "knowledge_time_column": "knowledge_time",
        },
        "preprocessing_report.json": {
            "schema_version": "1.0.0",
            "steps": [],
            "timings": {},
            "warnings": [],
        },
        "splits_manifest.json": {
            "schema_version": "1.0.0",
            "split_method": "walk_forward",
            "purge_window": 0,
            "embargo_window": 0,
            "splits": [],
        },
        "stat_validity_report.json": {
            "schema_version": "v1",
            "sharpe_ratio": 1.1,
            "dsr": {
                "value": 0.8,
                "p_value": 0.01,
                "n_trials": 1,
                "skewness": 0.0,
                "excess_kurtosis": 0.0,
                "gate_result": "PASS",
            },
            "min_trl": {
                "years_needed": 1.0,
                "years_available": 4.0,
                "target_confidence": 0.95,
                "gate_result": "PASS",
            },
            "bootstrap_ci": {
                "lower_95": 0.1,
                "upper_95": 2.0,
                "lower_99": 0.0,
                "upper_99": 2.2,
                "n_resamples": 10,
                "block_size": 1,
                "gate_result": "PASS",
            },
            "pbo": {"value": 0.2, "gate_result": "PASS"},
            "gate_result": "PASS",
        },
        "execution_assumptions.json": {
            "schema_version": "v1",
            "commission_bps": 5.0,
            "cost_model_id": "fixed_5bps",
        },
    }
    for name, payload in payloads.items():
        (bundle_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_emit_gate_failure_report_writes_canonical_failure_shape(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    report = emit_gate_failure_report(
        bundle_dir,
        gate_id="cost_gate",
        reason_code=ReasonCode.COST_GATE_REJECTED.value,
        message="turnover rejected",
        evidence={"upstream_reason_code": "TURNOVER_LIMIT"},
    )

    payload = json.loads((bundle_dir / "gate_result.json").read_text(encoding="utf-8"))
    assert report.overall_result == "FAIL"
    assert payload["gates"][0]["reason_code"] == "COST_GATE_REJECTED"
    assert payload["gates"][0]["evidence"]["upstream_reason_code"] == "TURNOVER_LIMIT"


def test_marketmind_gate_cli_delegates_to_canonical_main() -> None:
    assert compatibility_cli.main is main


def test_main_returns_fail_for_governed_gate_failure_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_minimal_bundle(bundle_dir)
    payload = json.loads((bundle_dir / "stat_validity_report.json").read_text(encoding="utf-8"))
    payload["gate_result"] = "FAIL"
    for section in ("dsr", "min_trl", "bootstrap_ci", "pbo"):
        payload[section]["gate_result"] = "FAIL"
    (bundle_dir / "stat_validity_report.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["check", str(bundle_dir)])
    report = json.loads((bundle_dir / "gate_result.json").read_text(encoding="utf-8"))
    assert exit_code == ExitCode.FAIL.value
    assert report["overall_result"] == "FAIL"


def test_validate_bundle_rejects_invalid_determinism_tier(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_minimal_bundle(bundle_dir)
    payload = json.loads((bundle_dir / "plan.json").read_text(encoding="utf-8"))
    payload["determinism_tier"] = "D9"
    (bundle_dir / "plan.json").write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = validate_bundle(bundle_dir)
    assert exit_code is ExitCode.FAIL
    assert report.overall_result == "FAIL"
    assert any(g.get("reason_code") == "INVALID_DETERMINISM_TIER" for g in report.gates)


def test_validate_bundle_echoes_reproducibility_metadata(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_minimal_bundle(bundle_dir)
    report, exit_code = validate_bundle(bundle_dir)
    assert exit_code is ExitCode.PASS
    rep = report.metadata.get("reproducibility")
    assert isinstance(rep, dict)
    assert rep.get("determinism_tier") == "D1"
    assert rep.get("plan_hash") == "sha256:plan"
    assert rep.get("git_sha") == "abc123"


def test_validate_bundle_uses_invalid_input_exit_code_for_bad_json(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_minimal_bundle(bundle_dir)
    (bundle_dir / "plan.json").write_text("{bad json", encoding="utf-8")

    report, exit_code = validate_bundle(bundle_dir)
    assert exit_code is ExitCode.INVALID_INPUT
    assert report.overall_result == "FAIL"

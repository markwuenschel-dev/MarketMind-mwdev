"""Tests for gate framework core."""

import json
from pathlib import Path

import pytest

from marketmind_gate.gates.core import (
    GateResult,
    GateStatus,
    ValidationResult,
    gate_files_exist,
    gate_json_valid,
    gate_max_drawdown,
    gate_sharpe_threshold,
    validate_bundle,
)


def _write_governed_bundle_files(bundle: Path) -> None:
    bundle_files = {
        "plan.json": {
            "schema_version": "1.0.0",
            "plan_hash": "sha256:plan",
            "as_of_time": "2026-01-01T00:00:00Z",
            "config_hash": "sha256:cfg",
            "determinism_tier": "D2",
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
            "time_range": {"start": "2020-01-01", "end": "2020-01-10"},
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
            "split_method": "none",
            "purge_window": 0,
            "embargo_window": 0,
            "splits": [],
        },
        "stat_validity_report.json": {
            "schema_version": "v1",
            "sharpe_ratio": 1.0,
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
                "years_available": 5.0,
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
            "commission_bps": 5,
            "cost_model_id": "fixed_5bps",
        },
    }
    for name, payload in bundle_files.items():
        (bundle / name).write_text(json.dumps(payload))


def _update_stat_validity_gate_result(bundle: Path, gate_result: str) -> None:
    stat_validity_path = bundle / "stat_validity_report.json"
    payload = json.loads(stat_validity_path.read_text())
    payload["gate_result"] = gate_result
    for key in ("dsr", "min_trl", "bootstrap_ci", "pbo"):
        if isinstance(payload.get(key), dict):
            payload[key]["gate_result"] = gate_result
    stat_validity_path.write_text(json.dumps(payload))


@pytest.fixture
def valid_bundle(tmp_path) -> Path:
    """Create a valid bundle with passing metrics."""
    bundle = tmp_path / "valid_bundle"
    bundle.mkdir()
    _write_governed_bundle_files(bundle)

    result = {
        "schema_version": "1.0.0",
        "meta": {"strategy": "sma_crossover"},
        "result": {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.08,
            "win_rate": 0.55,
            "num_trades": 24,
        },
    }
    (bundle / "backtest_result.json").write_text(json.dumps(result))
    return bundle


@pytest.fixture
def empty_bundle(tmp_path) -> Path:
    """Create an empty bundle directory."""
    bundle = tmp_path / "empty_bundle"
    bundle.mkdir()
    return bundle


@pytest.fixture
def bad_sharpe_bundle(tmp_path) -> Path:
    """Create a bundle with failing Sharpe ratio."""
    bundle = tmp_path / "bad_sharpe"
    bundle.mkdir()
    _write_governed_bundle_files(bundle)

    result = {
        "result": {
            "sharpe_ratio": -5.0,
            "max_drawdown": -0.05,
        },
    }
    (bundle / "backtest_result.json").write_text(json.dumps(result))
    _update_stat_validity_gate_result(bundle, "FAIL")
    return bundle


@pytest.fixture
def bad_drawdown_bundle(tmp_path) -> Path:
    """Create a bundle with failing max drawdown."""
    bundle = tmp_path / "bad_drawdown"
    bundle.mkdir()
    _write_governed_bundle_files(bundle)

    result = {
        "result": {
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.75,  # Exceeds -50% limit
        },
    }
    (bundle / "backtest_result.json").write_text(json.dumps(result))
    _update_stat_validity_gate_result(bundle, "FAIL")
    return bundle


@pytest.fixture
def invalid_json_bundle(tmp_path) -> Path:
    """Create a bundle with malformed JSON."""
    bundle = tmp_path / "invalid_json"
    bundle.mkdir()
    _write_governed_bundle_files(bundle)
    (bundle / "backtest_result.json").write_text("{not valid json")
    return bundle


class TestGateResult:
    def test_pass_result(self):
        result = GateResult(gate_id="test", status=GateStatus.PASS)
        assert result.status == GateStatus.PASS
        assert result.reason is None

    def test_fail_result_with_reason(self):
        result = GateResult(
            gate_id="test",
            status=GateStatus.FAIL,
            reason="Something went wrong",
        )
        assert result.status == GateStatus.FAIL
        assert result.reason == "Something went wrong"


class TestValidationResult:
    def test_to_dict(self, valid_bundle):
        result = ValidationResult(
            bundle_path=valid_bundle,
            overall_status=GateStatus.PASS,
            gates=[GateResult(gate_id="test", status=GateStatus.PASS)],
        )
        d = result.to_dict()

        assert d["overall_status"] == "PASS"
        assert len(d["gates"]) == 1
        assert d["gates"][0]["gate_id"] == "test"


class TestGateFilesExist:
    def test_pass_when_file_exists(self, valid_bundle):
        result = gate_files_exist(valid_bundle)
        assert result.status == GateStatus.PASS

    def test_fail_when_file_missing(self, empty_bundle):
        result = gate_files_exist(empty_bundle)
        assert result.status == GateStatus.FAIL
        assert "Missing" in result.reason


class TestGateJsonValid:
    def test_pass_with_valid_json(self, valid_bundle):
        result = gate_json_valid(valid_bundle)
        assert result.status == GateStatus.PASS

    def test_fail_with_invalid_json(self, invalid_json_bundle):
        result = gate_json_valid(invalid_json_bundle)
        assert result.status == GateStatus.FAIL
        assert "Invalid JSON" in result.reason

    def test_fail_when_missing_result_field(self, tmp_path):
        bundle = tmp_path / "no_result"
        bundle.mkdir()
        (bundle / "backtest_result.json").write_text('{"meta": {}}')

        result = gate_json_valid(bundle)
        assert result.status == GateStatus.FAIL
        assert "result" in result.reason


class TestGateSharpeThreshold:
    def test_pass_above_threshold(self, valid_bundle):
        result = gate_sharpe_threshold(valid_bundle, min_sharpe=-2.0)
        assert result.status == GateStatus.PASS

    def test_fail_below_threshold(self, bad_sharpe_bundle):
        result = gate_sharpe_threshold(bad_sharpe_bundle, min_sharpe=-2.0)
        assert result.status == GateStatus.FAIL
        assert "-5.00" in result.reason

    def test_evidence_contains_sharpe(self, valid_bundle):
        result = gate_sharpe_threshold(valid_bundle)
        assert result.evidence is not None
        assert "sharpe_ratio" in result.evidence


class TestGateMaxDrawdown:
    def test_pass_within_limit(self, valid_bundle):
        result = gate_max_drawdown(valid_bundle, max_allowed=-0.50)
        assert result.status == GateStatus.PASS

    def test_fail_exceeds_limit(self, bad_drawdown_bundle):
        result = gate_max_drawdown(bad_drawdown_bundle, max_allowed=-0.50)
        assert result.status == GateStatus.FAIL
        assert "exceeds" in result.reason

    def test_evidence_contains_drawdown(self, valid_bundle):
        result = gate_max_drawdown(valid_bundle)
        assert result.evidence is not None
        assert "max_drawdown" in result.evidence


class TestValidateBundle:
    def test_all_pass(self, valid_bundle):
        result = validate_bundle(valid_bundle)
        assert result.overall_status == GateStatus.PASS
        assert all(g.status == GateStatus.PASS for g in result.gates)
        assert any(g.gate_id == "bundle_exists" for g in result.gates)

    def test_fail_on_missing_files(self, empty_bundle):
        result = validate_bundle(empty_bundle)
        assert result.overall_status == GateStatus.FAIL

    def test_fail_on_bad_sharpe(self, bad_sharpe_bundle):
        result = validate_bundle(bad_sharpe_bundle)
        assert result.overall_status == GateStatus.FAIL
        assert any(g.status == GateStatus.FAIL for g in result.gates)

    def test_fail_on_bad_drawdown(self, bad_drawdown_bundle):
        result = validate_bundle(bad_drawdown_bundle)
        assert result.overall_status == GateStatus.FAIL
        assert any(g.status == GateStatus.FAIL for g in result.gates)

    def test_partial_failure_still_runs_all_gates(self, bad_sharpe_bundle):
        """Even if one gate fails, all gates should run."""
        result = validate_bundle(bad_sharpe_bundle)
        assert len(result.gates) >= 1

"""Tests for Gate CLI validation per Appendix C/D (v5.1).

These tests verify the Gate CLI correctly validates run bundles
against the specifications in Appendix C (bundle structure) and 
Appendix D (gate contract) of MarketMind Implementation Plan v5.1.
"""

import json
import pytest
from pathlib import Path

from pysrc.cli.gate import (
    main,
    validate_bundle,
    validate_bundle_exists,
    validate_required_files,
    validate_plan_identity,
    validate_splits_manifest,
    validate_splits_integrity,
    validate_leakage_invariants,
    GateReport,
    GateCheck,
    GateResult,
    ReasonCode,
    ExitCode,
    GATE_SCHEMA_VERSION,
    GATE_CLI_VERSION,
    REQUIRED_BUNDLE_FILES,
    _validate_schema_version,
    _parse_timestamp,
)
from pysrc.ops.hashing.canonical_frame import CANONICAL_FRAME_CI_STATUS_VALUE


def _minimal_stat_validity_report(gate_result="PASS", pbo_value=0.22, pbo_gate_result="PASS"):
    """Minimal valid stat_validity_report.json per Appendix H (for gate schema checks)."""
    return {
        "schema_version": "v1",
        "sharpe_ratio": 1.2,
        "dsr": {"value": 0.9, "p_value": 0.01, "n_trials": 1, "skewness": 0.0, "excess_kurtosis": 0.0, "gate_result": "PASS"},
        "min_trl": {"years_needed": 1.0, "years_available": 5.0, "target_confidence": 0.95, "gate_result": "PASS"},
        "bootstrap_ci": {"lower_95": 0.5, "upper_95": 2.0, "lower_99": 0.2, "upper_99": 2.5, "n_resamples": 10000, "block_size": 1, "gate_result": "PASS"},
        "pbo": {"value": pbo_value, "gate_result": pbo_gate_result},
        "gate_result": gate_result,
    }


def _minimal_execution_assumptions(commission_bps=10, cost_model_id=None):
    """Minimal valid execution_assumptions.json (schema_version + cost info)."""
    out = {"schema_version": "v1"}
    if commission_bps is not None:
        out["commission_bps"] = commission_bps
    if cost_model_id is not None:
        out["cost_model_id"] = cost_model_id
    return out


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_bundle(tmp_path):
    """Create a minimal bundle directory."""
    bundle = tmp_path / "test_bundle"
    bundle.mkdir()
    return bundle


@pytest.fixture
def valid_plan():
    """Valid plan.json per Appendix C.2 Key Fields."""
    return {
        "schema_version": "1.0.0",
        "plan_hash": "sha256:abc123def456",
        "as_of_time": "2026-01-01T00:00:00Z",
        "config_hash": "sha256:config789",
    }


@pytest.fixture
def valid_env():
    """Valid env_fingerprint.json per Appendix C.2 Key Fields."""
    return {
        "schema_version": "1.0.0",
        "python_version": "3.12.0",
        "git_sha": "a1b2c3d4",
        "deps": {"polars": "0.20.0", "numpy": "1.26.0"},
    }


@pytest.fixture
def valid_dataset():
    """Valid dataset_manifest.json per Appendix C.2 Key Fields."""
    return {
        "schema_version": "1.0.0",
        "dataset_id": "spy_2020",
        "symbols": ["SPY"],
        "row_count": 1000,
        "time_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
    }


@pytest.fixture
def valid_preprocessing():
    """Valid preprocessing_report.json per Appendix C.2 Key Fields."""
    return {
        "schema_version": "1.0.0",
        "steps": [{"name": "load", "duration_ms": 50}],
        "timings": {"total_ms": 50},
        "warnings": [],
    }


@pytest.fixture
def valid_splits():
    """Valid splits_manifest.json per Appendix C."""
    return {
        "schema_version": "1.0.0",
        "split_method": "walk_forward",
        "n_splits": 2,
        "purge_window": 5,
        "embargo_window": 0,
        "splits": [
            {
                "fold_id": 0,
                "train_start": "2020-01-01T00:00:00",
                "train_end": "2020-06-20T00:00:00",
                "test_start": "2020-07-01T00:00:00",
                "test_end": "2020-08-31T00:00:00",
            },
            {
                "fold_id": 1,
                "train_start": "2020-01-01T00:00:00",
                "train_end": "2020-08-20T00:00:00",
                "test_start": "2020-09-01T00:00:00",
                "test_end": "2020-10-31T00:00:00",
            },
        ],
        "warnings": [],
    }


@pytest.fixture
def complete_bundle(temp_bundle, valid_plan, valid_env, valid_dataset, valid_preprocessing, valid_splits):
    """Create a canonical governed bundle with required base and policy artifacts."""
    files = {
        "plan.json": valid_plan,
        "env_fingerprint.json": valid_env,
        "dataset_manifest.json": valid_dataset,
        "preprocessing_report.json": valid_preprocessing,
        "splits_manifest.json": valid_splits,
        "stat_validity_report.json": _minimal_stat_validity_report("PASS"),
        "execution_assumptions.json": _minimal_execution_assumptions(commission_bps=5, cost_model_id="fixed_5bps"),
    }
    for name, content in files.items():
        with open(temp_bundle / name, "w") as f:
            json.dump(content, f)
    return temp_bundle


# ============================================================================
# Unit Tests: Helpers
# ============================================================================

class TestSchemaVersionValidation:
    def test_valid_version(self):
        valid, reason, _ = _validate_schema_version("1.0.0")
        assert valid
        assert reason == ReasonCode.VALID.value
    
    def test_missing_version(self):
        valid, reason, _ = _validate_schema_version("")
        assert not valid
        assert reason == ReasonCode.MISSING_SCHEMA_VERSION.value
    
    def test_unsupported_major(self):
        valid, reason, _ = _validate_schema_version("99.0.0")
        assert not valid
        assert reason == ReasonCode.UNKNOWN_SCHEMA_VERSION.value


class TestTimestampParsing:
    def test_z_suffix(self):
        ts = _parse_timestamp("2026-01-01T00:00:00Z")
        assert ts is not None
    
    def test_offset(self):
        ts = _parse_timestamp("2026-01-01T00:00:00+00:00")
        assert ts is not None
    
    def test_negative_offset(self):
        """Negative timezone offsets must work (was a bug)."""
        ts = _parse_timestamp("2026-01-01T00:00:00-05:00")
        assert ts is not None
        assert ts.hour == 0  # Should preserve the local hour
    
    def test_invalid(self):
        assert _parse_timestamp("invalid") is None


# ============================================================================
# Unit Tests: GateReport
# ============================================================================

class TestGateReport:
    def test_timestamp_uses_z_suffix(self):
        """Appendix D.2 shows timestamps with 'Z' suffix."""
        report = GateReport()
        assert report.timestamp.endswith("Z")
    
    def test_tracks_invalid_input(self):
        """MALFORMED_JSON etc should trigger exit code 2."""
        report = GateReport()
        check = GateCheck(
            gate_id="test",
            result=GateResult.FAIL.value,
            reason_code=ReasonCode.MALFORMED_JSON.value,
            message="Bad JSON",
        )
        report.add_check(check)
        assert report._has_invalid_input is True

    def test_metadata_carries_canonical_frame_ci_status(self):
        report = GateReport()
        assert report.metadata["canonical_frame_ci_status"] == CANONICAL_FRAME_CI_STATUS_VALUE


# ============================================================================
# Unit Tests: Required Files (Appendix C.2)
# ============================================================================

class TestRequiredBundleFiles:
    def test_matches_appendix_c(self):
        """Required files match Appendix C.2 spec."""
        expected = {
            "plan.json",
            "env_fingerprint.json",
            "dataset_manifest.json",
            "preprocessing_report.json",
            "splits_manifest.json",
        }
        assert set(REQUIRED_BUNDLE_FILES) == expected


class TestValidateRequiredFiles:
    def test_all_present(self, complete_bundle):
        report = GateReport()
        results = validate_required_files(complete_bundle, report)
        assert all(results.values())
        assert all(g["result"] == "PASS" for g in report.gates)
    
    def test_missing_plan(self, complete_bundle):
        (complete_bundle / "plan.json").unlink()
        report = GateReport()
        results = validate_required_files(complete_bundle, report)
        assert results["plan.json"] is False


# ============================================================================
# Unit Tests: plan_identity Gate (Appendix D.2)
# ============================================================================

class TestValidatePlanIdentity:
    def test_valid_plan(self, temp_bundle, valid_plan):
        with open(temp_bundle / "plan.json", "w") as f:
            json.dump(valid_plan, f)
        
        report = GateReport()
        result = validate_plan_identity(temp_bundle, report)
        
        assert result is True
        assert report.gates[0]["gate_id"] == "plan_identity"
        assert report.gates[0]["result"] == "PASS"
    
    def test_missing_plan_hash(self, temp_bundle):
        plan = {"schema_version": "1.0.0", "as_of_time": "2026-01-01T00:00:00Z", "config_hash": "x"}
        with open(temp_bundle / "plan.json", "w") as f:
            json.dump(plan, f)
        
        report = GateReport()
        result = validate_plan_identity(temp_bundle, report)
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.MISSING_PLAN_HASH.value
    
    def test_missing_config_hash(self, temp_bundle):
        """config_hash is required per Appendix C.2 Key Fields."""
        plan = {
            "schema_version": "1.0.0",
            "plan_hash": "abc",
            "as_of_time": "2026-01-01T00:00:00Z",
            # Missing config_hash
        }
        with open(temp_bundle / "plan.json", "w") as f:
            json.dump(plan, f)
        
        report = GateReport()
        result = validate_plan_identity(temp_bundle, report)
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.MISSING_REQUIRED_FIELD.value
        assert "config_hash" in report.gates[0]["evidence"]["missing_fields"]
    
    def test_hash_mismatch(self, temp_bundle, valid_plan):
        """HASH_MISMATCH when expected != actual (Appendix D.2)."""
        with open(temp_bundle / "plan.json", "w") as f:
            json.dump(valid_plan, f)
        
        report = GateReport()
        result = validate_plan_identity(
            temp_bundle, 
            report, 
            expected_plan_hash="wrong_hash"
        )
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.HASH_MISMATCH.value
        assert report.gates[0]["evidence"]["expected"] == "wrong_hash"
        assert report.gates[0]["evidence"]["actual"] == valid_plan["plan_hash"]
    
    def test_hash_match_evidence(self, temp_bundle, valid_plan):
        """When hash matches, evidence shows expected/actual pattern."""
        with open(temp_bundle / "plan.json", "w") as f:
            json.dump(valid_plan, f)
        
        report = GateReport()
        result = validate_plan_identity(
            temp_bundle,
            report,
            expected_plan_hash=valid_plan["plan_hash"]
        )
        
        assert result is True
        assert report.gates[0]["evidence"]["expected"] == valid_plan["plan_hash"]
        assert report.gates[0]["evidence"]["actual"] == valid_plan["plan_hash"]
    
    def test_malformed_json(self, temp_bundle):
        with open(temp_bundle / "plan.json", "w") as f:
            f.write("{bad json")
        
        report = GateReport()
        result = validate_plan_identity(temp_bundle, report)
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.MALFORMED_JSON.value
        assert report._has_invalid_input is True
    
    def test_unknown_schema_version(self, temp_bundle, valid_plan):
        valid_plan["schema_version"] = "99.0.0"
        with open(temp_bundle / "plan.json", "w") as f:
            json.dump(valid_plan, f)
        
        report = GateReport()
        result = validate_plan_identity(temp_bundle, report)
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.UNKNOWN_SCHEMA_VERSION.value
        assert report._has_invalid_input is True


# ============================================================================
# Unit Tests: Leakage Invariants (Known Leakage Patterns v5.1)
# ============================================================================

class TestValidateLeakageInvariants:
    def test_no_leakage(self, temp_bundle, valid_splits):
        with open(temp_bundle / "splits_manifest.json", "w") as f:
            json.dump(valid_splits, f)
        
        report = GateReport()
        result = validate_leakage_invariants(temp_bundle, report)
        
        assert result is True
    
    def test_temporal_overlap(self, temp_bundle, valid_splits):
        """train_end >= test_start should fail."""
        valid_splits["splits"] = [{
            "fold_id": 0,
            "train_start": "2020-01-01T00:00:00",
            "train_end": "2020-07-15T00:00:00",  # After test_start
            "test_start": "2020-07-01T00:00:00",
            "test_end": "2020-08-31T00:00:00",
        }]
        with open(temp_bundle / "splits_manifest.json", "w") as f:
            json.dump(valid_splits, f)
        
        report = GateReport()
        result = validate_leakage_invariants(temp_bundle, report)
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.LEAKAGE_DETECTED.value
    
    def test_purge_gap_enforced(self, temp_bundle, valid_splits):
        """Gap must be >= purge_window days."""
        valid_splits["purge_window"] = 30  # 30 days required
        valid_splits["splits"] = [{
            "fold_id": 0,
            "train_start": "2020-01-01T00:00:00",
            "train_end": "2020-06-25T00:00:00",  # Only 6 days before test
            "test_start": "2020-07-01T00:00:00",
            "test_end": "2020-08-31T00:00:00",
        }]
        with open(temp_bundle / "splits_manifest.json", "w") as f:
            json.dump(valid_splits, f)
        
        report = GateReport()
        result = validate_leakage_invariants(temp_bundle, report)
        
        assert result is False
        assert report.gates[0]["reason_code"] == ReasonCode.PURGE_VIOLATION.value


# ============================================================================
# Integration Tests: Full Bundle Validation
# ============================================================================

class TestValidateBundle:
    def test_valid_bundle_passes(self, complete_bundle):
        report, exit_code = validate_bundle(complete_bundle)
        
        assert exit_code == ExitCode.PASS
        assert report.overall_result == "PASS"

    def test_missing_knowledge_time_column_fails_governed_path(self, complete_bundle):
        payload = json.loads((complete_bundle / "dataset_manifest.json").read_text())
        payload.pop("knowledge_time_column")
        (complete_bundle / "dataset_manifest.json").write_text(json.dumps(payload))

        report, exit_code = validate_bundle(complete_bundle)

        lineage_gate = next(g for g in report.gates if g["gate_id"] == "data_lineage")
        assert exit_code == ExitCode.FAIL
        assert lineage_gate["result"] == "FAIL"
        assert lineage_gate["reason_code"] == ReasonCode.MISSING_KNOWLEDGE_TIME_COLUMN.value

    def test_non_pit_dataset_manifest_fails_governed_path(self, complete_bundle):
        payload = json.loads((complete_bundle / "dataset_manifest.json").read_text())
        payload["pit_compliant"] = False
        (complete_bundle / "dataset_manifest.json").write_text(json.dumps(payload))

        report, exit_code = validate_bundle(complete_bundle)

        lineage_gate = next(g for g in report.gates if g["gate_id"] == "data_lineage")
        assert exit_code == ExitCode.FAIL
        assert lineage_gate["reason_code"] == ReasonCode.PIT_NON_COMPLIANT.value

    def test_content_hash_mismatch_fails_when_expected_hash_present(self, complete_bundle):
        payload = json.loads((complete_bundle / "dataset_manifest.json").read_text())
        payload["content_hash"] = "sha256:actual"
        payload["content_hash_expected"] = "sha256:expected"
        (complete_bundle / "dataset_manifest.json").write_text(json.dumps(payload))

        report, exit_code = validate_bundle(complete_bundle)

        lineage_gate = next(g for g in report.gates if g["gate_id"] == "data_lineage")
        assert exit_code == ExitCode.FAIL
        assert lineage_gate["reason_code"] == ReasonCode.CONTENT_HASH_MISMATCH.value

    def test_stale_download_warns_but_bundle_still_passes(self, complete_bundle):
        payload = json.loads((complete_bundle / "dataset_manifest.json").read_text())
        payload["download_timestamp"] = "2020-01-01T00:00:00Z"
        (complete_bundle / "dataset_manifest.json").write_text(json.dumps(payload))

        report, exit_code = validate_bundle(complete_bundle)

        lineage_gate = next(g for g in report.gates if g["gate_id"] == "data_lineage")
        assert exit_code == ExitCode.PASS
        assert lineage_gate["result"] == "PASS"
        assert lineage_gate["reason_code"] == ReasonCode.STALE_DOWNLOAD_WARNING.value

    def test_missing_bundle_returns_exit_2(self, tmp_path):
        """Missing bundle -> exit 2 (invalid input)."""
        missing = tmp_path / "nonexistent"
        report, exit_code = validate_bundle(missing)
        
        assert exit_code == ExitCode.INVALID_INPUT
    
    def test_missing_bundle_writes_report_to_output(self, tmp_path):
        """Missing bundle still writes report when --output specified."""
        missing = tmp_path / "nonexistent"
        output = tmp_path / "result.json"
        
        report, exit_code = validate_bundle(missing, output_path=output)
        
        assert exit_code == ExitCode.INVALID_INPUT
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert data["overall_result"] == "FAIL"
        assert any(g["gate_id"] == "bundle_exists" for g in data["gates"])
    
    def test_missing_bundle_writes_report_to_cwd_fallback(self, tmp_path, monkeypatch):
        """Missing bundle writes to ./gate_result.json when no --output."""
        missing = tmp_path / "nonexistent"
        monkeypatch.chdir(tmp_path)
        
        report, exit_code = validate_bundle(missing)
        
        assert exit_code == ExitCode.INVALID_INPUT
        fallback = tmp_path / "gate_result.json"
        assert fallback.exists()
    
    def test_missing_required_file_returns_exit_1(self, complete_bundle):
        """Missing required file -> exit 1 (gate failure)."""
        (complete_bundle / "plan.json").unlink()
        report, exit_code = validate_bundle(complete_bundle)
        
        assert exit_code == ExitCode.FAIL
    
    def test_malformed_json_returns_exit_2(self, complete_bundle):
        """Malformed JSON -> exit 2 (invalid input)."""
        with open(complete_bundle / "plan.json", "w") as f:
            f.write("{invalid json")
        
        report, exit_code = validate_bundle(complete_bundle)
        
        assert exit_code == ExitCode.INVALID_INPUT
    
    def test_unknown_schema_returns_exit_2(self, complete_bundle):
        """Unknown schema version -> exit 2 (invalid input)."""
        with open(complete_bundle / "plan.json") as f:
            plan = json.load(f)
        plan["schema_version"] = "99.0.0"
        with open(complete_bundle / "plan.json", "w") as f:
            json.dump(plan, f)
        
        report, exit_code = validate_bundle(complete_bundle)
        
        assert exit_code == ExitCode.INVALID_INPUT
    
    def test_output_is_gate_result_json(self, complete_bundle):
        """Output file is gate_result.json per Appendix D.1."""
        report, _ = validate_bundle(complete_bundle)
        
        result_path = complete_bundle / "gate_result.json"
        assert result_path.exists()
        
        # And NOT gate_report.json
        assert not (complete_bundle / "gate_report.json").exists()


# ============================================================================
# CLI Tests (Appendix D.1 Contract)
# ============================================================================

class TestCLI:
    def test_check_command_positional(self, complete_bundle):
        """check with positional bundle works."""
        exit_code = main(["check", str(complete_bundle)])
        assert exit_code == ExitCode.PASS.value
    
    def test_check_command_flag(self, complete_bundle):
        """check --bundle works (per Appendix D.1)."""
        exit_code = main(["check", "--bundle", str(complete_bundle)])
        assert exit_code == ExitCode.PASS.value
    
    def test_validate_alias(self, complete_bundle):
        """validate alias works."""
        exit_code = main(["validate", str(complete_bundle)])
        assert exit_code == ExitCode.PASS.value
    
    def test_custom_output(self, complete_bundle, tmp_path):
        """--output flag works."""
        custom = tmp_path / "custom_result.json"
        exit_code = main(["check", str(complete_bundle), "--output", str(custom)])
        
        assert exit_code == ExitCode.PASS.value
        assert custom.exists()
    
    def test_json_flag(self, complete_bundle, capsys):
        """--json outputs to stdout."""
        exit_code = main(["check", str(complete_bundle), "--json"])
        
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == GATE_SCHEMA_VERSION


# ============================================================================
# Exit Code Contract Tests (Appendix D.1)
# ============================================================================

class TestExitCodeContract:
    """Verify exit codes match Appendix D.1 specification exactly."""
    
    def test_exit_0_all_pass(self, complete_bundle):
        """Exit 0 when all gates PASS."""
        _, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.PASS
        assert exit_code.value == 0
    
    def test_exit_1_gate_fail(self, complete_bundle):
        """Exit 1 when one or more gates FAIL."""
        (complete_bundle / "plan.json").unlink()
        _, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.FAIL
        assert exit_code.value == 1
    
    def test_exit_2_missing_bundle(self, tmp_path):
        """Exit 2 for missing bundle."""
        _, exit_code = validate_bundle(tmp_path / "nonexistent")
        assert exit_code == ExitCode.INVALID_INPUT
        assert exit_code.value == 2
    
    def test_exit_2_malformed_json(self, complete_bundle):
        """Exit 2 for malformed JSON."""
        with open(complete_bundle / "plan.json", "w") as f:
            f.write("not json")
        
        _, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        assert exit_code.value == 2
    
    def test_exit_2_unknown_schema(self, complete_bundle):
        """Exit 2 for unknown schema version."""
        with open(complete_bundle / "plan.json") as f:
            plan = json.load(f)
        plan["schema_version"] = "99.0.0"
        with open(complete_bundle / "plan.json", "w") as f:
            json.dump(plan, f)
        
        _, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        assert exit_code.value == 2


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_bundle(self, temp_bundle):
        """Empty bundle fails with proper gates."""
        report, exit_code = validate_bundle(temp_bundle)
        
        assert exit_code == ExitCode.FAIL
        assert len(report.gates) > 1  # bundle_exists + file checks
    
    def test_unicode_content(self, complete_bundle):
        """Unicode in files is handled."""
        with open(complete_bundle / "plan.json") as f:
            plan = json.load(f)
        plan["description"] = "日本語テスト"
        with open(complete_bundle / "plan.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False)
        
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.PASS


# ============================================================================
# Integration-ish: Canonical policy artifacts (stat_validity_report, execution_assumptions)
# ============================================================================

class TestCanonicalArtifactStatValidityReport:
    """Bundle contains stat_validity_report.json: canonical policy interpretation only."""

    def test_stat_validity_report_pass_overall_pass(self, complete_bundle):
        """Bundle with valid stat_validity_report.json gate_result=PASS → gate overall PASS."""
        with open(complete_bundle / "stat_validity_report.json", "w") as f:
            json.dump(_minimal_stat_validity_report("PASS"), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.PASS
        assert report.overall_result == "PASS"
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["result"] == "PASS"

    def test_stat_validity_report_fail_overall_fail_exit_1(self, complete_bundle):
        """Bundle with stat_validity_report.json gate_result=FAIL → gate overall FAIL, exit 1."""
        with open(complete_bundle / "stat_validity_report.json", "w") as f:
            json.dump(_minimal_stat_validity_report("FAIL"), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.FAIL
        assert report.overall_result == "FAIL"
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["reason_code"] == ReasonCode.STAT_VALIDITY_GATE_FAIL.value

    @pytest.mark.determinism("d1")
    def test_stat_validity_report_missing_fails_exit_1(self, complete_bundle):
        """Missing stat_validity_report.json must fail the canonical governed path."""
        (complete_bundle / "stat_validity_report.json").unlink()
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.FAIL
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["reason_code"] == ReasonCode.MISSING_FILE.value
        assert stat_gates[0]["result"] == "FAIL"

    def test_stat_validity_report_malformed_exit_2(self, complete_bundle):
        """Bundle with malformed stat_validity_report.json → exit 2 (invalid input)."""
        with open(complete_bundle / "stat_validity_report.json", "w") as f:
            f.write("{ not valid json ")
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        assert report._has_invalid_input
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["reason_code"] == ReasonCode.STAT_VALIDITY_INVALID_STRUCTURE.value

    @pytest.mark.determinism("d1")
    def test_stat_validity_report_malformed_pbo_exit_2(self, complete_bundle):
        """Malformed pbo must be rejected as invalid structure without recomputing stats."""
        with open(complete_bundle / "stat_validity_report.json", "w") as f:
            json.dump(_minimal_stat_validity_report("PASS", pbo_value=1.25), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["reason_code"] == ReasonCode.STAT_VALIDITY_INVALID_STRUCTURE.value
        assert stat_gates[0]["result"] == "FAIL"
    @pytest.mark.determinism("d1")
    def test_stat_validity_report_missing_pbo_field_exit_2(self, complete_bundle):
        """Missing top-level pbo must fail the canonical report contract."""
        payload = _minimal_stat_validity_report("PASS")
        del payload["pbo"]
        with open(complete_bundle / "stat_validity_report.json", "w") as f:
            json.dump(payload, f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["reason_code"] == ReasonCode.STAT_VALIDITY_INVALID_STRUCTURE.value
        assert stat_gates[0]["result"] == "FAIL"

    @pytest.mark.determinism("d1")
    def test_stat_validity_report_invalid_nested_pbo_gate_result_exit_2(self, complete_bundle):
        """Nested pbo.gate_result must stay within PASS/WARN/FAIL when present."""
        with open(complete_bundle / "stat_validity_report.json", "w") as f:
            json.dump(_minimal_stat_validity_report("PASS", pbo_value=0.22, pbo_gate_result="MAYBE"), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        stat_gates = [g for g in report.gates if g["gate_id"] == "stat_validity_report"]
        assert len(stat_gates) == 1
        assert stat_gates[0]["reason_code"] == ReasonCode.STAT_VALIDITY_INVALID_STRUCTURE.value
        assert stat_gates[0]["result"] == "FAIL"


class TestCanonicalArtifactExecutionAssumptions:
    """Bundle contains or lacks execution_assumptions.json under strict canonical policy."""

    @pytest.mark.determinism("d1")
    def test_execution_assumptions_missing_fails(self, complete_bundle):
        """Missing execution_assumptions.json must fail the canonical governed path."""
        (complete_bundle / "execution_assumptions.json").unlink()
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.FAIL
        assert report.overall_result == "FAIL"
        exec_gates = [g for g in report.gates if g["gate_id"] == "execution_assumptions"]
        assert len(exec_gates) == 1
        assert exec_gates[0]["reason_code"] == ReasonCode.COST_ASSUMPTION_MISSING.value
        assert exec_gates[0]["result"] == "FAIL"

    def test_execution_assumptions_malformed_exit_2(self, complete_bundle):
        """Malformed execution_assumptions.json → exit 2."""
        with open(complete_bundle / "execution_assumptions.json", "w") as f:
            f.write("not json")
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.INVALID_INPUT
        exec_gates = [g for g in report.gates if g["gate_id"] == "execution_assumptions"]
        assert len(exec_gates) == 1
        assert exec_gates[0]["reason_code"] == ReasonCode.EXECUTION_ASSUMPTIONS_INVALID_STRUCTURE.value

    @pytest.mark.determinism("d1")
    def test_execution_assumptions_explicit_zero_fails(self, complete_bundle):
        """Explicit zero cost must fail Phase I-E policy enforcement."""
        with open(complete_bundle / "execution_assumptions.json", "w") as f:
            json.dump(_minimal_execution_assumptions(commission_bps=0, cost_model_id="fees.zero"), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.FAIL
        exec_gates = [g for g in report.gates if g["gate_id"] == "execution_assumptions"]
        assert len(exec_gates) == 1
        assert exec_gates[0]["reason_code"] == ReasonCode.ZERO_COST_ASSUMED.value
        assert exec_gates[0]["result"] == "FAIL"

    @pytest.mark.determinism("d1")
    def test_execution_assumptions_explicit_nonzero_pass(self, complete_bundle):
        """Explicit non-zero cost → PASS, VALID reason."""
        with open(complete_bundle / "execution_assumptions.json", "w") as f:
            json.dump(_minimal_execution_assumptions(commission_bps=5, cost_model_id="fixed_5bps"), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.PASS
        exec_gates = [g for g in report.gates if g["gate_id"] == "execution_assumptions"]
        assert len(exec_gates) == 1
        assert exec_gates[0]["reason_code"] == ReasonCode.VALID.value
        assert exec_gates[0]["result"] == "PASS"

    def test_execution_assumptions_cost_model_id_only_pass(self, complete_bundle):
        """A non-zero cost model id without numeric fields remains acceptable if not declared zero."""
        with open(complete_bundle / "execution_assumptions.json", "w") as f:
            json.dump(_minimal_execution_assumptions(commission_bps=None, cost_model_id="fixed_2bps"), f)
        report, exit_code = validate_bundle(complete_bundle)
        assert exit_code == ExitCode.PASS
        exec_gates = [g for g in report.gates if g["gate_id"] == "execution_assumptions"]
        assert len(exec_gates) == 1
        assert exec_gates[0]["reason_code"] == ReasonCode.VALID.value




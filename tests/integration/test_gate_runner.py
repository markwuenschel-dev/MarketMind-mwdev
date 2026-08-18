"""Integration tests for the gate runner CLI.

Per spec §15.2, tests cover:
- valid_bundle: PASS in promote mode
- hash_mismatch: FAIL with HASH_MISMATCH
- threshold_violation: FAIL with THRESHOLD_VIOLATION
- unknown_field: FAIL with SCHEMA_VIOLATION
- missing_lock: FAIL with INCOMPARABLE
- run_ids_only: PASS with --allow-run-ids, FAIL in promote mode
- missing_required_artifact: FAIL with MISSING_REQUIRED_ARTIFACT
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "bundles"
SCHEMAS_DIR = REPO_ROOT / "schemas"
POLICY_PATH = REPO_ROOT / "policies" / "gating_policy.v1.yaml"


def run_gate(bundle: str, mode: str, allow_run_ids: bool = False) -> tuple[int, dict, dict]:
    """Run the gate runner and return (exit_code, gate_decision, promotion_event).

    Copies the fixture bundle to a temporary directory so that parallel
    test workers don't race on the shared out/gate_decision.json file.
    """
    src_dir = FIXTURES_DIR / bundle

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / bundle
        # Copy fixture files (ignore existing out/ with stale results)
        shutil.copytree(src_dir, bundle_dir, ignore=shutil.ignore_patterns("out"))

        cmd = [
            sys.executable,
            "-m",
            "marketmind_gate.cli",
            mode,
            str(bundle_dir),
            "--schemas",
            str(SCHEMAS_DIR),
            "--policy",
            str(POLICY_PATH),
        ]
        if allow_run_ids and mode == "validate":
            cmd.append("--allow-run-ids")

        result = subprocess.run(cmd, capture_output=True, cwd=str(REPO_ROOT))

        decision_path = bundle_dir / "out" / "gate_decision.json"
        decision = {}
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())

        # Also capture promotion_event if present
        event_path = bundle_dir / "out" / "promotion_event.json"
        event = {}
        if event_path.exists():
            event = json.loads(event_path.read_text())

    return result.returncode, decision, event


class TestValidBundle:
    """Tests for valid_bundle fixture."""

    def test_promote_passes(self) -> None:
        """Valid bundle should PASS in promote mode."""
        exit_code, decision, _ = run_gate("valid_bundle", "promote")

        assert exit_code == 0
        assert decision["result"] == "PASS"
        assert len(decision["reasons"]) == 0

    def test_validate_passes(self) -> None:
        """Valid bundle should also pass in validate mode."""
        exit_code, decision, _ = run_gate("valid_bundle", "validate")

        assert exit_code == 0
        assert decision["result"] == "PASS"

    def test_promotion_event_created(self) -> None:
        """Promotion event should be created on PASS."""
        _, _, event = run_gate("valid_bundle", "promote")

        assert event, "promotion_event.json should be created on PASS"
        assert event["schema_version"] == "v1"
        assert "gate_policy_hash" in event
        assert "gate_decision_hash" in event
        assert "intent_hash" in event
        assert "plan_hash" in event


class TestHashMismatch:
    """Tests for hash_mismatch fixture."""

    def test_validate_fails_with_hash_mismatch(self) -> None:
        """Hash mismatch should cause FAIL."""
        exit_code, decision, _ = run_gate("hash_mismatch", "validate")

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "HASH_MISMATCH" in error_codes

    def test_error_includes_hash_details(self) -> None:
        """Hash mismatch error should include expected and actual hashes."""
        _, decision, _ = run_gate("hash_mismatch", "validate")

        hash_errors = [r for r in decision["reasons"] if r["code"] == "HASH_MISMATCH"]
        assert len(hash_errors) > 0

        error = hash_errors[0]
        assert "context" in error
        assert "declared_hash" in error["context"]
        assert "computed_hash" in error["context"]


class TestThresholdViolation:
    """Tests for threshold_violation fixture."""

    def test_promote_fails_with_threshold_violation(self) -> None:
        """Threshold violation should cause FAIL in promote mode."""
        exit_code, decision, _ = run_gate("threshold_violation", "promote")

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "THRESHOLD_VIOLATION" in error_codes

    def test_spearman_rho_below_threshold(self) -> None:
        """Should fail when spearman_rho < 0.70."""
        _, decision, _ = run_gate("threshold_violation", "promote")

        spearman_errors = [
            r for r in decision["reasons"]
            if r["code"] == "THRESHOLD_VIOLATION" and "spearman_rho" in r.get("path", "")
        ]
        assert len(spearman_errors) > 0

    def test_top_k_overlap_below_threshold(self) -> None:
        """Should fail when top_k_overlap < 0.60."""
        _, decision, _ = run_gate("threshold_violation", "promote")

        overlap_errors = [
            r for r in decision["reasons"]
            if r["code"] == "THRESHOLD_VIOLATION" and "top_" in r.get("path", "")
        ]
        assert len(overlap_errors) > 0

    def test_no_promotion_event_on_fail(self) -> None:
        """Promotion event should NOT be created on FAIL."""
        _, _, event = run_gate("threshold_violation", "promote")

        assert not event, "promotion_event.json should NOT be created on FAIL"


class TestUnknownField:
    """Tests for unknown_field fixture."""

    def test_validate_fails_with_schema_violation(self) -> None:
        """Unknown field should cause SCHEMA_VIOLATION."""
        exit_code, decision, _ = run_gate("unknown_field", "validate")

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "SCHEMA_VIOLATION" in error_codes

    def test_error_identifies_unknown_property(self) -> None:
        """Error should identify the unknown property."""
        _, decision, _ = run_gate("unknown_field", "validate")

        schema_errors = [r for r in decision["reasons"] if r["code"] == "SCHEMA_VIOLATION"]
        assert len(schema_errors) > 0


class TestMissingLock:
    """Tests for missing_lock fixture (fidelity mismatch between lanes)."""

    def test_promote_fails_with_incomparable(self) -> None:
        """Mismatched fidelity_id should cause INCOMPARABLE."""
        exit_code, decision, _ = run_gate("missing_lock", "promote")

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "INCOMPARABLE" in error_codes

    def test_error_identifies_lock_mismatch(self) -> None:
        """Error should identify fidelity_id mismatch."""
        _, decision, _ = run_gate("missing_lock", "promote")

        incomparable_errors = [r for r in decision["reasons"] if r["code"] == "INCOMPARABLE"]
        assert len(incomparable_errors) > 0

        error = incomparable_errors[0]
        assert error["context"]["lock"] == "fidelity_id"
        assert error["context"]["lane_a_value"] == "high"
        assert error["context"]["lane_b_value"] == "low"


class TestRunIdsOnly:
    """Tests for run_ids_only fixture."""

    def test_validate_passes_with_allow_run_ids(self) -> None:
        """Validate with --allow-run-ids should PASS."""
        exit_code, decision, _ = run_gate("run_ids_only", "validate", allow_run_ids=True)

        assert exit_code == 0
        assert decision["result"] == "PASS"

    def test_validate_fails_without_allow_run_ids(self) -> None:
        """Validate without --allow-run-ids should FAIL."""
        exit_code, decision, _ = run_gate("run_ids_only", "validate", allow_run_ids=False)

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "BINDING_INVALID" in error_codes

    def test_promote_fails_with_binding_invalid(self) -> None:
        """Promote mode should FAIL with BINDING_INVALID."""
        exit_code, decision, _ = run_gate("run_ids_only", "promote")

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "BINDING_INVALID" in error_codes


class TestMissingRequiredArtifact:
    """Tests for missing_required_artifact fixture."""

    def test_promote_fails_with_missing_artifact(self) -> None:
        """Promote without transfer_report should FAIL."""
        exit_code, decision, _ = run_gate("missing_required_artifact", "promote")

        assert exit_code == 2
        assert decision["result"] == "FAIL"

        error_codes = [r["code"] for r in decision["reasons"]]
        assert "MISSING_REQUIRED_ARTIFACT" in error_codes

    def test_validate_passes_without_transfer_report(self) -> None:
        """Validate mode doesn't require transfer_report."""
        exit_code, decision, _ = run_gate("missing_required_artifact", "validate")

        assert exit_code == 0
        assert decision["result"] == "PASS"


class TestDeterminism:
    """Tests for hash determinism."""

    def test_gate_decision_hash_is_deterministic(self) -> None:
        """Same inputs should produce same gate_decision_hash."""
        _, decision1, _ = run_gate("valid_bundle", "promote")
        _, decision2, _ = run_gate("valid_bundle", "promote")

        assert decision1["gate_decision_hash"] == decision2["gate_decision_hash"]

    def test_gate_policy_hash_is_consistent(self) -> None:
        """gate_policy_hash should be same across runs."""
        _, decision1, _ = run_gate("valid_bundle", "promote")
        _, decision2, _ = run_gate("threshold_violation", "promote")

        assert decision1["gate_policy_hash"] == decision2["gate_policy_hash"]



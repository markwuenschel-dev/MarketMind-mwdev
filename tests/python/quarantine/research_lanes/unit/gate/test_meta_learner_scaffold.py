"""Tests for Phase II meta-learner scaffold gate shell."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marketmind_gate.gates.meta_learner_scaffold import (
    REASON_GOVERNED_CONTRACT_MISMATCH,
    REASON_GOVERNED_SCHEMA_INVALID,
    REASON_META_VALIDITY_OVERALL_REDACTED,
    REASON_META_VALIDITY_OVERALL_SCAFFOLD,
    MetaLearnerScaffoldStatus,
    evaluate_meta_learner_scaffold,
)
from pysrc.meta.execution_assumptions_emitter import emit_execution_assumptions
from pysrc.meta.meta_validity_emitter import emit_meta_validity_report
from pysrc.meta.task_manifest_emitter import TaskManifestTaskInput, emit_task_manifest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.determinism("d1")
def test_scaffold_gate_never_pass_fail(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = TaskManifestTaskInput(
        regime_id="r",
        regime_class="bull",
        t0="t0",
        t1="t1",
        signal_ids_hash="h",
        signal_set_version="v",
        support_last_timestamp="ts",
    )
    emit_task_manifest(tmp_path / "task_manifest.json", tasks=[task], seed=1)
    emit_meta_validity_report(tmp_path / "meta_validity_report.json", seed=1)
    emit_execution_assumptions(tmp_path / "execution_assumptions.json", seed=1)
    res = evaluate_meta_learner_scaffold(tmp_path)
    assert res.status == MetaLearnerScaffoldStatus.SCAFFOLD_INCOMPLETE
    payload = res.to_json_dict()
    assert payload["status"] == "SCAFFOLD_INCOMPLETE"
    assert payload["promotable_claim_emitted"] is False
    assert payload["artifacts_present"]["task_manifest.json"] is True
    assert payload["incomplete_reason_codes"] == [
        REASON_GOVERNED_CONTRACT_MISMATCH,
        REASON_GOVERNED_SCHEMA_INVALID,
        REASON_META_VALIDITY_OVERALL_SCAFFOLD,
    ]
    roles = [s["role"] for s in payload["artifact_surfaces"]]
    assert roles == ["task_manifest", "meta_validity_report", "execution_assumptions"]
    dumped = json.dumps(payload, sort_keys=True)
    assert "PASS" not in dumped
    assert "FAIL" not in dumped
    assert payload["meta_validity_report"]["overall_result_shell_visible"] == "SCAFFOLD_INCOMPLETE"
    assert payload["meta_validity_report"]["overall_result_non_scaffold_redacted"] is False


@pytest.mark.determinism("d1")
def test_scaffold_gate_missing_artifact(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    res = evaluate_meta_learner_scaffold(tmp_path)
    assert res.status == MetaLearnerScaffoldStatus.SCAFFOLD_INCOMPLETE
    assert res.artifacts_present["task_manifest.json"] is False
    assert "MISSING_ARTIFACTS" in res.incomplete_reason_codes


@pytest.mark.determinism("d1")
def test_scaffold_gate_bad_schema_prefix(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    (tmp_path / "task_manifest.json").write_text(
        json.dumps({"schema_version": "wrong"}),
        encoding="utf-8",
    )
    (tmp_path / "meta_validity_report.json").write_text(
        json.dumps(
            {
                "schema_version": "meta_validity_report.scaffold.v1",
                "overall_result": "SCAFFOLD_INCOMPLETE",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "execution_assumptions.json").write_text(
        json.dumps({"schema_version": "execution_assumptions.scaffold.v1"}),
        encoding="utf-8",
    )
    res = evaluate_meta_learner_scaffold(tmp_path)
    assert res.schema_ok["task_manifest.json"] is False
    assert "SCHEMA_PREFIX_MISMATCH" in res.incomplete_reason_codes


@pytest.mark.determinism("d1")
def test_scaffold_gate_json_never_echoes_promotable_overall_result(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    (tmp_path / "task_manifest.json").write_text(
        json.dumps({"schema_version": "rg09.task_manifest.scaffold.v1"}),
        encoding="utf-8",
    )
    (tmp_path / "meta_validity_report.json").write_text(
        json.dumps(
            {
                "schema_version": "meta_validity_report.scaffold.v1",
                "overall_result": "PASS",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "execution_assumptions.json").write_text(
        json.dumps({"schema_version": "execution_assumptions.scaffold.v1"}),
        encoding="utf-8",
    )
    res = evaluate_meta_learner_scaffold(tmp_path)
    assert res.overall_result == "PASS"
    payload = res.to_json_dict()
    assert REASON_META_VALIDITY_OVERALL_REDACTED in res.incomplete_reason_codes
    assert payload["meta_validity_report"]["overall_result_shell_visible"] is None
    assert payload["meta_validity_report"]["overall_result_non_scaffold_redacted"] is True
    dumped = json.dumps(payload, sort_keys=True)
    assert "PASS" not in dumped

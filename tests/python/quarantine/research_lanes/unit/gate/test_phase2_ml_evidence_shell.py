"""Phase II-0B ML evidence shell tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marketmind_gate.gates.phase2_ml_evidence_shell import (
    Phase2MLEvidenceShellStatus,
    evaluate_phase2_ml_evidence_shell,
)
from pysrc.meta.threshold_catalog import THR_RG09_V03

pytest.importorskip("pysrc.meta.phase2_artifact_contract", reason="II-0C lane archived")
from pysrc.meta.phase2_artifact_contract import (  # type: ignore[import-not-found]
    PhaseIIRunContext,
    canonical_artifact_content_hash,
    emit_phase2_artifacts,
)
from pysrc.meta.task_manifest_emitter import TaskManifestTaskInput  # type: ignore[import-not-found]
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.task_generator import derive_signal_ids_hash


def _task() -> TaskManifestTaskInput:
    ids, mask = build_fixed_slot_surface_from_sparse_slots({0: "phase2.shell.test.sig"})
    return TaskManifestTaskInput(
        regime_id="r",
        regime_class="bull",
        t0="2020-01-01T00:00:00+00:00",
        t1="2020-03-01T00:00:00+00:00",
        signal_ids_hash=derive_signal_ids_hash(signal_ids=ids, signal_mask=mask),
        signal_set_version="rg09.v1",
        support_last_timestamp="2020-02-15T00:00:00+00:00",
        signal_ids=ids,
        signal_mask=mask,
        active_k=sum(1 for item in mask if item),
    )


def _ctx(tmp_path: Path) -> PhaseIIRunContext:
    return PhaseIIRunContext(
        output_dir=tmp_path,
        seed=1,
        timestamp_utc="2026-04-06T12:00:00Z",
        tasks=[_task()],
        dataset_manifest={
            "pit_compliant": True,
            "knowledge_time_column": "knowledge_time",
            "content_hash": "a",
            "content_hash_expected": "a",
        },
        inner_loop_gain_by_regime={},
        harvey_t_statistic=1.0,
        encoder_coherence_score=1.0,
        crisis_episode_ic=1.0,
        forgetting_metric=1.0,
        plasticity_metric=1.0,
        baseline_comparison={
            "baseline_kind": "xgboost_incumbent",
            "baseline_run_id": "xgboost-b",
            "challenger_run_id": "c",
            "splits_fingerprint": "s",
            "data_fingerprint": "d",
            "cost_assumptions_fingerprint": "x",
            "data_parity": True,
            "split_parity": True,
            "cost_parity": True,
            "net_result_against_incumbent": "not_evaluated_non_promotable",
        },
        cost_model={"commission_bps": 1.0},
        slippage_model={"model": "fixed_bps"},
        borrow_funding={"rate_bps": 0.0},
        latency_fill={"latency_ms": 5},
        shared_comparison_context={
            "splits_fingerprint": "s",
            "data_fingerprint": "d",
            "cost_assumptions_fingerprint": "x",
        },
        threshold_references=(
            {
                "threshold_id": THR_RG09_V03,
                "consumer": "phase2.shell.test.harvey_t_statistic",
                "gate_critical": False,
                "usage_role": "meta_validity_report.harvey_t_statistic",
            },
        ),
    )


def _rewrite_artifact(path: Path, payload: dict[str, object], *, rehash: bool = True) -> None:
    if rehash:
        payload["content_hash"] = {
            "algorithm": "sha256",
            "canonicalization": "json.sort_keys.no_ws.omit_content_hash.v1",
            "value": canonical_artifact_content_hash(payload),
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_accepts_structurally_usable_triple(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_STRUCTURALLY_USABLE
    assert result.promotable_claim_emitted is False
    assert result.reason_codes == ()
    threshold_summary = result.evidence["threshold_governance"]
    assert threshold_summary["all_consumers_lineage_bound"] is True
    assert threshold_summary["has_provisional_thresholds"] is False
    assert threshold_summary["references"][0]["threshold_id"] == "THR-RG09-V03"


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_promotable_claim(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["promotable_claim_emitted"] = True
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "PROMOTABLE_CLAIM_EMITTED" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_tampered_content_hash(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "task_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content_hash"]["value"] = "sha256:" + ("0" * 64)
    _rewrite_artifact(path, payload, rehash=False)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_pit_mismatch(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pit_boundary"] = "2020-02-16T00:00:00+00:00"
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_signal_hash_mismatch(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signal_ids_hash"] = "sha256:" + ("1" * 64)
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_baseline_shared_context_mismatch(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "execution_assumptions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shared_comparison_context"]["data_fingerprint"] = "different-data"
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_rg09_anchor_as_incumbent(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_comparison"]["baseline_run_id"] = "rg09-strict-h3-reference-anchor"
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_gate_critical_provisional_threshold(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold_references"] = [
        {
            "threshold_id": "THR-RG09-V17",
            "state": "PROVISIONAL",
            "consumer": "phase2.shell.test",
            "gate_critical": True,
            "current_expression": "recovery_ratio >= 0.0",
            "usage_role": "test",
            "used_as_pass_fail_criterion": True,
        }
    ]
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_unknown_threshold(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold_references"] = [
        {
            "threshold_id": "THR-DOES-NOT-EXIST",
            "state": "VALIDATED",
            "consumer": "phase2.shell.test",
            "gate_critical": False,
            "current_expression": "x > 0",
            "usage_role": "test",
            "used_as_pass_fail_criterion": False,
        }
    ]
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_deprecated_threshold(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold_references"] = [
        {
            "threshold_id": "THR-RG09-V20",
            "state": "VALIDATED",
            "consumer": "phase2.shell.test",
            "gate_critical": False,
            "current_expression": "recovery_ratio >= 0.0",
            "usage_role": "test",
            "used_as_pass_fail_criterion": False,
        }
    ]
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_rejected_threshold(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.threshold_governance as tg

    payload = json.loads(tg.default_register_path().read_text(encoding="utf-8"))
    payload["records"].append(
        {
            "threshold_id": "THR-SHELL-REJECTED",
            "name": "Rejected shell threshold",
            "governing_surface": "test",
            "consumer_surface": "test",
            "state": "REJECTED",
            "current_expression": "x > 0",
            "evidence_required": "none",
            "evidence_location": "none",
            "authority": "test",
            "gate_critical": False,
            "supersedes": None,
            "superseded_by": None,
            "last_reviewed": "2026-04-15",
        }
    )
    register_path = tmp_path / "threshold_register.json"
    register_path.write_text(json.dumps(payload), encoding="utf-8")
    tg.clear_threshold_register_cache()
    monkeypatch.setattr(tg, "default_register_path", lambda: register_path)
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["threshold_references"] = [
        {
            "threshold_id": "THR-SHELL-REJECTED",
            "state": "VALIDATED",
            "consumer": "phase2.shell.test",
            "gate_critical": False,
            "current_expression": "x > 0",
            "usage_role": "test",
            "used_as_pass_fail_criterion": False,
        }
    ]
    _rewrite_artifact(path, meta)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes
    tg.clear_threshold_register_cache()


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_threshold_state_tamper(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold_references"] = [
        {
            "threshold_id": "THR-RG09-V17",
            "state": "VALIDATED",
            "consumer": "phase2.shell.test",
            "gate_critical": False,
            "current_expression": "min_episode_regime_class_purity = 0.70 when boundary_recovery active",
            "usage_role": "test",
            "used_as_pass_fail_criterion": False,
        }
    ]
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_threshold_expression_tamper(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold_references"] = [
        {
            "threshold_id": "THR-RG09-V17",
            "state": "PROVISIONAL",
            "consumer": "phase2.shell.test",
            "gate_critical": False,
            "current_expression": "tampered_expression >= 1.0",
            "usage_role": "test",
            "used_as_pass_fail_criterion": False,
        }
    ]
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_allows_non_gate_provisional_threshold(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "meta_validity_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold_references"] = [
        {
            "threshold_id": "THR-RG09-V17",
            "state": "PROVISIONAL",
            "consumer": "phase2.shell.test",
            "gate_critical": False,
            "current_expression": "min_episode_regime_class_purity = 0.70 when boundary_recovery active",
            "usage_role": "test",
            "used_as_pass_fail_criterion": False,
        }
    ]
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_STRUCTURALLY_USABLE
    threshold_summary = result.evidence["threshold_governance"]
    assert threshold_summary["has_provisional_thresholds"] is True
    assert threshold_summary["references"][0]["state"] == "PROVISIONAL"


@pytest.mark.determinism("d1")
def test_phase2_ml_evidence_shell_rejects_seed_lineage_mismatch(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    emit_phase2_artifacts(_ctx(tmp_path))
    path = tmp_path / "execution_assumptions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed_lineage"]["derived_seeds"][0]["uint64_seed"] += 1
    _rewrite_artifact(path, payload)

    result = evaluate_phase2_ml_evidence_shell(tmp_path)
    assert result.status == Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE
    assert "SEMANTIC_INVARIANT_INVALID" in result.reason_codes

"""MLN-05 frozen inference boundary contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.phase2_artifact_contract import (
    PhaseIIArtifactError,
    PhaseIIRunContext,
    emit_phase2_artifacts,
)
from pysrc.meta.task_manifest_emitter import TaskManifestTaskInput
from pysrc.meta_learning.confidence_contract import synthetic_confidence_calibration_pass_block
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.inference_boundary import (
    CONTRACT_VERSION,
    ExecutionPath,
    ParameterRole,
    RolloutStage,
    ThetaDayPrimeCheckpointRef,
    assert_no_live_gradients,
    build_inference_boundary_audit_block,
    ensure_training_only_task_prime,
    promote_theta_day_prime,
    rollback_theta_day_prime,
    rollout_stage_assumes_frozen_live_checkpoint,
    validate_frozen_inference_request,
    validate_inference_boundary_audit_block,
)
from pysrc.meta_learning.task_generator import derive_signal_ids_hash


def _task_row() -> TaskManifestTaskInput:
    ids, mask = build_fixed_slot_surface_from_sparse_slots({0: "mln05.test.sig"})
    h = derive_signal_ids_hash(signal_ids=ids, signal_mask=mask)
    ak = sum(1 for m in mask if m)
    return TaskManifestTaskInput(
        regime_id="r",
        regime_class="bull",
        t0="2020-01-01T00:00:00+00:00",
        t1="2020-03-01T00:00:00+00:00",
        signal_ids_hash=h,
        signal_set_version="v1",
        support_last_timestamp="2020-02-15T00:00:00+00:00",
        signal_ids=ids,
        signal_mask=mask,
        active_k=ak,
    )


def _minimal_ctx(tmp_path: Path, *, audit: dict[str, object] | None = None) -> PhaseIIRunContext:
    cm, sm, bf, lf = (
        {"commission_bps": 1.0},
        {"model": "fixed_bps"},
        {"rate_bps": 0.0},
        {"latency_ms": 5},
    )
    return PhaseIIRunContext(
        output_dir=tmp_path,
        seed=1,
        timestamp_utc="2026-04-07T12:00:00Z",
        tasks=[_task_row()],
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
        cost_model=cm,
        slippage_model=sm,
        borrow_funding=bf,
        latency_fill=lf,
        shared_comparison_context={
            "splits_fingerprint": "s",
            "data_fingerprint": "d",
            "cost_assumptions_fingerprint": "x",
        },
        confidence_calibration=synthetic_confidence_calibration_pass_block(ece_value=0.05),
        inference_boundary_audit=audit,
    )


@pytest.mark.determinism("d0")
def test_live_path_rejects_gradients(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError):
        validate_frozen_inference_request(
            execution_path=ExecutionPath.LIVE_INFERENCE,
            checkpoint_role=ParameterRole.THETA_DAY_PRIME,
            allows_gradients=True,
        )
    assert_no_live_gradients(execution_path=ExecutionPath.TRAINING, allows_gradients=True)


@pytest.mark.determinism("d0")
def test_live_inference_rejects_theta_meta(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError):
        validate_frozen_inference_request(
            execution_path=ExecutionPath.LIVE_INFERENCE,
            checkpoint_role=ParameterRole.THETA_META,
            allows_gradients=False,
        )


@pytest.mark.determinism("d0")
def test_task_prime_forbidden_on_live(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError):
        ensure_training_only_task_prime(
            checkpoint_role=ParameterRole.THETA_TASK_PRIME,
            execution_path=ExecutionPath.LIVE_INFERENCE,
        )


@pytest.mark.determinism("d0")
def test_promotion_requires_gate_and_training(deterministic_seed: int) -> None:
    _ = deterministic_seed
    live = ThetaDayPrimeCheckpointRef(checkpoint_id="live-a")
    cand = ThetaDayPrimeCheckpointRef(checkpoint_id="cand-b")
    same, none_ = promote_theta_day_prime(
        current_live=live,
        candidate=cand,
        gate_passed=False,
        nightly_training_succeeded=True,
    )
    assert same == live
    assert none_ is None
    same2, none2 = promote_theta_day_prime(
        current_live=live,
        candidate=cand,
        gate_passed=True,
        nightly_training_succeeded=False,
    )
    assert same2 == live
    assert none2 is None
    new_live, rb = promote_theta_day_prime(
        current_live=live,
        candidate=cand,
        gate_passed=True,
        nightly_training_succeeded=True,
    )
    assert new_live == cand
    assert rb == live


@pytest.mark.determinism("d0")
def test_rollback_selects_prior_promoted(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cur = ThetaDayPrimeCheckpointRef(checkpoint_id="live-now")
    prev = ThetaDayPrimeCheckpointRef(checkpoint_id="live-prior")
    assert rollback_theta_day_prime(current_live=cur, rollback_target=prev) == prev


@pytest.mark.determinism("d0")
def test_rollout_stages_assume_frozen(deterministic_seed: int) -> None:
    _ = deterministic_seed
    assert rollout_stage_assumes_frozen_live_checkpoint(RolloutStage.SHADOW) is True


@pytest.mark.determinism("d1")
def test_emit_phase2_accepts_valid_inference_boundary_audit(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    audit = build_inference_boundary_audit_block(
        previous_live_theta_day_prime_ref="cas.v1:b3-256:pre",
        live_theta_day_prime_ref="cas.v1:b3-256:promoted",
        rollback_theta_day_prime_ref="cas.v1:b3-256:pre",
        theta_meta_ref="cas.v1:b3-256:meta",
        training_outcome="success",
        rollout_stage=RolloutStage.SHADOW,
    )
    validate_inference_boundary_audit_block(audit)
    emit_phase2_artifacts(_minimal_ctx(tmp_path, audit=audit))
    mv = json.loads((tmp_path / "meta_validity_report.json").read_text(encoding="utf-8"))
    assert mv["inference_boundary"]["schema_version"] == CONTRACT_VERSION
    assert mv["inference_boundary"]["live_theta_day_prime_ref"] == "cas.v1:b3-256:promoted"
    assert mv["inference_boundary"]["previous_live_theta_day_prime_ref"] == "cas.v1:b3-256:pre"


@pytest.mark.determinism("d0")
def test_audit_failed_training_requires_unchanged_live(deterministic_seed: int) -> None:
    _ = deterministic_seed
    bad = build_inference_boundary_audit_block(
        previous_live_theta_day_prime_ref="cas.v1:unchanged",
        live_theta_day_prime_ref="cas.v1:wrong_new",
        rollback_theta_day_prime_ref="cas.v1:rollback",
        theta_meta_ref=None,
        training_outcome="failed",
    )
    with pytest.raises(DataPreconditionError, match="failed or skipped"):
        validate_inference_boundary_audit_block(bad)


@pytest.mark.determinism("d0")
def test_audit_skipped_requires_unchanged_live(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ok = build_inference_boundary_audit_block(
        previous_live_theta_day_prime_ref="cas.v1:same",
        live_theta_day_prime_ref="cas.v1:same",
        rollback_theta_day_prime_ref="cas.v1:rollback",
        theta_meta_ref=None,
        training_outcome="skipped",
    )
    validate_inference_boundary_audit_block(ok)


@pytest.mark.determinism("d1")
def test_emit_phase2_rejects_bad_inference_boundary_audit(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    bad = {"schema_version": "wrong"}
    with pytest.raises(PhaseIIArtifactError, match="MLN-05"):
        emit_phase2_artifacts(_minimal_ctx(tmp_path, audit=bad))

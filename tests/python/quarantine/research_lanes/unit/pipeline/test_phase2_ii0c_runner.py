from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.confidence_contract import insufficient_confidence_calibration_block
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.phase2_ii0c_comparison import (
    Phase2II0CComparisonBundle,
)
from pysrc.meta_learning.phase2_ii0c_encoder import II0CEncoderTaskOutput
from pysrc.meta_learning.phase2_ii0c_tasks import (
    II0CMetaTaskPayload,
    II0CMetaTaskRequest,
    build_ii0c_meta_task_payload,
)
from pysrc.meta_learning.task_generator import derive_signal_ids_hash
from pysrc.pipeline.phase2_ii0c_runner import (
    II0C_DRY_RUN_SUMMARY_FILENAME,
    II0C_DRY_RUN_SUMMARY_LEGACY_FILENAME,
    II0C_PILOT_REPORT_FILENAME,
    PHASE2_II0C_EVIDENCE_SUBDIR,
    PHASE2_II0C_SUMMARY_FILENAME,
    Phase2II0CComparisonPack,
    Phase2II0CProviders,
    Phase2II0CRunRequest,
    default_ii0c_comparison_provider,
    default_ii0c_encoder_provider,
    run_phase2_ii0c_dry_run,
    run_phase2_ii0c_pilot,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _request(tmp_path: Path, seed: int = 11) -> Phase2II0CRunRequest:
    prices = pd.DataFrame(
        {
            "knowledge_time": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                utc=True,
            ),
            "valid_time": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                utc=True,
            ),
            "close": [100.0, 101.0],
        }
    )
    return Phase2II0CRunRequest(
        bundle_dir=tmp_path / "bundle",
        seed=seed,
        strategy_id="phase2_ii0c_scaffold",
        source_prices=prices,
        run_metadata={"timestamp_utc": "2024-01-02T00:00:00+00:00"},
    )


def test_phase2_ii0c_runner_emits_governed_triple_with_explicit_providers(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    sig_ids, sig_mask = build_fixed_slot_surface_from_sparse_slots({0: "ii0c.stub.signal"})
    signal_ids_hash = derive_signal_ids_hash(signal_ids=sig_ids, signal_mask=sig_mask)

    task_payload = build_ii0c_meta_task_payload(
        II0CMetaTaskRequest(
            regime_id="phase2_ii0c_scaffold::run_sha256_test",
            regime_class="sideways",
            support_set=("2024-01-01T00:00:00+00:00",),
            query_set=("2024-01-02T00:00:00+00:00",),
            horizon=1,
            signal_bindings={0: "ii0c.stub.signal"},
            signal_set_version="phase2_ii0c.scaffold.v1",
            pit_boundary="2024-01-01T00:00:00+00:00",
        )
    )
    assert task_payload.task_manifest_input.signal_ids_hash == signal_ids_hash

    encoder_output = II0CEncoderTaskOutput(
        embedding=(0.0,) * 64,
        metadata={
            "schema_version": "phase2.ii0c.encoder_stub.v1",
            "encoder_stub_version": "phase2.ii0c.encoder_stub.v1",
            "scaffold_reference_only": True,
            "signature": "encoder.sig::11",
            "task_id": task_payload.meta_task.task_id,
        },
    )
    comparison_pack = Phase2II0CComparisonPack(
        dataset_manifest={
            "pit_compliant": True,
            "knowledge_time_column": "knowledge_time",
            "content_hash": "sha256:" + "1" * 64,
            "content_hash_expected": "sha256:" + "1" * 64,
        },
        baseline_comparison={
            "baseline_kind": "xgboost_incumbent",
            "baseline_run_id": "xgboost-incumbent:phase2_ii0c_scaffold:task",
            "challenger_run_id": task_payload.meta_task.task_id,
            "splits_fingerprint": "sha256:" + "2" * 64,
            "data_fingerprint": "sha256:" + "3" * 64,
            "cost_assumptions_fingerprint": "sha256:" + "4" * 64,
            "data_parity": True,
            "split_parity": True,
            "cost_parity": True,
            "net_result_against_incumbent": "not_evaluated_scaffold",
        },
        shared_comparison_context={
            "baseline_run_id": "xgboost-incumbent:phase2_ii0c_scaffold:task",
            "challenger_run_id": task_payload.meta_task.task_id,
            "splits_fingerprint": "sha256:" + "2" * 64,
            "data_fingerprint": "sha256:" + "3" * 64,
            "cost_assumptions_fingerprint": "sha256:" + "4" * 64,
            "encoder_signature": "encoder.sig::11",
            "task_id": task_payload.meta_task.task_id,
        },
        cost_model={
            "mode": "non_promotable_governed_reference",
            "fingerprint": "sha256:" + "4" * 64,
        },
        slippage_model={
            "mode": "non_promotable_governed_reference",
            "fingerprint": "sha256:" + "4" * 64,
        },
        borrow_funding={
            "mode": "non_promotable_governed_reference",
            "fingerprint": "sha256:" + "4" * 64,
        },
        latency_fill={
            "mode": "non_promotable_governed_reference",
            "fingerprint": "sha256:" + "4" * 64,
        },
        inner_loop_gain_by_regime={"ii0c_scaffold": 0.0},
        harvey_t_statistic=0.0,
        encoder_coherence_score=0.0,
        crisis_episode_ic=0.0,
        forgetting_metric=0.0,
        plasticity_metric=0.0,
        entry_conditions={
            "phase_i_f_truth_boundary_closed": True,
            "phase_i_g_policy_baseline_available": True,
            "replay_fixture_dependency_satisfied": True,
            "governed_pit_access_path": True,
            "artifact_emission_capability": True,
            "frozen_incumbent_baseline_identified": True,
        },
        confidence_calibration=insufficient_confidence_calibration_block(
            reason="II-0C scaffold dry run does not claim calibrated encoder behavior.",
        ),
        overall_result="INSUFFICIENT",
        run_seed_root="0" * 64,
        seed_derivations=(
            ("task_sampling", "phase2_ii0c_scaffold:11"),
            ("checkpoint_reference_eval", "phase2_ii0c_scaffold:11:reference"),
        ),
    )

    def _task_provider(request: Phase2II0CRunRequest) -> II0CMetaTaskPayload:
        assert request.seed == 11
        return task_payload

    def _encoder_provider(
        request: Phase2II0CRunRequest, payload: II0CMetaTaskPayload
    ) -> II0CEncoderTaskOutput:
        assert payload.meta_task.task_id == task_payload.meta_task.task_id
        return encoder_output

    def _comparison_provider(
        request: Phase2II0CRunRequest,
        payload: II0CMetaTaskPayload,
        encoder_payload: II0CEncoderTaskOutput,
    ) -> Phase2II0CComparisonPack:
        assert payload.meta_task.task_id == task_payload.meta_task.task_id
        assert encoder_payload.metadata["signature"] == "encoder.sig::11"
        return comparison_pack

    providers = Phase2II0CProviders(
        task_provider=_task_provider,
        encoder_provider=_encoder_provider,
        comparison_provider=_comparison_provider,
    )

    result = run_phase2_ii0c_pilot(_request(tmp_path), providers=providers)

    assert result.phase == "II-0C"
    assert result.run_mode == "pilot"
    assert result.pilot_semantic_outcome == "HARNESS_EXERCISED_NON_PROMOTABLE"
    assert result.scaffold is True
    assert result.non_promotable is True
    assert result.gate_ii_deferred is True
    assert result.pilot_report_path.name == II0C_PILOT_REPORT_FILENAME
    assert result.pilot_report["research_only_surface"] is True
    assert result.pilot_report["schema_version"] == "phase2.ii0c.pilot_report.v1"
    assert (result.pilot_report_path).is_file()
    assert result.governed_evidence_dir.exists()
    assert (result.governed_evidence_dir / "task_manifest.json").is_file()
    assert (result.governed_evidence_dir / "meta_validity_report.json").is_file()
    assert (result.governed_evidence_dir / "execution_assumptions.json").is_file()
    assert result.governed_artifact_summary["current_governed_evidence"] is True
    assert result.governed_artifact_summary["governed_lane"] == "II-0B"
    assert result.task.signal_set_version == "phase2_ii0c.scaffold.v1"
    assert result.task_payload.meta_task.task_id == task_payload.meta_task.task_id
    assert result.encoder_output.metadata["scaffold_reference_only"] is True
    assert result.comparison_pack.baseline_comparison["baseline_kind"] == "xgboost_incumbent"


def test_phase2_ii0c_runner_fails_closed_when_governed_artifact_plumbing_is_missing(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    monkeypatch.setattr("pysrc.pipeline.phase2_ii0c_runner.emit_phase2_artifacts", None)

    with pytest.raises(RuntimeError, match="governed artifact plumbing"):
        run_phase2_ii0c_pilot(_request(tmp_path))


def test_phase2_ii0c_default_pilot_preserves_single_task_identity_across_evidence(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    result = run_phase2_ii0c_pilot(_request(tmp_path, seed=17))

    task_doc = json.loads(
        (result.governed_evidence_dir / "task_manifest.json").read_text(encoding="utf-8")
    )
    manifest_task_id = task_doc["tasks"][0]["task_id"]

    assert manifest_task_id == result.task_payload.meta_task.task_id
    assert result.encoder_output.metadata["task_id"] == manifest_task_id
    assert result.comparison_pack.baseline_comparison["challenger_run_id"] == manifest_task_id
    assert result.comparison_pack.shared_comparison_context["task_id"] == manifest_task_id
    pilot_disk = json.loads(
        (tmp_path / "bundle" / II0C_PILOT_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert pilot_disk["identity_binding_audit"]["task_id"] == manifest_task_id


def test_phase2_ii0c_default_dry_run_preserves_single_task_identity_across_evidence(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    result = run_phase2_ii0c_dry_run(
        output_dir=tmp_path,
        seed=23,
        timestamp_utc="2026-04-17T12:00:00+00:00",
    )

    task_doc = json.loads(
        (result.governed_evidence_dir / "task_manifest.json").read_text(encoding="utf-8")
    )
    manifest_task_id = task_doc["tasks"][0]["task_id"]

    assert result.summary_path.name == II0C_DRY_RUN_SUMMARY_FILENAME
    assert result.summary_path.name == PHASE2_II0C_SUMMARY_FILENAME
    assert result.run_mode == "dry_run"
    assert result.summary["research_only_surface"] is True
    assert result.summary["schema_version"] == "phase2.ii0c.dry_run_summary.v1"
    xs = result.summary["cross_section"]
    assert manifest_task_id == xs["task"]["task_id"]
    assert xs["encoder"]["task_id"] == manifest_task_id
    assert xs["comparison"]["challenger_run_id"] == manifest_task_id
    assert xs["shared_comparison_context"]["task_id"] == manifest_task_id

    legacy_path = tmp_path / II0C_DRY_RUN_SUMMARY_LEGACY_FILENAME
    assert legacy_path.is_file()
    legacy_disk = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert "cross_section" not in legacy_disk
    assert legacy_disk["task"]["task_id"] == manifest_task_id
    assert legacy_disk == result.summary["legacy_compact_summary"]


def _assert_dry_run_identity_failure_leaves_no_artifacts(tmp_path: Path) -> None:
    evidence_dir = tmp_path / PHASE2_II0C_EVIDENCE_SUBDIR
    assert not evidence_dir.exists()
    assert not (tmp_path / II0C_DRY_RUN_SUMMARY_FILENAME).exists()
    assert not (tmp_path / II0C_DRY_RUN_SUMMARY_LEGACY_FILENAME).exists()


def test_phase2_ii0c_dry_run_rejects_encoder_task_identity_drift_before_emission(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    def _bad_encoder(*, task_payload: II0CMetaTaskPayload, seed: int) -> II0CEncoderTaskOutput:
        output = default_ii0c_encoder_provider(task_payload=task_payload, seed=seed)
        metadata = dict(output.metadata)
        metadata["task_id"] = "wrong-task-id"
        return II0CEncoderTaskOutput(embedding=output.embedding, metadata=metadata)

    with pytest.raises(DataPreconditionError, match="encoder metadata"):
        run_phase2_ii0c_dry_run(
            output_dir=tmp_path,
            seed=29,
            timestamp_utc="2026-04-17T12:00:00+00:00",
            encoder_provider=_bad_encoder,
            artifact_emitter=lambda **_: pytest.fail(
                "artifact emitter must not run after identity drift"
            ),
        )

    _assert_dry_run_identity_failure_leaves_no_artifacts(tmp_path)


def test_phase2_ii0c_dry_run_rejects_comparison_challenger_identity_drift_before_emission(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    def _bad_comparison(
        *, task_payload: II0CMetaTaskPayload, seed: int
    ) -> Phase2II0CComparisonBundle:
        bundle = default_ii0c_comparison_provider(task_payload=task_payload, seed=seed)
        baseline = dict(bundle.baseline_comparison)
        baseline["challenger_run_id"] = "wrong-task-id"
        return Phase2II0CComparisonBundle(
            baseline_comparison=baseline,
            shared_comparison_context=bundle.shared_comparison_context,
        )

    with pytest.raises(DataPreconditionError, match="challenger_run_id"):
        run_phase2_ii0c_dry_run(
            output_dir=tmp_path,
            seed=31,
            timestamp_utc="2026-04-17T12:00:00+00:00",
            comparison_provider=_bad_comparison,
            artifact_emitter=lambda **_: pytest.fail(
                "artifact emitter must not run after identity drift"
            ),
        )

    _assert_dry_run_identity_failure_leaves_no_artifacts(tmp_path)


def test_phase2_ii0c_dry_run_rejects_baseline_shared_fingerprint_drift_before_emission(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    def _bad_comparison(
        *, task_payload: II0CMetaTaskPayload, seed: int
    ) -> Phase2II0CComparisonBundle:
        bundle = default_ii0c_comparison_provider(task_payload=task_payload, seed=seed)
        shared = dict(bundle.shared_comparison_context)
        shared["data_fingerprint"] = "sha256:drift"
        return Phase2II0CComparisonBundle(
            baseline_comparison=bundle.baseline_comparison,
            shared_comparison_context=shared,
        )

    with pytest.raises(DataPreconditionError, match="identical"):
        run_phase2_ii0c_dry_run(
            output_dir=tmp_path,
            seed=41,
            timestamp_utc="2026-04-17T12:00:00+00:00",
            comparison_provider=_bad_comparison,
            artifact_emitter=lambda **_: pytest.fail(
                "artifact emitter must not run after parity drift"
            ),
        )

    _assert_dry_run_identity_failure_leaves_no_artifacts(tmp_path)


def test_phase2_ii0c_dry_run_rejects_shared_context_identity_drift_before_emission(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed

    def _bad_comparison(
        *, task_payload: II0CMetaTaskPayload, seed: int
    ) -> Phase2II0CComparisonBundle:
        bundle = default_ii0c_comparison_provider(task_payload=task_payload, seed=seed)
        shared = dict(bundle.shared_comparison_context)
        shared["task_id"] = "wrong-task-id"
        return Phase2II0CComparisonBundle(
            baseline_comparison=bundle.baseline_comparison,
            shared_comparison_context=shared,
        )

    with pytest.raises(DataPreconditionError, match="shared comparison context"):
        run_phase2_ii0c_dry_run(
            output_dir=tmp_path,
            seed=37,
            timestamp_utc="2026-04-17T12:00:00+00:00",
            comparison_provider=_bad_comparison,
            artifact_emitter=lambda **_: pytest.fail(
                "artifact emitter must not run after identity drift"
            ),
        )

    _assert_dry_run_identity_failure_leaves_no_artifacts(tmp_path)

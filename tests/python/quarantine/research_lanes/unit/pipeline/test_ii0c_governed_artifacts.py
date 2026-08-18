from __future__ import annotations

import json
from pathlib import Path

import pytest

from marketmind_gate.gates.phase2_ml_evidence_shell import (
    Phase2MLEvidenceShellResult,
    Phase2MLEvidenceShellStatus,
)
from pysrc.meta.phase2_artifact_contract import (
    PhaseIIArtifactError,
    PhaseIIRunContext,
    emit_phase2_artifacts,
)
from pysrc.meta.rg09_threshold_catalog import THR_RG09_V03
from pysrc.meta.task_manifest_emitter import TaskManifestTaskInput
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.task_generator import derive_signal_ids_hash
from pysrc.pipeline.ii0c_governed_artifacts import (
    II0C_GOVERNED_SUMMARY_FILENAME,
    II0CGovernedArtifactError,
    emit_ii0c_governed_artifacts,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _task() -> TaskManifestTaskInput:
    ids, mask = build_fixed_slot_surface_from_sparse_slots({0: "ii0c.test.signal"})
    return TaskManifestTaskInput(
        regime_id="ii0c__scaffold",
        regime_class="bull",
        t0="2026-04-01T00:00:00+00:00",
        t1="2026-04-02T00:00:00+00:00",
        signal_ids_hash=derive_signal_ids_hash(signal_ids=ids, signal_mask=mask),
        signal_set_version="rg09.v1",
        support_last_timestamp="2026-04-01T12:00:00+00:00",
        signal_ids=ids,
        signal_mask=mask,
        active_k=sum(1 for item in mask if item),
    )


def _ctx(tmp_path: Path) -> PhaseIIRunContext:
    return PhaseIIRunContext(
        output_dir=tmp_path,
        seed=11,
        timestamp_utc="2026-04-17T12:00:00Z",
        tasks=[_task()],
        dataset_manifest={
            "pit_compliant": True,
            "knowledge_time_column": "knowledge_time",
            "content_hash": "a",
            "content_hash_expected": "a",
        },
        inner_loop_gain_by_regime={"bull": 0.0},
        harvey_t_statistic=1.0,
        encoder_coherence_score=1.0,
        crisis_episode_ic=1.0,
        forgetting_metric=1.0,
        plasticity_metric=1.0,
        baseline_comparison={
            "baseline_kind": "xgboost_incumbent",
            "baseline_run_id": "xgboost-incumbent-11",
            "challenger_run_id": "ii0c-scaffold-11",
            "splits_fingerprint": "splits:11",
            "data_fingerprint": "data:11",
            "cost_assumptions_fingerprint": "cost:11",
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
            "splits_fingerprint": "splits:11",
            "data_fingerprint": "data:11",
            "cost_assumptions_fingerprint": "cost:11",
        },
        threshold_references=(
            {
                "threshold_id": THR_RG09_V03,
                "consumer": "ii0c.test.harvey_t_statistic",
                "gate_critical": False,
                "usage_role": "meta_validity_report.harvey_t_statistic",
            },
        ),
    )


@pytest.mark.determinism("d1")
def test_emit_ii0c_governed_artifacts_emits_wrapper_and_validates_triple(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ctx = _ctx(tmp_path)

    result = emit_ii0c_governed_artifacts(run_context=ctx)

    for name in ("task_manifest.json", "meta_validity_report.json", "execution_assumptions.json"):
        assert (tmp_path / name).is_file()
        payload = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert isinstance(payload["content_hash"], dict)
        assert payload["content_hash"]["value"].startswith("sha256:")

    assert result.wrapper_summary_path.name == II0C_GOVERNED_SUMMARY_FILENAME
    assert result.wrapper_summary["phase"] == "II-0C"
    assert result.wrapper_summary["non_promotable"] is True
    assert result.wrapper_summary["scaffold_only"] is True
    assert result.wrapper_summary["wrapped_governed_phase"] == "II-0B"
    assert result.wrapper_summary["content_hash_verified"] is True
    assert result.wrapper_summary["shell_structurally_usable"] is True
    assert result.shell_result.status == Phase2MLEvidenceShellStatus.EVIDENCE_STRUCTURALLY_USABLE
    assert result.shell_result.reason_codes == ()
    assert result.wrapper_summary["threshold_governance"]["references"]
    assert (tmp_path / II0C_GOVERNED_SUMMARY_FILENAME).is_file()


@pytest.mark.determinism("d1")
def test_emit_ii0c_governed_artifacts_fails_closed_on_tampered_content_hash(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    ctx = _ctx(tmp_path)
    emit_phase2_artifacts(ctx)

    task_path = tmp_path / "task_manifest.json"
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["content_hash"]["value"] = "sha256:" + ("0" * 64)
    task_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(
        "pysrc.pipeline.ii0c_governed_artifacts.emit_phase2_artifacts",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(PhaseIIArtifactError, match="content_hash"):
        emit_ii0c_governed_artifacts(run_context=ctx)
    assert not (tmp_path / II0C_GOVERNED_SUMMARY_FILENAME).exists()


@pytest.mark.determinism("d1")
def test_emit_ii0c_governed_artifacts_rejects_non_usable_shell(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    ctx = _ctx(tmp_path)

    incomplete = Phase2MLEvidenceShellResult(
        status=Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE,
        artifacts_present={
            "task_manifest.json": True,
            "meta_validity_report.json": True,
            "execution_assumptions.json": True,
        },
        schema_ok={
            "task_manifest.json": True,
            "meta_validity_report.json": True,
            "execution_assumptions.json": True,
        },
        binding_ok={
            "task_manifest.json": True,
            "meta_validity_report.json": True,
            "execution_assumptions.json": True,
        },
        promotable_claim_emitted=False,
        reason_codes=("SCHEMA_INVALID",),
        evidence={"threshold_governance": {"references": []}},
    )
    monkeypatch.setattr(
        "pysrc.pipeline.ii0c_governed_artifacts.evaluate_phase2_ml_evidence_shell",
        lambda *_args, **_kwargs: incomplete,
    )

    with pytest.raises(II0CGovernedArtifactError, match="structurally usable"):
        emit_ii0c_governed_artifacts(run_context=ctx)
    assert not (tmp_path / II0C_GOVERNED_SUMMARY_FILENAME).exists()

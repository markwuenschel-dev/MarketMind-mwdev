"""Governed JSON Schema vs Phase II contract for W1 ``challenger_surface`` on ``meta_validity_report``."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from marketmind_gate.gates.schema import load_schema, validate_against_schema
from pysrc.meta.phase2_artifact_contract import (
    W1_CHALLENGER_SURFACE_SUMMARY_REQUIRED_KEYS_V1,
    PhaseIIRunContext,
    canonical_content_hash,
    derive_phase2_task_pool_identity_hash,
    emit_phase2_artifacts,
)
from pysrc.meta.seed_policy import derive_run_id
from pysrc.meta.task_manifest_emitter import TaskManifestTaskInput
from pysrc.meta_learning.confidence_contract import synthetic_confidence_calibration_pass_block
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.task_generator import derive_signal_ids_hash


def _task() -> TaskManifestTaskInput:
    ids, mask = build_fixed_slot_surface_from_sparse_slots({0: "phase2.challenger_schema.test.sig"})
    sig_hash = derive_signal_ids_hash(signal_ids=ids, signal_mask=mask)
    ak = sum(1 for m in mask if m)
    return TaskManifestTaskInput(
        regime_id="trend_bull__stable",
        regime_class="bull",
        t0="2020-01-01T00:00:00+00:00",
        t1="2020-03-01T00:00:00+00:00",
        signal_ids_hash=sig_hash,
        signal_set_version="rg09.v1",
        support_last_timestamp="2020-02-15T00:00:00+00:00",
        signal_ids=ids,
        signal_mask=mask,
        active_k=ak,
    )


def _dataset_ok() -> dict[str, object]:
    return {
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
        "content_hash": "a",
        "content_hash_expected": "a",
    }


def _shared() -> dict[str, object]:
    return {
        "splits_fingerprint": "split:abc",
        "data_fingerprint": "data:def",
        "cost_assumptions_fingerprint": "cost:ghi",
        "lane": "phase2-0",
    }


def _cost_stack() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    return (
        {"commission_bps": 1.0},
        {"model": "fixed_bps"},
        {"rate_bps": 0.0},
        {"latency_ms": 5},
    )


def _baseline_ii0c() -> dict[str, object]:
    return {
        "baseline_kind": "xgboost_incumbent",
        "baseline_run_id": "xgboost-incumbent-1",
        "challenger_run_id": "challenger-1",
        "splits_fingerprint": "split:abc",
        "data_fingerprint": "data:def",
        "cost_assumptions_fingerprint": "cost:ghi",
        "data_parity": True,
        "split_parity": True,
        "cost_parity": True,
        "net_result_against_incumbent": "not_evaluated_non_promotable",
    }


def _challenger_surface_summary(
    *,
    task_pool_hash: str,
    data_fp: str,
    splits_fp: str,
    cost_fp: str,
    model_state_hash: str | None = None,
    lineage_ok: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": "w1_challenger_surface.v1",
        "source": "unit_test_allocator.v1",
        "model_family": "reptile_meta",
        "task_pool_hash": task_pool_hash,
        "data_fingerprint": data_fp,
        "splits_fingerprint": splits_fp,
        "cost_assumptions_fingerprint": cost_fp,
        "signal_set_version": "rg09.v1",
        "leakage_policy": "pit_support_only_no_query_labels",
        "uses_query_labels": False,
        "uses_xgboost_outputs": False,
        "uses_post_query_metrics": False,
        "governed_checkpoint_lineage_verified": lineage_ok,
    }
    if model_state_hash is not None:
        out["model_state_hash"] = model_state_hash
    return out


def _real_baseline_comparison_with_challenger(
    *,
    task_pool_hash: str,
    challenger_run_id: str,
    cost_assumptions: dict[str, float],
    challenger_surface: dict[str, Any],
    challenger_surface_content_hash: str | None = None,
) -> dict[str, Any]:
    data_fp = "sha256:" + "d" * 64
    splits_fp = "sha256:" + "e" * 64
    cost_fp = canonical_content_hash({"cost_assumptions": cost_assumptions})
    assert str(challenger_surface["data_fingerprint"]) == data_fp
    assert str(challenger_surface["splits_fingerprint"]) == splits_fp
    assert str(challenger_surface["cost_assumptions_fingerprint"]) == cost_fp
    assert str(challenger_surface["task_pool_hash"]) == task_pool_hash
    bc: dict[str, Any] = {
        "baseline_identity": "xgboost_bocpd_incumbent_v1",
        "challenger_run_id": challenger_run_id,
        "data_parity": "governed_real_meta_task_pool",
        "split_parity": "walk_forward_folds_matched",
        "assumption_parity": "cost_assumptions_matched",
        "cost_assumptions": cost_assumptions,
        "aggregate_metrics": {
            "mean_net_sharpe": 0.1,
            "mean_net_sharpe_normal_null_p": 0.2,
            "mean_gross_ic": 0.03,
            "mean_crisis_ic": None,
            "mean_max_drawdown": -1.0,
            "mean_turnover": 0.4,
            "n_folds_with_crisis": 1,
        },
        "n_folds": 2,
        "task_pool_hash": task_pool_hash,
        "threshold_references": [
            {"id": "THR-W1-NS01", "state": "PROVISIONAL", "current_expression": "expr-ns"},
            {"id": "THR-W1-IC01", "state": "PROVISIONAL", "current_expression": "expr-ic"},
        ],
        "net_result": "INSUFFICIENT_REAL_DATA",
        "schema_version": "baseline_comparison.v1",
        "promotion_note": "Governed real pool. GATE-II remains DEFERRED.",
        "evidence_source": {
            "market_data_source": "governed_historical_dataview",
            "label_source": "governed_pit_regime_label_surface",
            "target_source": "query_realized_net_utility",
            "challenger_source": "unit_test_allocator.v1",
            "real_market_data_evidence": True,
            "w1_gate_closure_eligible": True,
            "pit_heterogeneity_governance_acknowledged": False,
        },
        "data_fingerprint": data_fp,
        "splits_fingerprint": splits_fp,
        "cost_assumptions_fingerprint": cost_fp,
        "challenger_surface": challenger_surface,
    }
    if challenger_surface_content_hash is not None:
        bc["challenger_surface_content_hash"] = challenger_surface_content_hash
    return bc


def _emit_meta_validity_base(tmp_path: Path) -> dict[str, Any]:
    cm, sm, bf, lf = _cost_stack()
    task = _task()
    ctx = PhaseIIRunContext(
        output_dir=tmp_path,
        seed=201,
        timestamp_utc="2026-04-24T12:00:00Z",
        tasks=[task],
        dataset_manifest=_dataset_ok(),
        inner_loop_gain_by_regime={"bull": 0.01},
        harvey_t_statistic=1.0,
        encoder_coherence_score=1.0,
        crisis_episode_ic=0.0,
        forgetting_metric=0.0,
        plasticity_metric=0.5,
        baseline_comparison=_baseline_ii0c(),
        cost_model=cm,
        slippage_model=sm,
        borrow_funding=bf,
        latency_fill=lf,
        shared_comparison_context=_shared(),
        confidence_calibration=synthetic_confidence_calibration_pass_block(ece_value=0.05),
    )
    emit_phase2_artifacts(ctx)
    raw = json.loads((tmp_path / "meta_validity_report.json").read_text(encoding="utf-8"))
    return cast(dict[str, Any], raw)


def _phase2_mv_schema() -> dict[str, Any]:
    return load_schema(Path("schemas/phase2_meta_validity_report.schema.json"))


@pytest.mark.determinism("d1")
def test_meta_validity_real_lane_with_learned_challenger_surface_passes_governed_schema(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    mv = _emit_meta_validity_base(tmp_path)
    task = _task()
    task_pool_hash = derive_phase2_task_pool_identity_hash([task])
    rid = derive_run_id(201)
    cost = {"spread_bps": 5.0, "slippage_bps": 2.0, "borrow_rate_ann": 0.005}
    data_fp = "sha256:" + "d" * 64
    splits_fp = "sha256:" + "e" * 64
    cost_fp = canonical_content_hash({"cost_assumptions": cost})
    msh = "sha256:" + "a" * 64
    csh = "sha256:" + "f" * 64
    cs = _challenger_surface_summary(
        task_pool_hash=task_pool_hash,
        data_fp=data_fp,
        splits_fp=splits_fp,
        cost_fp=cost_fp,
        model_state_hash=msh,
        lineage_ok=True,
    )
    mv2 = deepcopy(mv)
    mv2["baseline_comparison"] = _real_baseline_comparison_with_challenger(
        task_pool_hash=task_pool_hash,
        challenger_run_id=rid,
        cost_assumptions=cost,
        challenger_surface=cs,
        challenger_surface_content_hash=csh,
    )
    schema = _phase2_mv_schema()
    result = validate_against_schema(mv2, schema, file="meta_validity_report.json")
    assert result.valid, f"errors={result.errors!r}"


@pytest.mark.determinism("d1")
def test_challenger_surface_schema_rejects_non_boolean_lineage_flag(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    mv = _emit_meta_validity_base(tmp_path)
    task_pool_hash = derive_phase2_task_pool_identity_hash([_task()])
    rid = derive_run_id(201)
    cost = {"spread_bps": 5.0, "slippage_bps": 2.0, "borrow_rate_ann": 0.005}
    data_fp = "sha256:" + "d" * 64
    splits_fp = "sha256:" + "e" * 64
    cost_fp = canonical_content_hash({"cost_assumptions": cost})
    cs = _challenger_surface_summary(
        task_pool_hash=task_pool_hash,
        data_fp=data_fp,
        splits_fp=splits_fp,
        cost_fp=cost_fp,
        lineage_ok=True,
    )
    bad_cs: dict[str, Any] = dict(cs)
    bad_cs["governed_checkpoint_lineage_verified"] = "not-a-bool"
    mv2 = deepcopy(mv)
    mv2["baseline_comparison"] = _real_baseline_comparison_with_challenger(
        task_pool_hash=task_pool_hash,
        challenger_run_id=rid,
        cost_assumptions=cost,
        challenger_surface=bad_cs,
    )
    schema = _phase2_mv_schema()
    result = validate_against_schema(mv2, schema, file="meta_validity_report.json")
    assert not result.valid
    assert result.errors


@pytest.mark.determinism("d1")
def test_challenger_surface_schema_rejects_bad_model_state_hash(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    mv = _emit_meta_validity_base(tmp_path)
    task_pool_hash = derive_phase2_task_pool_identity_hash([_task()])
    rid = derive_run_id(201)
    cost = {"spread_bps": 5.0, "slippage_bps": 2.0, "borrow_rate_ann": 0.005}
    data_fp = "sha256:" + "d" * 64
    splits_fp = "sha256:" + "e" * 64
    cost_fp = canonical_content_hash({"cost_assumptions": cost})
    cs = _challenger_surface_summary(
        task_pool_hash=task_pool_hash,
        data_fp=data_fp,
        splits_fp=splits_fp,
        cost_fp=cost_fp,
        model_state_hash="sha256:NOTHEX",
        lineage_ok=False,
    )
    mv2 = deepcopy(mv)
    mv2["baseline_comparison"] = _real_baseline_comparison_with_challenger(
        task_pool_hash=task_pool_hash,
        challenger_run_id=rid,
        cost_assumptions=cost,
        challenger_surface=cs,
    )
    schema = _phase2_mv_schema()
    result = validate_against_schema(mv2, schema, file="meta_validity_report.json")
    assert not result.valid
    assert result.errors


@pytest.mark.determinism("d1")
def test_challenger_surface_schema_rejects_leakage_const_violation(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    mv = _emit_meta_validity_base(tmp_path)
    task_pool_hash = derive_phase2_task_pool_identity_hash([_task()])
    rid = derive_run_id(201)
    cost = {"spread_bps": 5.0, "slippage_bps": 2.0, "borrow_rate_ann": 0.005}
    data_fp = "sha256:" + "d" * 64
    splits_fp = "sha256:" + "e" * 64
    cost_fp = canonical_content_hash({"cost_assumptions": cost})
    cs = _challenger_surface_summary(
        task_pool_hash=task_pool_hash,
        data_fp=data_fp,
        splits_fp=splits_fp,
        cost_fp=cost_fp,
    )
    cs["uses_query_labels"] = True
    mv2 = deepcopy(mv)
    mv2["baseline_comparison"] = _real_baseline_comparison_with_challenger(
        task_pool_hash=task_pool_hash,
        challenger_run_id=rid,
        cost_assumptions=cost,
        challenger_surface=cs,
    )
    schema = _phase2_mv_schema()
    result = validate_against_schema(mv2, schema, file="meta_validity_report.json")
    assert not result.valid
    assert result.errors


@pytest.mark.determinism("d1")
def test_challenger_surface_schema_rejects_unknown_nested_property(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    mv = _emit_meta_validity_base(tmp_path)
    task_pool_hash = derive_phase2_task_pool_identity_hash([_task()])
    rid = derive_run_id(201)
    cost = {"spread_bps": 5.0, "slippage_bps": 2.0, "borrow_rate_ann": 0.005}
    data_fp = "sha256:" + "d" * 64
    splits_fp = "sha256:" + "e" * 64
    cost_fp = canonical_content_hash({"cost_assumptions": cost})
    cs = _challenger_surface_summary(
        task_pool_hash=task_pool_hash,
        data_fp=data_fp,
        splits_fp=splits_fp,
        cost_fp=cost_fp,
    )
    cs["drift_field"] = True
    mv2 = deepcopy(mv)
    mv2["baseline_comparison"] = _real_baseline_comparison_with_challenger(
        task_pool_hash=task_pool_hash,
        challenger_run_id=rid,
        cost_assumptions=cost,
        challenger_surface=cs,
    )
    schema = _phase2_mv_schema()
    result = validate_against_schema(mv2, schema, file="meta_validity_report.json")
    assert not result.valid
    assert result.errors
    assert any("drift_field" in e.message for e in result.errors)


@pytest.mark.determinism("d1")
def test_w1_challenger_surface_summary_required_keys_match_schema_def(
    tmp_path: Path, deterministic_seed: int
) -> None:
    """``W1_CHALLENGER_SURFACE_SUMMARY_REQUIRED_KEYS_V1`` matches ``$defs.w1_challenger_surface_summary_v1``."""
    _ = deterministic_seed
    _ = tmp_path
    schema = _phase2_mv_schema()
    required = schema["$defs"]["w1_challenger_surface_summary_v1"]["required"]
    assert tuple(required) == W1_CHALLENGER_SURFACE_SUMMARY_REQUIRED_KEYS_V1
    props = frozenset(schema["$defs"]["w1_challenger_surface_summary_v1"]["properties"])
    assert frozenset(W1_CHALLENGER_SURFACE_SUMMARY_REQUIRED_KEYS_V1).issubset(props)
    assert "model_state_hash" in props

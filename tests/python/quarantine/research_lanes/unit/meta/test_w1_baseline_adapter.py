"""Unit tests for :mod:`pysrc.meta.w1_baseline_adapter`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import pysrc.meta.phase2_artifact_contract as phase2_ac
from pysrc.meta.phase2_artifact_contract import canonical_content_hash
from pysrc.meta.w1_baseline_adapter import W1AdapterError, build_baseline_comparison
from pysrc.meta.w1_baseline_types import (
    W1AggregateMetrics,
    W1BaselineResult,
    W1ComparisonFingerprints,
    W1CostAssumptions,
    W1EvidenceSource,
    W1ThresholdReference,
)


def _w1_result(
    *,
    baseline_id: str = "xgboost_bocpd_incumbent_v1",
    n_folds: int = 2,
    threshold_references: tuple[W1ThresholdReference, ...] | None = None,
) -> W1BaselineResult:
    refs = threshold_references or (
        W1ThresholdReference("THR-W1-NS01", "PROVISIONAL", "net_sharpe > 0.5 ⚑ VALIDATE"),
        W1ThresholdReference("THR-W1-IC01", "PROVISIONAL", "mean_gross_ic > 0.003 ⚑ VALIDATE"),
    )
    return W1BaselineResult(
        schema_version="w1_baseline.v2",
        baseline_id=baseline_id,
        run_timestamp_utc="2026-04-01T00:00:00Z",
        n_folds=n_folds,
        fold_results=(),
        aggregate=W1AggregateMetrics(
            mean_net_sharpe=0.55,
            mean_net_sharpe_normal_null_p=0.27,
            mean_gross_ic=0.038,
            mean_crisis_ic=-0.014,
            mean_max_drawdown=-3.2,
            mean_turnover=0.52,
            n_folds_with_crisis=5,
        ),
        cost_assumptions=W1CostAssumptions(5.0, 2.0, 0.005),
        threshold_references=refs,
        gate_ii_posture="DEFERRED",
        promotion_note="Bounded stub harness. Not allocator promotion evidence. GATE-II remains DEFERRED.",
        content_hash={
            "algorithm": "sha256",
            "canonicalization": "json.sort_keys.no_ws.omit_content_hash.omit_run_timestamp.v1",
            "value": "sha256:" + "a" * 64,
        },
    )


def test_build_baseline_comparison_returns_dict() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert isinstance(out, dict)


def test_baseline_identity_in_output() -> None:
    w1 = _w1_result(baseline_id="xgboost_custom_v1")
    out = build_baseline_comparison(
        w1,
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["baseline_identity"] == "xgboost_custom_v1"


def test_net_result_is_stub_constant() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["net_result"] == "STUB_PENDING_REAL_DATA"


def test_cost_assumptions_propagated() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["cost_assumptions"] == {
        "spread_bps": 5.0,
        "slippage_bps": 2.0,
        "borrow_rate_ann": 0.005,
    }


def test_threshold_references_propagated() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["threshold_references"] == [
        {
            "id": "THR-W1-NS01",
            "state": "PROVISIONAL",
            "current_expression": "net_sharpe > 0.5 ⚑ VALIDATE",
        },
        {
            "id": "THR-W1-IC01",
            "state": "PROVISIONAL",
            "current_expression": "mean_gross_ic > 0.003 ⚑ VALIDATE",
        },
    ]


def test_schema_version() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["schema_version"] == "baseline_comparison.v1"


def test_gate_ii_posture_in_note() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert "GATE-II remains DEFERRED" in out["promotion_note"]
    assert (
        out["promotion_note"]
        == "Bounded stub harness. Not allocator promotion evidence. GATE-II remains DEFERRED."
    )


def test_raises_on_empty_baseline_id() -> None:
    w1 = _w1_result(baseline_id="   ")
    with pytest.raises(W1AdapterError, match="baseline_id"):
        build_baseline_comparison(
            w1,
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="sha256:" + "c" * 64,
        )


def test_raises_on_wrong_baseline_prefix() -> None:
    w1 = _w1_result(baseline_id="lightgbm_x")
    with pytest.raises(W1AdapterError, match="xgboost_"):
        build_baseline_comparison(
            w1,
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="sha256:" + "c" * 64,
        )


def test_raises_on_empty_challenger_run_id() -> None:
    with pytest.raises(W1AdapterError, match="challenger_run_id"):
        build_baseline_comparison(
            _w1_result(),
            challenger_run_id="   ",
            task_pool_hash="sha256:" + "c" * 64,
        )


def test_raises_on_empty_task_pool_hash() -> None:
    with pytest.raises(W1AdapterError, match="task_pool_hash"):
        build_baseline_comparison(
            _w1_result(),
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="",
        )


def test_raises_on_zero_folds() -> None:
    w1 = _w1_result(n_folds=0)
    with pytest.raises(W1AdapterError, match="n_folds"):
        build_baseline_comparison(
            w1,
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="sha256:" + "c" * 64,
        )


def test_raises_on_missing_threshold_id() -> None:
    w1 = _w1_result(
        threshold_references=(W1ThresholdReference("THR-W1-NS01", "PROVISIONAL", "e ⚑ VALIDATE"),),
    )
    with pytest.raises(W1AdapterError, match="THR-W1-IC01|threshold"):
        build_baseline_comparison(
            w1,
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="sha256:" + "c" * 64,
        )


@patch(
    "pysrc.meta.w1_baseline_adapter._cost_to_dict",
    return_value={"spread_bps": 1.0, "slippage_bps": 2.0, "borrow_rate_ann": True},
)
def test_raises_on_invalid_cost_field_type(_mock_cost: object) -> None:
    with pytest.raises(W1AdapterError, match="borrow_rate_ann"):
        build_baseline_comparison(
            _w1_result(),
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="sha256:" + "c" * 64,
        )


def test_raises_on_non_provisional_threshold() -> None:
    w1 = _w1_result(
        threshold_references=(
            W1ThresholdReference("THR-W1-NS01", "PROVISIONAL", "e1 ⚑ VALIDATE"),
            W1ThresholdReference("THR-W1-IC01", "VALIDATED", "e2 ⚑ VALIDATE"),
        ),
    )
    with pytest.raises(W1AdapterError, match="PROVISIONAL"):
        build_baseline_comparison(
            w1,
            challenger_run_id="run.sha256:" + "b" * 64,
            task_pool_hash="sha256:" + "c" * 64,
        )


def test_absence_sentinel_shape() -> None:
    sentinel = {
        "schema_version": "baseline_comparison.v1",
        "net_result": "NOT_RUN",
        "promotion_note": (
            "W1 harness not run for this evidence surface. GATE-II remains DEFERRED."
        ),
    }
    phase2_ac._validate_baseline_comparison(sentinel)


def _fingerprints_for_cost(cost: W1CostAssumptions) -> W1ComparisonFingerprints:
    cdict = {
        "spread_bps": cost.spread_bps,
        "slippage_bps": cost.slippage_bps,
        "borrow_rate_ann": cost.borrow_rate_ann,
    }
    cfp = canonical_content_hash({"cost_assumptions": cdict})
    dfp = "sha256:" + "d" * 64
    sfp = "sha256:" + "e" * 64
    return W1ComparisonFingerprints(
        data_fingerprint=dfp, splits_fingerprint=sfp, cost_assumptions_fingerprint=cfp
    )


def test_governed_real_without_challenger_is_insufficient() -> None:
    cost = W1CostAssumptions(5.0, 2.0, 0.005)
    w1 = W1BaselineResult(
        schema_version="w1_baseline.v2",
        baseline_id="xgboost_bocpd_incumbent_v1",
        run_timestamp_utc="2026-04-01T00:00:00Z",
        n_folds=2,
        fold_results=(),
        aggregate=W1AggregateMetrics(
            mean_net_sharpe=0.55,
            mean_net_sharpe_normal_null_p=0.27,
            mean_gross_ic=0.038,
            mean_crisis_ic=-0.014,
            mean_max_drawdown=-3.2,
            mean_turnover=0.52,
            n_folds_with_crisis=5,
        ),
        cost_assumptions=cost,
        threshold_references=(
            W1ThresholdReference("THR-W1-NS01", "PROVISIONAL", "net_sharpe > 0.5 ⚑ VALIDATE"),
            W1ThresholdReference("THR-W1-IC01", "PROVISIONAL", "mean_gross_ic > 0.003 ⚑ VALIDATE"),
        ),
        gate_ii_posture="DEFERRED",
        promotion_note="Governed real MetaTask pool. Not allocator promotion evidence. GATE-II remains DEFERRED.",
        content_hash={
            "algorithm": "sha256",
            "canonicalization": "json.sort_keys.no_ws.omit_content_hash.omit_run_timestamp.v1",
            "value": "sha256:" + "a" * 64,
        },
        evidence_lane="governed_real",
        challenger_aggregate=None,
        evidence_source=W1EvidenceSource(
            market_data_source="governed_historical_dataview",
            label_source="governed_pit_regime_label_surface",
            target_source="query_realized_net_utility",
            challenger_source="absent",
            real_market_data_evidence=True,
            w1_gate_closure_eligible=False,
        ),
        comparison_fingerprints=_fingerprints_for_cost(cost),
    )
    out = build_baseline_comparison(
        w1,
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["net_result"] == "INSUFFICIENT_REAL_DATA"
    assert out["evidence_source"]["w1_gate_closure_eligible"] is False
    assert out["evidence_source"]["pit_heterogeneity_governance_acknowledged"] is False


def test_governed_real_challenger_without_closure_stays_insufficient() -> None:
    """Challenger metrics exist but taxonomy is gated on w1_gate_closure_eligible."""
    cost = W1CostAssumptions(5.0, 2.0, 0.005)
    fp_lane = _fingerprints_for_cost(cost)
    ch = W1AggregateMetrics(
        mean_net_sharpe=99.0,
        mean_net_sharpe_normal_null_p=0.01,
        mean_gross_ic=0.5,
        mean_crisis_ic=None,
        mean_max_drawdown=-0.1,
        mean_turnover=0.01,
        n_folds_with_crisis=0,
    )
    w1 = W1BaselineResult(
        schema_version="w1_baseline.v2",
        baseline_id="xgboost_bocpd_incumbent_v1",
        run_timestamp_utc="2026-04-01T00:00:00Z",
        n_folds=2,
        fold_results=(),
        aggregate=W1AggregateMetrics(
            mean_net_sharpe=0.55,
            mean_net_sharpe_normal_null_p=0.27,
            mean_gross_ic=0.038,
            mean_crisis_ic=-0.014,
            mean_max_drawdown=-3.2,
            mean_turnover=0.52,
            n_folds_with_crisis=5,
        ),
        cost_assumptions=cost,
        threshold_references=(
            W1ThresholdReference("THR-W1-NS01", "PROVISIONAL", "net_sharpe > 0.5 ⚑ VALIDATE"),
            W1ThresholdReference("THR-W1-IC01", "PROVISIONAL", "mean_gross_ic > 0.003 ⚑ VALIDATE"),
        ),
        gate_ii_posture="DEFERRED",
        promotion_note="Governed real MetaTask pool. Not allocator promotion evidence. GATE-II remains DEFERRED.",
        content_hash={
            "algorithm": "sha256",
            "canonicalization": "json.sort_keys.no_ws.omit_content_hash.omit_run_timestamp.v1",
            "value": "sha256:" + "a" * 64,
        },
        evidence_lane="governed_real",
        challenger_aggregate=ch,
        evidence_source=W1EvidenceSource(
            market_data_source="governed_historical_dataview",
            label_source="governed_pit_regime_label_surface",
            target_source="query_realized_net_utility",
            challenger_source="test_fixture_allocator.v1",
            real_market_data_evidence=True,
            w1_gate_closure_eligible=False,
            pit_heterogeneity_governance_acknowledged=False,
        ),
        comparison_fingerprints=fp_lane,
        heterogeneous_task_pits_supported=False,
        challenger_surface_summary={
            "schema_version": "w1_challenger_surface.v1",
            "source": "test_fixture_allocator.v1",
            "model_family": "fixture",
            "task_pool_hash": "sha256:" + "c" * 64,
            "data_fingerprint": fp_lane.data_fingerprint,
            "splits_fingerprint": fp_lane.splits_fingerprint,
            "cost_assumptions_fingerprint": fp_lane.cost_assumptions_fingerprint,
            "signal_set_version": "1",
            "leakage_policy": "pit_support_only_no_query_labels",
            "uses_query_labels": False,
            "uses_xgboost_outputs": False,
            "uses_post_query_metrics": False,
            "governed_checkpoint_lineage_verified": False,
        },
        challenger_surface_content_hash="sha256:" + "f" * 64,
    )
    out = build_baseline_comparison(
        w1,
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["net_result"] == "INSUFFICIENT_REAL_DATA"
    assert out["evidence_source"]["w1_gate_closure_eligible"] is False


def test_parity_literals_and_aggregate() -> None:
    out = build_baseline_comparison(
        _w1_result(),
        challenger_run_id="run.sha256:" + "b" * 64,
        task_pool_hash="sha256:" + "c" * 64,
    )
    assert out["data_parity"] == "synthetic_pool_matched"
    assert out["split_parity"] == "walk_forward_folds_matched"
    assert out["assumption_parity"] == "cost_assumptions_matched"
    assert out["aggregate_metrics"]["mean_net_sharpe"] == 0.55
    assert out["n_folds"] == 2
    assert out["challenger_run_id"] == "run.sha256:" + "b" * 64
    assert out["task_pool_hash"] == "sha256:" + "c" * 64

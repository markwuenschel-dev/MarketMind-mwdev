"""Unit tests for W1 governed-real pre-closure evidence sanity (ADR-005)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.python.unit.meta.test_w1_baseline_runner import _bull_only_task

from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.w1_baseline_config import (
    W1BaselineConfig,
    W1BaselineConfigError,
    W1EvidenceSanityConfig,
)
from pysrc.meta.w1_baseline_types import W1FoldResult
from pysrc.meta.w1_evidence_fingerprints import derive_w1_splits_fingerprint
from pysrc.meta.w1_evidence_sanity import W1FoldSanityDiag, w1_evidence_sanity_closure_ok
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _same_window_task(task_id: str, *, emb_scale: float) -> MetaTask:
    """Two tasks sharing the same calendar window fingerprint (support/query geometry)."""
    from datetime import UTC, datetime, timedelta

    day = datetime(2024, 7, 1, tzinfo=UTC)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(6))
    query = tuple((day + timedelta(days=20 + j)).isoformat() for j in range(4))
    emb = np.full(4, emb_scale, dtype=np.float32)
    return MetaTask(
        task_id=task_id,
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class="bull",
        t0=support[0],
        t1=(day + timedelta(days=45)).isoformat(),
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=_mask(),
        signal_set_version="1",
        signal_ids_hash="sha256:w1_stub",
        horizon=1,
        active_k=1,
        regime_embedding=emb,
    )


def _fold_result_stub(
    *, gross_ic: float = 0.0, net_sharpe: float = 0.0, turnover: float = 0.0
) -> W1FoldResult:
    return W1FoldResult(
        fold_index=0,
        fold_start="2020-01-02T00:00:00Z",
        fold_end="2020-01-03T00:00:00Z",
        net_sharpe=float(net_sharpe),
        gross_ic=float(gross_ic),
        crisis_ic=None,
        max_drawdown=0.0,
        turnover=float(turnover),
        n_query_rows_scored=2,
        n_unique_tasks_scored=2,
        regime_breakdown=dict.fromkeys(REGIME_CLASS_ORDER),
    )


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_flat_query_targets_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): 0.5 for t in tasks}
    ts = {str(t.task_id): float(i) * 0.01 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 1.0, 1.0, 1.0),)
    fr = (_fold_result_stub(gross_ic=0.1),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr,
        fold_results_ch=None,
        cfg=cfg,
    )
    assert ok is False
    assert any("query_target_std_below_min" in r for r in rs)


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_flat_support_targets_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): float(i) * 0.02 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): 1.0 for t in tasks}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 1.0, 1.0, 1.0),)
    fr = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr,
        fold_results_ch=None,
        cfg=cfg,
    )
    assert ok is False
    assert any("support_target_std_below_min" in r for r in rs)


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_distinct_query_values_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1), _bull_only_task(2))
    tq = {str(t.task_id): 0.7 for t in tasks}
    ts = {str(t.task_id): float(i) * 0.01 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = tuple(W1FoldSanityDiag(i, 1.0, 1.0, 1.0) for i in range(3))
    fr = tuple(_fold_result_stub() for _ in range(3))
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr,
        fold_results_ch=None,
        cfg=cfg,
    )
    assert ok is False
    assert "distinct_query_target_values_below_min" in rs


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_fold_query_variance_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): float(i) * 0.03 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): float(i) * 0.01 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 0.0, 1.0, 1.0),)
    fr = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr,
        fold_results_ch=None,
        cfg=cfg,
    )
    assert ok is False
    assert any("fold_query_target_variance_below_min" in r for r in rs)


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_heterogeneous_windows_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_same_window_task("a", emb_scale=0.1), _same_window_task("b", emb_scale=0.2))
    tq = {str(t.task_id): float(i) * 0.05 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): float(i) * 0.02 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 1.0, 1.0, 1.0),)
    fr = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr,
        fold_results_ch=None,
        cfg=cfg,
    )
    assert ok is False
    assert "task_windows_not_heterogeneous" in rs


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_flat_incumbent_preds_per_fold_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): float(i) * 0.04 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): float(i) * 0.01 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 1.0, 0.0, 1.0),)
    fr = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr,
        fold_results_ch=None,
        cfg=cfg,
    )
    assert ok is False
    assert any("incumbent_pred_std_below_min" in r for r in rs)


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_flat_challenger_preds_per_fold_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): float(i) * 0.04 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): float(i) * 0.01 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 1.0, 1.0, 0.0),)
    fr_i = (_fold_result_stub(),)
    fr_c = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr_i,
        fold_results_ch=fr_c,
        cfg=cfg,
    )
    assert ok is False
    assert any("challenger_pred_std_below_min" in r for r in rs)


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_fold_metrics_non_degenerate_fail(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): float(i) * 0.04 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): float(i) * 0.01 for i, t in enumerate(tasks)}
    cfg = replace(
        W1EvidenceSanityConfig(),
        require_fold_metrics_non_degenerate=True,
        min_abs_gross_ic=0.5,
    )
    di = (W1FoldSanityDiag(0, 1.0, 1.0, 1.0),)
    fr_i = (_fold_result_stub(gross_ic=0.0, net_sharpe=0.0, turnover=0.0),)
    fr_c = (_fold_result_stub(gross_ic=0.0, net_sharpe=0.0, turnover=0.0),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr_i,
        fold_results_ch=fr_c,
        cfg=cfg,
    )
    assert ok is False
    assert "incumbent_fold_metrics_all_degenerate" in rs
    assert "challenger_fold_metrics_all_degenerate" in rs


@pytest.mark.determinism("d1")
def test_derive_w1_splits_fingerprint_includes_evidence_sanity_config(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    base = W1BaselineConfig()
    fp0 = derive_w1_splits_fingerprint(cfg=base)
    fp1 = derive_w1_splits_fingerprint(
        cfg=replace(base, evidence_sanity=W1EvidenceSanityConfig(min_std_query_targets=0.5)),
    )
    assert fp0 != fp1
    assert fp0.startswith("sha256:")
    assert fp1.startswith("sha256:")


@pytest.mark.determinism("d1")
def test_derive_w1_splits_fingerprint_includes_evidence_sanity_mode(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    base = W1BaselineConfig()
    fp0 = derive_w1_splits_fingerprint(cfg=base)
    fp_fixture = derive_w1_splits_fingerprint(
        cfg=replace(base, evidence_sanity_mode="fixture_smoke")
    )
    assert fp0 != fp_fixture


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_config_rejects_non_positive_closure_floors(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    with pytest.raises(W1BaselineConfigError):
        W1EvidenceSanityConfig(min_std_query_targets=0.0)


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_degenerate_constant_surface_rejected_by_defaults(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): 1.0 for t in tasks}
    ts = {str(t.task_id): 2.0 for t in tasks}
    cfg = W1EvidenceSanityConfig()
    di = (W1FoldSanityDiag(0, 0.0, 0.0, 0.0),)
    fr_i = (_fold_result_stub(),)
    fr_c = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr_i,
        fold_results_ch=fr_c,
        cfg=cfg,
    )
    assert ok is False
    assert rs


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_fixture_smoke_profile_allows_degenerate(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1))
    tq = {str(t.task_id): 1.0 for t in tasks}
    ts = {str(t.task_id): 2.0 for t in tasks}
    cfg = W1EvidenceSanityConfig.fixture_smoke()
    di = (W1FoldSanityDiag(0, 0.0, 0.0, 0.0),)
    fr_i = (_fold_result_stub(),)
    fr_c = (_fold_result_stub(),)
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr_i,
        fold_results_ch=fr_c,
        cfg=cfg,
    )
    assert ok is True
    assert rs == ()


@pytest.mark.determinism("d1")
def test_w1_evidence_sanity_positive_path(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = (_bull_only_task(0), _bull_only_task(1), _bull_only_task(2))
    tq = {str(t.task_id): float(i) * 0.07 for i, t in enumerate(tasks)}
    ts = {str(t.task_id): float(i) * 0.03 + 0.1 for i, t in enumerate(tasks)}
    cfg = W1EvidenceSanityConfig()
    di = tuple(W1FoldSanityDiag(i, 0.5, 0.5, 0.5) for i in range(3))
    fr_i = tuple(_fold_result_stub(gross_ic=0.02, net_sharpe=0.5, turnover=0.05) for _ in range(3))
    fr_c = tuple(_fold_result_stub(gross_ic=0.01, net_sharpe=0.4, turnover=0.04) for _ in range(3))
    ok, rs = w1_evidence_sanity_closure_ok(
        tasks=tasks,
        task_query_targets=tq,
        task_support_targets=ts,
        fold_diags=di,
        fold_results_inc=fr_i,
        fold_results_ch=fr_c,
        cfg=cfg,
    )
    assert ok is True
    assert rs == ()

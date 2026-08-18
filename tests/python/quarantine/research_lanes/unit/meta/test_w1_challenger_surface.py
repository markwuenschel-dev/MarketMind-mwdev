"""Unit tests for :mod:`pysrc.meta.w1_challenger_surface`."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.w1_baseline_config import W1BaselineConfig
from pysrc.meta.w1_challenger_surface import (
    W1_CHALLENGER_SURFACE_SCHEMA_VERSION,
    W1ChallengerPrediction,
    W1ChallengerSurface,
    W1ChallengerSurfaceValidationError,
    challenger_mean_query_scores_for_tasks,
    derive_w1_fold_plan,
    valid_learned_model_state_hash,
    validate_w1_challenger_surface,
    w1_challenger_surface_closure_eligible,
)
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(i: int, *, n_query: int = 3) -> MetaTask:
    day = datetime(2024, 7, 1, tzinfo=UTC) + timedelta(days=i * 7)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(6))
    query = tuple((day + timedelta(days=20 + j)).isoformat() for j in range(n_query))
    emb = np.full(4, 0.01 * float(i), dtype=np.float32)
    return MetaTask(
        task_id=f"w1-t-{i:05d}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=REGIME_CLASS_ORDER[i % len(REGIME_CLASS_ORDER)],
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


def _surface_for_pool(
    tasks: tuple[MetaTask, ...],
    *,
    fold_plan,
    wrong_fp: bool = False,
    wrong_leak: bool = False,
) -> W1ChallengerSurface:
    st = sorted(tasks, key=lambda t: (t.t0, t.task_id))
    preds: list[W1ChallengerPrediction] = []
    for fi in range(fold_plan.n_walk_forward_folds):
        for tid in fold_plan.task_ids_by_fold[fi]:
            t = next(x for x in st if x.task_id == tid)
            scores = tuple(0.01 * float(k + 1) for k in range(len(t.query_set)))
            preds.append(
                W1ChallengerPrediction(
                    task_id=t.task_id,
                    fold_index=fi,
                    query_scores=scores,
                    prediction_time_utc="2026-01-01T00:00:00Z",
                    pit_boundary=str(t.pit_boundary),
                ),
            )
    df = "sha256:" + ("b" if wrong_fp else "a") * 64
    return W1ChallengerSurface(
        schema_version=W1_CHALLENGER_SURFACE_SCHEMA_VERSION,
        source="test_fixture_allocator.v1",
        model_family="fixture",
        predictions=tuple(preds),
        task_pool_hash="sha256:" + "c" * 64,
        data_fingerprint=df,
        splits_fingerprint="sha256:" + "d" * 64,
        cost_assumptions_fingerprint="sha256:" + "e" * 64,
        signal_set_version="1",
        created_at_utc="2026-01-01T00:00:00Z",
        leakage_policy="pit_support_only_no_query_labels",
        uses_query_labels=wrong_leak,
        uses_xgboost_outputs=False,
        uses_post_query_metrics=False,
    )


@pytest.mark.determinism("d1")
def test_challenger_surface_validates_task_pool_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="task_pool_hash"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "f" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_validates_data_fingerprint(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan, wrong_fp=True)
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="data_fingerprint"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_validates_splits_fingerprint(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="splits_fingerprint"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "0" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_validates_cost_fingerprint(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="cost_assumptions_fingerprint"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "0" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_rejects_query_label_usage(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan, wrong_leak=True)
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="leakage"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_rejects_xgboost_output_usage(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    s0 = _surface_for_pool(pool, fold_plan=fold_plan)
    surface = W1ChallengerSurface(
        schema_version=s0.schema_version,
        source=s0.source,
        model_family=s0.model_family,
        predictions=s0.predictions,
        task_pool_hash=s0.task_pool_hash,
        data_fingerprint=s0.data_fingerprint,
        splits_fingerprint=s0.splits_fingerprint,
        cost_assumptions_fingerprint=s0.cost_assumptions_fingerprint,
        signal_set_version=s0.signal_set_version,
        created_at_utc=s0.created_at_utc,
        leakage_policy=s0.leakage_policy,
        uses_query_labels=False,
        uses_xgboost_outputs=True,
        uses_post_query_metrics=False,
    )
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="leakage"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_rejects_post_query_metrics_usage(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    s0 = _surface_for_pool(pool, fold_plan=fold_plan)
    surface = W1ChallengerSurface(
        schema_version=s0.schema_version,
        source=s0.source,
        model_family=s0.model_family,
        predictions=s0.predictions,
        task_pool_hash=s0.task_pool_hash,
        data_fingerprint=s0.data_fingerprint,
        splits_fingerprint=s0.splits_fingerprint,
        cost_assumptions_fingerprint=s0.cost_assumptions_fingerprint,
        signal_set_version=s0.signal_set_version,
        created_at_utc=s0.created_at_utc,
        leakage_policy=s0.leakage_policy,
        uses_query_labels=False,
        uses_xgboost_outputs=False,
        uses_post_query_metrics=True,
    )
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="leakage"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_rejects_missing_task(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    preds = tuple(
        p
        for p in surface.predictions
        if p.fold_index != 0 or p.task_id != fold_plan.task_ids_by_fold[0][0]
    )
    surface2 = W1ChallengerSurface(
        schema_version=surface.schema_version,
        source=surface.source,
        model_family=surface.model_family,
        predictions=preds,
        task_pool_hash=surface.task_pool_hash,
        data_fingerprint=surface.data_fingerprint,
        splits_fingerprint=surface.splits_fingerprint,
        cost_assumptions_fingerprint=surface.cost_assumptions_fingerprint,
        signal_set_version=surface.signal_set_version,
        created_at_utc=surface.created_at_utc,
        leakage_policy=surface.leakage_policy,
        uses_query_labels=False,
        uses_xgboost_outputs=False,
        uses_post_query_metrics=False,
    )
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="exactly cover"):
        validate_w1_challenger_surface(
            surface2,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_challenger_surface_rejects_wrong_query_length(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface0 = _surface_for_pool(pool, fold_plan=fold_plan)
    p0 = surface0.predictions[0]
    bad = W1ChallengerPrediction(
        task_id=p0.task_id,
        fold_index=p0.fold_index,
        query_scores=(0.1,),
        prediction_time_utc=p0.prediction_time_utc,
        pit_boundary=p0.pit_boundary,
    )
    preds = (bad,) + tuple(surface0.predictions[1:])
    surface2 = W1ChallengerSurface(
        schema_version=surface0.schema_version,
        source=surface0.source,
        model_family=surface0.model_family,
        predictions=preds,
        task_pool_hash=surface0.task_pool_hash,
        data_fingerprint=surface0.data_fingerprint,
        splits_fingerprint=surface0.splits_fingerprint,
        cost_assumptions_fingerprint=surface0.cost_assumptions_fingerprint,
        signal_set_version=surface0.signal_set_version,
        created_at_utc=surface0.created_at_utc,
        leakage_policy=surface0.leakage_policy,
        uses_query_labels=False,
        uses_xgboost_outputs=False,
        uses_post_query_metrics=False,
    )
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="query_scores length"):
        validate_w1_challenger_surface(
            surface2,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_validate_w1_challenger_surface_ok(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    validate_w1_challenger_surface(
        surface,
        expected_task_ids_by_fold=exp,
        expected_query_lengths=ql,
        task_pool_hash="sha256:" + "c" * 64,
        data_fingerprint="sha256:" + "a" * 64,
        splits_fingerprint="sha256:" + "d" * 64,
        cost_assumptions_fingerprint="sha256:" + "e" * 64,
        signal_set_version="1",
    )


@pytest.mark.determinism("d1")
def test_challenger_mean_query_scores_for_tasks(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=1, fold_length_bars=2, warmup_bars=1)
    pool = tuple(_task(i) for i in range(3))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    st = sorted(pool, key=lambda t: (t.t0, t.task_id))
    chunk_ids = fold_plan.task_ids_by_fold[0]
    chunk = tuple(next(t for t in st if t.task_id == tid) for tid in chunk_ids)
    arr = challenger_mean_query_scores_for_tasks(surface, chunk, 0)
    assert arr.shape == (len(chunk),)


@pytest.mark.determinism("d1")
def test_validate_rejects_invalid_model_state_hash_format(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = replace(_surface_for_pool(pool, fold_plan=fold_plan), model_state_hash="not-sha256")
    exp = {fi: fold_plan.task_ids_by_fold[fi] for fi in range(2)}
    ql = {t.task_id: len(t.query_set) for t in sorted(pool, key=lambda t: (t.t0, t.task_id))}
    with pytest.raises(W1ChallengerSurfaceValidationError, match="model_state_hash"):
        validate_w1_challenger_surface(
            surface,
            expected_task_ids_by_fold=exp,
            expected_query_lengths=ql,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
        )


@pytest.mark.determinism("d1")
def test_closure_ineligible_without_model_state_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = _surface_for_pool(pool, fold_plan=fold_plan)
    assert not valid_learned_model_state_hash(surface.model_state_hash)
    assert not w1_challenger_surface_closure_eligible(surface)


@pytest.mark.determinism("d1")
def test_closure_ineligible_when_model_family_banned(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = replace(
        _surface_for_pool(pool, fold_plan=fold_plan),
        model_state_hash="sha256:" + "a" * 64,
        source="reptile_learned_w1.v1",
    )
    assert valid_learned_model_state_hash(surface.model_state_hash)
    assert not w1_challenger_surface_closure_eligible(surface)


@pytest.mark.determinism("d1")
def test_closure_eligible_clean_learned_provenance(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = replace(
        _surface_for_pool(pool, fold_plan=fold_plan),
        model_state_hash="sha256:" + "a" * 64,
        source="reptile_learned_w1.v1",
        model_family="reptile_meta_allocator_checkpoint",
        governed_checkpoint_lineage_verified=True,
    )
    assert w1_challenger_surface_closure_eligible(surface)

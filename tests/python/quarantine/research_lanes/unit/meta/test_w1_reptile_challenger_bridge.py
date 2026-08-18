"""Unit tests for :mod:`pysrc.meta.w1_reptile_challenger_bridge`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.w1_baseline_config import W1BaselineConfig
from pysrc.meta.w1_challenger_surface import (
    derive_w1_fold_plan,
    w1_challenger_surface_closure_eligible,
)
from pysrc.meta.w1_fold_safe_meta_allocator_adapter import W1FoldSafeSupportOnlyMetaAllocator
from pysrc.meta.w1_reptile_challenger_bridge import (
    W1ChallengerUnavailableError,
    build_reptile_w1_challenger_surface,
)
from pysrc.meta.w1_reptile_trained_meta_allocator_adapter import ReptileTrainedMetaAllocatorAdapter
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _fixture_ckpt_expectations() -> dict[str, str]:
    return {
        "expected_signal_set_version": "1",
        "expected_training_task_pool_hash": "sha256:" + "c" * 64,
        "expected_training_data_fingerprint": "sha256:" + "a" * 64,
        "expected_training_splits_fingerprint": "sha256:" + "d" * 64,
    }


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(i: int) -> MetaTask:
    day = datetime(2024, 7, 1, tzinfo=UTC) + timedelta(days=i * 7)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(6))
    query = tuple((day + timedelta(days=20 + j)).isoformat() for j in range(3))
    emb = np.full(4, 0.01 * float(i), dtype=np.float32)
    return MetaTask(
        task_id=f"w1-br-{i:05d}",
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


class _NoMethod:
    pass


class _ConstScores:
    def predict_query_scores(self, task: MetaTask, *, fold_index: int) -> tuple[float, ...]:
        return tuple(0.01 for _ in task.query_set)


@pytest.mark.determinism("d1")
def test_build_reptile_surface_requires_predict(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    with pytest.raises(W1ChallengerUnavailableError, match="predict_query_scores"):
        build_reptile_w1_challenger_surface(
            task_pool=pool,
            fold_plan=fold_plan,
            trained_state=_NoMethod(),
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
            created_at_utc="2026-01-02T00:00:00Z",
        )


@pytest.mark.determinism("d1")
def test_build_reptile_w1_challenger_surface_smoke(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    out = build_reptile_w1_challenger_surface(
        task_pool=pool,
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint="sha256:" + "c" * 64),
        task_pool_hash="sha256:" + "c" * 64,
        data_fingerprint="sha256:" + "a" * 64,
        splits_fingerprint="sha256:" + "d" * 64,
        cost_assumptions_fingerprint="sha256:" + "e" * 64,
        signal_set_version="1",
        created_at_utc="2026-01-02T00:00:00Z",
    )
    assert out.schema_version == "w1_challenger_surface.v1"
    assert len(out.predictions) == 2 * len(fold_plan.task_ids_by_fold[0])
    assert out.model_state_hash is None
    assert out.governed_checkpoint_lineage_verified is False
    assert "support_only" in out.source


@pytest.mark.determinism("d1")
def test_build_reptile_generic_learned_requires_model_state_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    with pytest.raises(W1ChallengerUnavailableError, match="model_state_hash"):
        build_reptile_w1_challenger_surface(
            task_pool=pool,
            fold_plan=fold_plan,
            trained_state=_ConstScores(),
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
            created_at_utc="2026-01-02T00:00:00Z",
        )


@pytest.mark.determinism("d1")
def test_build_reptile_generic_learned_with_model_state_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    msh = "sha256:" + "b" * 64
    out = build_reptile_w1_challenger_surface(
        task_pool=pool,
        fold_plan=fold_plan,
        trained_state=_ConstScores(),
        task_pool_hash="sha256:" + "c" * 64,
        data_fingerprint="sha256:" + "a" * 64,
        splits_fingerprint="sha256:" + "d" * 64,
        cost_assumptions_fingerprint="sha256:" + "e" * 64,
        signal_set_version="1",
        created_at_utc="2026-01-02T00:00:00Z",
        model_state_hash=msh,
        source="reptile_learned_custom.v1",
        model_family="reptile_meta_allocator",
    )
    assert out.model_state_hash == msh
    assert out.governed_checkpoint_lineage_verified is False
    assert not w1_challenger_surface_closure_eligible(out)


@pytest.mark.determinism("d1")
def test_reptile_trained_checkpoint_model_state_hash_mismatch_raises(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    src = (
        Path(__file__).resolve().parents[3] / "fixtures" / "w1" / "minimal_learned_checkpoint.json"
    )
    trained = ReptileTrainedMetaAllocatorAdapter(src, **_fixture_ckpt_expectations())
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    with pytest.raises(W1ChallengerUnavailableError, match="match"):
        build_reptile_w1_challenger_surface(
            task_pool=pool,
            fold_plan=fold_plan,
            trained_state=trained,
            task_pool_hash="sha256:" + "c" * 64,
            data_fingerprint="sha256:" + "a" * 64,
            splits_fingerprint="sha256:" + "d" * 64,
            cost_assumptions_fingerprint="sha256:" + "e" * 64,
            signal_set_version="1",
            created_at_utc="2026-01-02T00:00:00Z",
            model_state_hash="sha256:" + "c" * 64,
        )


@pytest.mark.determinism("d1")
def test_reptile_trained_missing_checkpoint_raises(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    p = tmp_path / "missing.json"
    with pytest.raises(W1ChallengerUnavailableError, match="readable"):
        ReptileTrainedMetaAllocatorAdapter(p, **_fixture_ckpt_expectations())


@pytest.mark.determinism("d0")
def test_fold_safe_meta_allocator_deterministic(deterministic_seed: int) -> None:
    _ = deterministic_seed
    t = _task(0)
    alloc = W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint="sha256:" + "c" * 64)
    a = alloc.predict_query_scores(t, fold_index=1)
    b = alloc.predict_query_scores(t, fold_index=1)
    assert a == b

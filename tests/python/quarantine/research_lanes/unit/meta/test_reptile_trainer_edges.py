from __future__ import annotations

import importlib
from collections import Counter
from datetime import UTC, datetime, timedelta
from unittest import mock

import numpy as np
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.curriculum import CurriculumBatch, CurriculumSampler, CurriculumSamplerConfig
from pysrc.meta.reptile_trainer import (
    ReptileTrainer,
    _as_theta,
    _build_design_matrix,
    _inner_adapt,
    _pairwise_ranking_loss,
    _rankdata,
    _spearman_proxy,
    reptile_outer_update,
)
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask_single() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(
    regime_class: str,
    i: int,
    *,
    active_k: int = 1,
    mask: tuple[bool, ...] | None = None,
    regime_embedding: np.ndarray | None = None,
) -> MetaTask:
    mask = mask or _mask_single()
    day = datetime(2024, 3, 1, tzinfo=UTC) + timedelta(days=i * 5)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(5))
    query = tuple((day + timedelta(days=10 + j)).isoformat() for j in range(3))
    return MetaTask(
        task_id=f"edge-{regime_class}-{i}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0=support[0],
        t1=(day + timedelta(days=25)).isoformat(),
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=mask,
        signal_set_version="1",
        signal_ids_hash="sha256:edge",
        horizon=1,
        active_k=active_k,
        regime_embedding=regime_embedding,
    )


def _batch(tasks: tuple[MetaTask, ...]) -> CurriculumBatch:
    counts = Counter(t.regime_class for t in tasks)
    bucket_counts = {b: int(counts.get(b, 0)) for b in REGIME_CLASS_ORDER}
    return CurriculumBatch(
        tasks=tasks,
        phase="bootstrap",
        bucket_counts=bucket_counts,
        priority_alpha=0.6,
        importance_beta=0.4,
        importance_weights=(),
    )


def test_as_theta_wrong_shape() -> None:
    with pytest.raises(DataPreconditionError, match="shape"):
        _as_theta("t", np.zeros(3, dtype=np.float32))


def test_as_theta_non_finite() -> None:
    with pytest.raises(DataPreconditionError, match="finite"):
        _as_theta("t", np.full(MAX_SIGNALS, np.nan, dtype=np.float32))


def test_build_design_matrix_active_k_mismatch() -> None:
    bad_mask = tuple(i in (0, 1) for i in range(MAX_SIGNALS))
    day = datetime(2024, 3, 1, tzinfo=UTC)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(5))

    class _BadTask:
        task_id = "bad-active-k"
        signal_mask = bad_mask
        active_k = 1
        support_set = support
        regime_embedding = None

    rng = np.random.default_rng(0)
    with pytest.raises(DataPreconditionError, match="signal_mask"):
        _build_design_matrix(_BadTask(), support, rng=rng)  # type: ignore[arg-type]


def test_pairwise_ranking_loss_large_positive_diff() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y = np.array([1.0, 0.0], dtype=np.float32)
    w = np.array([80.0, 0.0], dtype=np.float64)
    loss = _pairwise_ranking_loss(X, y, w)
    assert np.isfinite(loss)


def test_pairwise_ranking_loss_large_negative_diff() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y = np.array([1.0, 0.0], dtype=np.float32)
    w = np.array([-80.0, 0.0], dtype=np.float64)
    loss = _pairwise_ranking_loss(X, y, w)
    assert np.isfinite(loss)


def test_rankdata_all_ties() -> None:
    r = _rankdata(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.allclose(r, np.array([1.0, 1.0, 1.0]))


def test_spearman_too_short() -> None:
    assert np.isnan(
        _spearman_proxy(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32))
    )


def test_spearman_zero_denom() -> None:
    pred = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    assert np.isnan(_spearman_proxy(pred, y))


def test_inner_adapt_value_error_nonfinite_propagates() -> None:
    t = _task("crisis", 0, regime_embedding=np.zeros(2, dtype=np.float32))
    cfg = ReptileTrainerConfig(K=2)
    rng = np.random.default_rng(0)
    with (
        mock.patch(
            "pysrc.meta.reptile_trainer._pairwise_ranking_loss_grad",
            return_value=np.full(3, np.nan, dtype=np.float64),
        ),
        pytest.raises(ValueError, match="inner_nonfinite"),
    ):
        _inner_adapt(t, np.zeros(MAX_SIGNALS, dtype=np.float32), config=cfg, rng=rng)


def test_curriculum_sample_failed_emits_fail_report() -> None:
    class _Bad:
        @property
        def trainable_tasks(self) -> tuple[MetaTask, ...]:
            return ()

        @property
        def bucket_governance_minimums(self) -> dict[str, int]:
            return dict.fromkeys(REGIME_CLASS_ORDER, 0)

        def sample_bootstrap(self) -> CurriculumBatch:
            raise RuntimeError("sampler down")

    tr = ReptileTrainer(ReptileTrainerConfig(), _Bad(), seed=0)
    res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=0)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "CURRICULUM_SAMPLE_FAILED" in res.meta_validity_report["fail_reasons"][0]


def test_inner_adapt_generic_exception_counts_consecutive() -> None:
    tasks = tuple(
        _task("crisis", i, regime_embedding=np.zeros(2, dtype=np.float32)) for i in range(4)
    )
    tr = ReptileTrainer(
        ReptileTrainerConfig(task_failure_abort_threshold=3), _FakeSampler(tasks), seed=1
    )
    with mock.patch(
        "pysrc.meta.reptile_trainer._inner_adapt",
        side_effect=RuntimeError("boom"),
    ):
        res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=1)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "CONSECUTIVE_TASK_FAILURES" in res.meta_validity_report["fail_reasons"]


def test_nonfinite_theta_task_output_increments_failures() -> None:
    tasks = tuple(
        _task("crisis", i, regime_embedding=np.zeros(2, dtype=np.float32)) for i in range(4)
    )

    def _bad_inner(*_a, **_k):
        nan_vec = np.full(MAX_SIGNALS, np.nan, dtype=np.float32)
        return nan_vec, [], 1.0, float("nan")

    tr = ReptileTrainer(
        ReptileTrainerConfig(task_failure_abort_threshold=3), _FakeSampler(tasks), seed=2
    )
    with mock.patch("pysrc.meta.reptile_trainer._inner_adapt", side_effect=_bad_inner):
        res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=2)
    assert res.meta_validity_report["overall_result"] == "FAIL"


def test_no_successful_adaptations() -> None:
    tasks = tuple(
        _task("crisis", i, regime_embedding=np.zeros(2, dtype=np.float32)) for i in range(4)
    )
    tr = ReptileTrainer(
        ReptileTrainerConfig(task_failure_abort_threshold=99), _FakeSampler(tasks), seed=3
    )

    def _skip(*_a, **_k):
        raise ValueError("inner_nonfinite")

    with mock.patch("pysrc.meta.reptile_trainer._inner_adapt", side_effect=_skip):
        res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=3)
    assert "NO_SUCCESSFUL_ADAPTATIONS" in res.meta_validity_report["fail_reasons"]


def test_theta_day_prime_build_failed() -> None:
    tasks = tuple(
        _task("crisis", i, regime_embedding=np.zeros(2, dtype=np.float32)) for i in range(3)
    )
    seq: list = []
    for _ in tasks:
        z = np.zeros(MAX_SIGNALS, dtype=np.float32)
        z[0] = 0.1
        seq.append((z, [], 0.1, 0.2))
    for _ in tasks:
        seq.append(RuntimeError("day phase fail"))

    tr = ReptileTrainer(ReptileTrainerConfig(), _FakeSampler(tasks), seed=4)
    with mock.patch("pysrc.meta.reptile_trainer._inner_adapt", side_effect=seq):
        res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=4)
    assert "THETA_DAY_PRIME_BUILD_FAILED" in res.meta_validity_report["fail_reasons"]


def test_nan_meta_after_outer_causes_day_prime_build_fail() -> None:
    tasks = []
    for i, b in enumerate(REGIME_CLASS_ORDER):
        tasks.append(_task(b, i, regime_embedding=np.linspace(0, 1, 3, dtype=np.float32)))
    for j in range(8):
        tasks.append(
            _task("crisis", 50 + j, regime_embedding=np.linspace(0, 1, 3, dtype=np.float32))
        )
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    samp = CurriculumSampler(
        tasks,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks), crisis_floor_fraction=0.10, bucket_minimums=mins, seed=5
        ),
    )
    tr = ReptileTrainer(ReptileTrainerConfig(), samp, seed=5)

    def _fake_ewc(self, theta, *, lambda_ewc):
        return np.full(MAX_SIGNALS, np.nan, dtype=np.float32)

    with mock.patch.object(ReptileTrainer, "apply_ewc_seam", _fake_ewc):
        res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=5)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "THETA_DAY_PRIME_BUILD_FAILED" in res.meta_validity_report["fail_reasons"]


def test_anil_wrong_encoder_type() -> None:
    pytest.importorskip("torch")
    importlib.import_module("pysrc.meta_learning.context_encoder")

    class _NotEncoder:
        def is_frozen(self) -> bool:
            return True

    tasks = (_task("crisis", 0),)
    tr = ReptileTrainer(
        ReptileTrainerConfig(algorithm="anil"),
        _FakeSampler(tasks),
        seed=6,
        context_encoder=_NotEncoder(),
    )
    with pytest.raises(DataPreconditionError, match="ContextEncoder"):
        tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=6)


def test_query_ic_nonfinite_skips_ic_lists_but_keeps_adaptation() -> None:
    tasks = []
    for i, b in enumerate(REGIME_CLASS_ORDER):
        tasks.append(_task(b, i, regime_embedding=np.linspace(0, 1, 3, dtype=np.float32)))
    for j in range(8):
        tasks.append(
            _task("crisis", 50 + j, regime_embedding=np.linspace(0, 1, 3, dtype=np.float32))
        )

    def _nan_ic(*_a, **_k):
        z = np.zeros(MAX_SIGNALS, dtype=np.float32)
        z[0] = 0.5
        return z, [], 0.1, float("nan")

    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    samp = CurriculumSampler(
        tasks,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks), crisis_floor_fraction=0.10, bucket_minimums=mins, seed=11
        ),
    )
    tr = ReptileTrainer(ReptileTrainerConfig(task_failure_abort_threshold=20), samp, seed=11)
    with mock.patch("pysrc.meta.reptile_trainer._inner_adapt", side_effect=_nan_ic):
        res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=11)
    assert res.meta_validity_report["overall_result"] == "PASS"
    assert res.meta_validity_report["inner_loop_gain"]["mean_query_ic"] is None
    assert res.meta_validity_report["inner_loop_gain"]["by_regime_class"] is None


def test_trainer_stores_last_promoted_theta_vector() -> None:
    tasks = []
    for i, b in enumerate(REGIME_CLASS_ORDER):
        tasks.append(_task(b, i, regime_embedding=np.linspace(0, 1, 3, dtype=np.float32)))
    for j in range(10):
        tasks.append(
            _task(
                "crisis" if j % 2 == 0 else "bull",
                100 + j,
                regime_embedding=np.linspace(0, 1, 3, dtype=np.float32),
            )
        )
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    samp = CurriculumSampler(
        tasks,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks), crisis_floor_fraction=0.10, bucket_minimums=mins, seed=7
        ),
    )
    tr = ReptileTrainer(ReptileTrainerConfig(task_failure_abort_threshold=10), samp, seed=7)
    tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=7)
    v1 = tr.theta_day_prime_promoted
    assert v1 is not None
    tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=7)
    v2 = tr.theta_day_prime_promoted
    assert v2 is not None
    assert not np.shares_memory(v1, v2)


def test_inner_value_error_non_inner_raises() -> None:
    tasks = (_task("crisis", 0, regime_embedding=np.zeros(2, dtype=np.float32)),)
    tr = ReptileTrainer(ReptileTrainerConfig(), _FakeSampler(tasks), seed=8)
    with mock.patch("pysrc.meta.reptile_trainer._inner_adapt", side_effect=ValueError("other")):
        with pytest.raises(ValueError, match="other"):
            tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=8)


def test_reptile_outer_update_preserves_exact_formula_edge() -> None:
    theta = np.ones(MAX_SIGNALS, dtype=np.float32) * 3.0
    adapted = np.stack([theta + 2.0, theta - 1.0], axis=0).astype(np.float32)
    out = reptile_outer_update(theta, adapted, outer_lr=0.25)
    mean_delta = np.mean(adapted - theta.reshape(1, -1), axis=0)
    np.testing.assert_allclose(
        out, theta + np.float32(0.25) * mean_delta.astype(np.float32), rtol=0, atol=1e-5
    )


class _FakeSampler:
    def __init__(self, tasks: tuple[MetaTask, ...]) -> None:
        self._tasks = tasks

    @property
    def trainable_tasks(self) -> tuple[MetaTask, ...]:
        return self._tasks

    @property
    def bucket_governance_minimums(self) -> dict[str, int]:
        c = Counter(t.regime_class for t in self._tasks)
        return {b: 1 if c.get(b, 0) > 0 else 0 for b in REGIME_CLASS_ORDER}

    def sample_bootstrap(self) -> CurriculumBatch:
        return _batch(self._tasks)

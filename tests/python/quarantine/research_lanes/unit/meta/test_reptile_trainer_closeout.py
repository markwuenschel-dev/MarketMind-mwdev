"""MLC-3 closeout: task-pool sufficiency gate and D0 adapted-parameter replay."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np

from pysrc.meta.curriculum import CurriculumBatch, CurriculumSampler, CurriculumSamplerConfig
from pysrc.meta.reptile_trainer import ReptileTrainer, collect_inner_adapted_thetas_d0
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(regime_class: str, i: int) -> MetaTask:
    day = datetime(2024, 8, 1, tzinfo=UTC) + timedelta(days=i * 3)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(5))
    query = tuple((day + timedelta(days=10 + j)).isoformat() for j in range(3))
    emb = np.zeros(2, dtype=np.float32)
    return MetaTask(
        task_id=f"close-{regime_class}-{i}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0=support[0],
        t1=(day + timedelta(days=25)).isoformat(),
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=_mask(),
        signal_set_version="1",
        signal_ids_hash="sha256:close",
        horizon=1,
        active_k=1,
        regime_embedding=emb,
    )


def _batch(tasks: tuple[MetaTask, ...]) -> CurriculumBatch:
    c = Counter(t.regime_class for t in tasks)
    bc = {b: int(c.get(b, 0)) for b in REGIME_CLASS_ORDER}
    return CurriculumBatch(
        tasks=tasks,
        phase="bootstrap",
        bucket_counts=bc,
        priority_alpha=0.6,
        importance_beta=0.4,
        importance_weights=(),
    )


class _GovernanceFailSampler:
    """Trainable pool is sparse but governance demands v2-scale floors."""

    def __init__(self, tasks: tuple[MetaTask, ...]) -> None:
        self._tasks = tasks

    @property
    def trainable_tasks(self) -> tuple[MetaTask, ...]:
        return self._tasks

    @property
    def bucket_governance_minimums(self) -> dict[str, int]:
        return dict.fromkeys(REGIME_CLASS_ORDER, 50)

    def sample_bootstrap(self) -> CurriculumBatch:
        return _batch(self._tasks)


def test_task_pool_insufficient_governance_fail_closed() -> None:
    tasks = tuple(_task("bull", i) for i in range(4))
    tr = ReptileTrainer(ReptileTrainerConfig(), _GovernanceFailSampler(tasks), seed=0)
    res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=0)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert any(
        "TASK_POOL_INSUFFICIENT_GOVERNANCE" in r for r in res.meta_validity_report["fail_reasons"]
    )


def test_d0_inner_adapted_vectors_identical_replays() -> None:
    pool = [_task(b, i) for i, b in enumerate(REGIME_CLASS_ORDER)]
    for j in range(10):
        pool.append(_task("crisis" if j % 2 == 0 else "bull", 50 + j))
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    samp = CurriculumSampler(
        pool,
        config=CurriculumSamplerConfig(
            batch_size=8, crisis_floor_fraction=0.10, bucket_minimums=mins, seed=777
        ),
    )
    batch = samp.sample_bootstrap()
    cfg = ReptileTrainerConfig(task_failure_abort_threshold=10)
    theta = np.random.default_rng(12345).standard_normal(MAX_SIGNALS).astype(
        np.float32
    ) * np.float32(0.01)
    a = collect_inner_adapted_thetas_d0(theta_meta=theta, tasks=batch.tasks, config=cfg, seed=999)
    b = collect_inner_adapted_thetas_d0(theta_meta=theta, tasks=batch.tasks, config=cfg, seed=999)
    assert len(a) == len(b) == len(batch.tasks)
    for i in range(len(a)):
        np.testing.assert_array_equal(a[i], b[i])


def test_curriculum_sampler_exposes_governance_minimums() -> None:
    pool: list[MetaTask] = []
    for b in REGIME_CLASS_ORDER:
        pool.append(_task(b, 0))
        pool.append(_task(b, 1))
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 2)
    samp = CurriculumSampler(
        pool,
        config=CurriculumSamplerConfig(
            batch_size=5, crisis_floor_fraction=0.10, bucket_minimums=mins, seed=1
        ),
    )
    assert dict(samp.bucket_governance_minimums) == mins

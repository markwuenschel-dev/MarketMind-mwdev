from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from pysrc.meta.curriculum import (
    CurriculumSampler,
    CurriculumSamplerConfig,
    CurriculumSufficiencyError,
    HoldoutExclusionSurface,
)
from pysrc.meta.task import MAX_SIGNALS, MetaTask


def _ids() -> tuple[str, ...]:
    ids = [""] * MAX_SIGNALS
    ids[0] = "sig_a"
    return tuple(ids)


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(regime_class: str, i: int) -> MetaTask:
    day = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 10)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(3))
    query = (day + timedelta(days=4)).isoformat()
    return MetaTask(
        task_id=f"{regime_class}-{i}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0=support[0],
        t1=(day + timedelta(days=30)).isoformat(),
        pit_boundary=support[-1],
        support_set=support,
        query_set=(query,),
        signal_ids=_ids(),
        signal_mask=_mask(),
        signal_set_version="1",
        signal_ids_hash="sha256:abc",
        horizon=1,
        active_k=1,
    )


def _pool(per_bucket: int = 5) -> list[MetaTask]:
    classes = ("bull", "bear", "sideways", "high_vol", "crisis")
    tasks: list[MetaTask] = []
    for bucket_i, bucket in enumerate(classes):
        for i in range(per_bucket):
            tasks.append(_task(bucket, bucket_i * per_bucket + i))
    return tasks


def _config(
    *,
    batch_size: int = 10,
    bucket_minimums: Mapping[str, int] | None = None,
    priority_alpha: float = 0.6,
    importance_beta: float = 0.4,
    seed: int = 17,
) -> CurriculumSamplerConfig:
    return CurriculumSamplerConfig(
        batch_size=batch_size,
        bucket_minimums={
            "bull": 1,
            "bear": 1,
            "sideways": 1,
            "high_vol": 1,
            "crisis": 1,
        }
        if bucket_minimums is None
        else bucket_minimums,
        priority_alpha=priority_alpha,
        importance_beta=importance_beta,
        seed=seed,
    )


@pytest.mark.determinism("d0")
def test_bootstrap_phase_covers_all_five_buckets(deterministic_seed: int) -> None:
    _ = deterministic_seed
    batch = CurriculumSampler(_pool(), config=_config()).sample_bootstrap()
    assert set(batch.bucket_counts) == {"bull", "bear", "sideways", "high_vol", "crisis"}
    assert all(batch.bucket_counts[b] >= 1 for b in batch.bucket_counts)


@pytest.mark.determinism("d0")
def test_crisis_floor_is_enforced_per_batch(deterministic_seed: int) -> None:
    _ = deterministic_seed
    batch = CurriculumSampler(_pool(), config=_config(batch_size=20)).sample_bootstrap()
    assert batch.bucket_counts["crisis"] >= 2


@pytest.mark.determinism("d0")
def test_held_out_task_ids_never_appear(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pool = _pool()
    holdout = pool[0].task_id
    sampler = CurriculumSampler(
        pool,
        config=_config(),
        holdouts=HoldoutExclusionSurface(task_ids=frozenset({holdout})),
    )
    for _ in range(10):
        assert holdout not in {task.task_id for task in sampler.sample_bootstrap().tasks}


@pytest.mark.determinism("d0")
def test_bucket_counts_are_logged_for_every_batch(
    caplog: pytest.LogCaptureFixture, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    sampler = CurriculumSampler(_pool(), config=_config())
    with caplog.at_level("INFO"):
        _ = sampler.sample_bootstrap()
    assert any("curriculum_batch_sampled" in rec.getMessage() for rec in caplog.records)


@pytest.mark.determinism("d0")
def test_below_minimum_bucket_fails_closed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = _config(
        bucket_minimums={"bull": 99, "bear": 1, "sideways": 1, "high_vol": 1, "crisis": 1}
    )
    with pytest.raises(CurriculumSufficiencyError, match="bucket minimum"):
        CurriculumSampler(_pool(), config=cfg)


@pytest.mark.determinism("d1")
def test_per_like_phase_prioritizes_higher_synthetic_priority(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pool = _pool(per_bucket=10)
    top_ids = {task.task_id for task in pool if task.task_id.endswith("-9")}
    priorities = {task.task_id: (10_000.0 if task.task_id in top_ids else 1.0) for task in pool}
    sampler = CurriculumSampler(pool, config=_config(batch_size=10, seed=29))
    hits = 0
    draws = 30
    for _ in range(draws):
        batch = sampler.sample_per_like(priority_resolver=priorities.__getitem__)
        hits += len(top_ids & {task.task_id for task in batch.tasks})
    assert hits >= draws


@pytest.mark.determinism("d0")
def test_phase_switching_does_not_corrupt_sampler_state(deterministic_seed: int) -> None:
    _ = deterministic_seed
    sampler = CurriculumSampler(_pool(), config=_config())
    a = sampler.sample_bootstrap()
    b = sampler.sample_per_like(priority_resolver=lambda task_id: 1.0)
    c = sampler.sample_bootstrap()
    assert len(a.tasks) == len(b.tasks) == len(c.tasks) == 10
    assert b.phase == "per_like"
    assert c.phase == "bootstrap"


@pytest.mark.determinism("d0")
def test_alpha_beta_come_from_config(deterministic_seed: int) -> None:
    _ = deterministic_seed
    sampler = CurriculumSampler(
        _pool(),
        config=_config(priority_alpha=0.9, importance_beta=0.7),
    )
    batch = sampler.sample_per_like(priority_resolver=lambda task_id: 2.0)
    assert batch.priority_alpha == 0.9
    assert batch.importance_beta == 0.7
    assert len(batch.importance_weights) == len(batch.tasks)


@pytest.mark.determinism("d0")
def test_default_bucket_minimums_and_trainable_tasks_property(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pool = _pool(per_bucket=51)
    holdouts = HoldoutExclusionSurface(task_ids=frozenset({pool[0].task_id}))
    sampler = CurriculumSampler(
        pool,
        config=CurriculumSamplerConfig(seed=31),
        holdouts=holdouts,
    )
    assert len(sampler.trainable_tasks) == len(pool) - 1
    assert pool[0].task_id not in {task.task_id for task in sampler.trainable_tasks}


@pytest.mark.determinism("d0")
def test_config_validation_fails_closed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    valid_minimums = {"bull": 0, "bear": 0, "sideways": 0, "high_vol": 0, "crisis": 0}

    configs = (
        (CurriculumSamplerConfig(batch_size=0, bucket_minimums=valid_minimums), "batch_size"),
        (
            CurriculumSamplerConfig(crisis_floor_fraction=1.1, bucket_minimums=valid_minimums),
            "crisis_floor",
        ),
        (CurriculumSamplerConfig(priority_alpha=-0.1, bucket_minimums=valid_minimums), "alpha"),
        (CurriculumSamplerConfig(bucket_minimums={"bull": 0}), "bucket_minimums"),
    )
    for config, message in configs:
        with pytest.raises(CurriculumSufficiencyError, match=message):
            CurriculumSampler(_pool(per_bucket=1), config=config)


@pytest.mark.determinism("d0")
def test_unknown_regime_class_fails_closed(deterministic_seed: int) -> None:
    _ = deterministic_seed

    class _Task:
        task_id = "unknown-1"
        regime_class = "unknown"

    with pytest.raises(CurriculumSufficiencyError, match="unknown regime_class"):
        CurriculumSampler._bucket_tasks((cast(Any, _Task()),))


@pytest.mark.determinism("d0")
def test_batch_size_equal_to_crisis_floor_returns_only_floor_tasks(deterministic_seed: int) -> None:
    _ = deterministic_seed
    sampler = CurriculumSampler(
        _pool(per_bucket=1),
        config=_config(
            batch_size=1,
            bucket_minimums={"bull": 0, "bear": 0, "sideways": 0, "high_vol": 0, "crisis": 1},
        ),
    )
    batch = sampler.sample_bootstrap()
    assert len(batch.tasks) == 1
    assert batch.bucket_counts["crisis"] == 1


@pytest.mark.determinism("d0")
def test_empty_sampling_surfaces_fail_closed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    minimums = {"bull": 0, "bear": 0, "sideways": 0, "high_vol": 0, "crisis": 0}

    with pytest.raises(CurriculumSufficiencyError, match="empty bucket"):
        CurriculumSampler(
            [task for task in _pool(per_bucket=1) if task.regime_class != "crisis"],
            config=CurriculumSamplerConfig(batch_size=2, bucket_minimums=minimums, seed=5),
        ).sample_bootstrap()

    with pytest.raises(CurriculumSufficiencyError, match="empty task pool"):
        CurriculumSampler(
            [],
            config=CurriculumSamplerConfig(
                batch_size=1, crisis_floor_fraction=0.0, bucket_minimums=minimums, seed=5
            ),
        ).sample_bootstrap()


@pytest.mark.determinism("d0")
def test_sample_from_pool_zero_count_returns_empty(deterministic_seed: int) -> None:
    _ = deterministic_seed
    sampler = CurriculumSampler(_pool(per_bucket=1), config=_config(batch_size=1))
    assert (
        sampler._sample_from_pool(_pool(per_bucket=1), 0, uniform=True, priority_resolver=None)
        == ()
    )


@pytest.mark.determinism("d0")
def test_per_like_priority_surface_validation(deterministic_seed: int) -> None:
    _ = deterministic_seed
    minimums = {"bull": 0, "bear": 0, "sideways": 0, "high_vol": 0, "crisis": 0}
    sampler = CurriculumSampler(
        _pool(per_bucket=1),
        config=CurriculumSamplerConfig(batch_size=5, bucket_minimums=minimums, seed=13),
    )

    with pytest.raises(CurriculumSufficiencyError, match="priority_resolver"):
        sampler._sample_from_pool(sampler.trainable_tasks, 1, uniform=False, priority_resolver=None)

    with pytest.raises(CurriculumSufficiencyError, match="non-finite"):
        sampler.sample_per_like(priority_resolver=lambda task_id: float("nan"))

    batch = sampler.sample_per_like(priority_resolver=lambda task_id: -10.0)
    assert len(batch.tasks) == 5
    assert len(batch.importance_weights) == 5

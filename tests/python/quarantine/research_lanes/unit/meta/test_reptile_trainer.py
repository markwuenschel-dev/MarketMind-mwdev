from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from unittest import mock

import numpy as np
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.curriculum import CurriculumBatch, CurriculumSampler, CurriculumSamplerConfig
from pysrc.meta.reptile_trainer import (
    GovernedCurriculumFailure,
    ReptileTrainer,
    prepare_governed_curriculum_batch,
    reptile_outer_update,
)
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER
from pysrc.meta_learning.reports.meta_validity_report import (
    MetaValidityReportBuildError,
    build_meta_validity_report,
    scaffold_confidence_calibration,
    scaffold_inner_loop_gain,
    scaffold_task_pool_counts,
    validate_meta_validity_report_keys,
)


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask_single() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _mask_two() -> tuple[bool, ...]:
    return tuple(i in (0, 1) for i in range(MAX_SIGNALS))


def _task(
    regime_class: str,
    i: int,
    *,
    mask: tuple[bool, ...] | None = None,
    regime_embedding: np.ndarray | None = None,
) -> MetaTask:
    mask = mask or _mask_single()
    active_k = int(sum(mask))
    day = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 5)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(5))
    query = tuple((day + timedelta(days=10 + j)).isoformat() for j in range(3))
    return MetaTask(
        task_id=f"{regime_class}-{i}",
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
        signal_ids_hash="sha256:abc",
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


def _sampler_from_tasks(tasks: list[MetaTask]) -> CurriculumSampler:
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    return CurriculumSampler(
        tasks,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks), crisis_floor_fraction=0.10, bucket_minimums=mins, seed=42
        ),
    )


def test_reptile_outer_update_exact() -> None:
    theta = np.arange(MAX_SIGNALS, dtype=np.float32)
    a1 = theta + np.float32(1.0)
    a2 = theta - np.float32(0.5)
    stack = np.stack([a1, a2], axis=0)
    out = reptile_outer_update(theta, stack, outer_lr=0.5)
    mean_delta = np.mean(stack - theta.reshape(1, -1), axis=0)
    expected = theta + np.float32(0.5) * mean_delta
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)


def test_crisis_floor_fail() -> None:
    tasks = tuple(_task("bull", i) for i in range(8))
    cfg = ReptileTrainerConfig()
    tr = ReptileTrainer(cfg, _FakeSampler(tasks), seed=1)
    theta = np.zeros(MAX_SIGNALS, dtype=np.float32)
    res = tr.run_batch(theta_meta=theta, seed=1)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "INSUFFICIENT_CRISIS_TASKS" in res.meta_validity_report["fail_reasons"]


@pytest.mark.determinism("d1")
def test_prepare_governed_curriculum_batch_matches_crisis_floor_fail(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    tasks = tuple(_task("bull", i) for i in range(8))
    cfg = ReptileTrainerConfig()
    prep = prepare_governed_curriculum_batch(_FakeSampler(tasks), cfg)
    assert isinstance(prep, GovernedCurriculumFailure)
    assert prep.reasons[0] == "INSUFFICIENT_CRISIS_TASKS"


def test_crisis_floor_pass() -> None:
    tasks = []
    for i, bucket in enumerate(REGIME_CLASS_ORDER):
        tasks.append(_task(bucket, i))
    for j in range(10):
        tasks.append(_task("crisis" if j % 2 == 0 else "bull", 100 + j))
    samp = _sampler_from_tasks(tasks)
    cfg = ReptileTrainerConfig(task_failure_abort_threshold=10)
    tr = ReptileTrainer(cfg, samp, seed=2)
    theta = np.zeros(MAX_SIGNALS, dtype=np.float32)
    res = tr.run_batch(theta_meta=theta, seed=2)
    assert res.meta_validity_report["overall_result"] == "PASS"
    assert res.meta_validity_report["fail_reasons"] == []


def test_consecutive_task_failures_abort() -> None:
    tasks = tuple(
        _task("crisis", i, regime_embedding=np.zeros(3, dtype=np.float32)) for i in range(5)
    )
    cfg = ReptileTrainerConfig(task_failure_abort_threshold=3)
    tr = ReptileTrainer(cfg, _FakeSampler(tasks), seed=3)
    theta = np.zeros(MAX_SIGNALS, dtype=np.float32)
    with mock.patch(
        "pysrc.meta.reptile_trainer._inner_adapt",
        side_effect=ValueError("inner_nonfinite"),
    ):
        res = tr.run_batch(theta_meta=theta, seed=3)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "CONSECUTIVE_TASK_FAILURES" in res.meta_validity_report["fail_reasons"]


def test_outer_update_nan_aborts() -> None:
    tasks = tuple(
        _task("crisis", i, regime_embedding=np.zeros(2, dtype=np.float32)) for i in range(4)
    )
    cfg = ReptileTrainerConfig()
    tr = ReptileTrainer(cfg, _FakeSampler(tasks), seed=4)
    theta = np.zeros(MAX_SIGNALS, dtype=np.float32)
    bad = np.full(MAX_SIGNALS, np.nan, dtype=np.float32)
    with mock.patch("pysrc.meta.reptile_trainer.reptile_outer_update", return_value=bad):
        res = tr.run_batch(theta_meta=theta, seed=4)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "OUTER_UPDATE_NAN_INF" in res.meta_validity_report["fail_reasons"]


def test_anil_without_encoder_raises() -> None:
    tasks = (_task("crisis", 0),)
    cfg = ReptileTrainerConfig(algorithm="anil")
    tr = ReptileTrainer(cfg, _FakeSampler(tasks), seed=5)
    with pytest.raises(DataPreconditionError, match="ANIL requires a ContextEncoder"):
        tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=5)


def test_anil_unfrozen_encoder_raises() -> None:
    pytest.importorskip("torch")
    from pysrc.meta_learning.context_encoder import ContextEncoder

    enc = ContextEncoder(seed=0)
    enc.unfreeze()
    tasks = (_task("crisis", 0),)
    cfg = ReptileTrainerConfig(algorithm="anil")
    tr = ReptileTrainer(cfg, _FakeSampler(tasks), seed=6, context_encoder=enc)
    with pytest.raises(DataPreconditionError, match="frozen context encoder"):
        tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=6)


def test_anil_frozen_encoder_runs() -> None:
    pytest.importorskip("torch")
    from pysrc.meta_learning.context_encoder import ContextEncoder

    enc = ContextEncoder(seed=0)
    enc.freeze()
    tasks = []
    for i, b in enumerate(REGIME_CLASS_ORDER):
        tasks.append(_task(b, i, regime_embedding=np.linspace(0, 1, 4, dtype=np.float32)))
    for j in range(8):
        tasks.append(
            _task("crisis", 50 + j, regime_embedding=np.linspace(0, 1, 4, dtype=np.float32))
        )
    samp = _sampler_from_tasks(tasks)
    cfg = ReptileTrainerConfig(algorithm="anil")
    tr = ReptileTrainer(cfg, samp, seed=7, context_encoder=enc)
    res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=7)
    assert res.meta_validity_report["overall_result"] == "PASS"


def test_signal_mask_inactive_zero() -> None:
    pool = [
        _task(b, i, regime_embedding=np.zeros(2, dtype=np.float32))
        for i, b in enumerate(REGIME_CLASS_ORDER)
    ]
    pool.append(
        _task("crisis", 90, mask=_mask_two(), regime_embedding=np.zeros(2, dtype=np.float32))
    )
    pool.append(
        _task("crisis", 91, mask=_mask_single(), regime_embedding=np.zeros(2, dtype=np.float32))
    )
    for j in range(20):
        pool.append(
            _task(
                "bull" if j % 2 else "high_vol",
                200 + j,
                regime_embedding=np.zeros(2, dtype=np.float32),
            )
        )
    samp = _sampler_from_tasks(pool)
    cfg = ReptileTrainerConfig()
    tr = ReptileTrainer(cfg, samp, seed=8)
    res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=8)
    assert res.meta_validity_report["overall_result"] == "PASS"
    assert res.theta_meta.shape == (MAX_SIGNALS,)


def test_theta_day_prime_prior_unchanged_on_fail() -> None:
    prior = np.linspace(0.1, 0.2, MAX_SIGNALS, dtype=np.float32)
    tasks = tuple(_task("bull", i) for i in range(8))
    cfg = ReptileTrainerConfig()
    tr = ReptileTrainer(cfg, _FakeSampler(tasks), seed=9)
    res = tr.run_batch(
        theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), theta_day_prime_prior=prior, seed=9
    )
    assert res.meta_validity_report["overall_result"] == "FAIL"
    np.testing.assert_array_equal(res.theta_day_prime, prior)


def test_theta_day_prime_promoted_on_pass_distinct() -> None:
    tasks = []
    for i, b in enumerate(REGIME_CLASS_ORDER):
        emb = np.full(3, float(i + 1), dtype=np.float32)
        tasks.append(_task(b, i, regime_embedding=emb))
    for j in range(10):
        emb = np.full(3, float(j), dtype=np.float32)
        tasks.append(_task("crisis" if j % 2 == 0 else "high_vol", 100 + j, regime_embedding=emb))
    samp = _sampler_from_tasks(tasks)
    cfg = ReptileTrainerConfig()
    tr = ReptileTrainer(cfg, samp, seed=10)
    theta = np.random.default_rng(0).standard_normal(MAX_SIGNALS).astype(np.float32) * np.float32(
        0.01
    )
    res = tr.run_batch(theta_meta=theta, seed=10)
    assert res.theta_day_prime_promoted is True
    assert res.theta_day_prime is not None
    assert not np.allclose(res.theta_day_prime, res.theta_meta)


def test_ewc_stub_no_op() -> None:
    cfg = ReptileTrainerConfig()
    tr = ReptileTrainer(cfg, _FakeSampler((_task("crisis", 0),)), seed=11)
    theta = np.ones(MAX_SIGNALS, dtype=np.float32)
    out = tr.apply_ewc_seam(theta, lambda_ewc=0.25)
    np.testing.assert_array_equal(out, theta)


def test_pass_and_fail_reports_structurally_valid() -> None:
    ok = build_meta_validity_report(
        schema_version="v1",
        run_id="run.sha256:abc",
        overall_result="PASS",
        reporting_gate="MLC3_SCAFFOLD",
        inner_loop_gain=scaffold_inner_loop_gain(
            mean_query_ic=0.01, by_regime_class={"bull": 0.01}
        ),
        shuffle_test_p_value=None,
        proxy_IC_pearson_r=None,
        crisis_holdout_ic=None,
        forgetting_ic_degradation_pct=None,
        task_pool_counts=scaffold_task_pool_counts(
            batch_size=1, crisis_count=1, crisis_required=0, bucket_counts={"bull": 1}
        ),
        confidence_calibration=scaffold_confidence_calibration(),
        fail_reasons=[],
        theta_day_prime_promoted=True,
        timestamp_utc="2026-04-20T00:00:00Z",
    )
    validate_meta_validity_report_keys(ok)
    bad = {"schema_version": "v1", "run_id": "x"}
    with pytest.raises(MetaValidityReportBuildError):
        validate_meta_validity_report_keys(bad)


def test_report_builder_rejects_bad_overall() -> None:
    with pytest.raises(MetaValidityReportBuildError):
        build_meta_validity_report(
            schema_version="v1",
            run_id="run.sha256:abc",
            overall_result="MAYBE",
            reporting_gate="MLC3_SCAFFOLD",
            inner_loop_gain=scaffold_inner_loop_gain(),
            shuffle_test_p_value=None,
            proxy_IC_pearson_r=None,
            crisis_holdout_ic=None,
            forgetting_ic_degradation_pct=None,
            task_pool_counts=None,
            confidence_calibration=None,
            fail_reasons=["x"],
            theta_day_prime_promoted=False,
        )

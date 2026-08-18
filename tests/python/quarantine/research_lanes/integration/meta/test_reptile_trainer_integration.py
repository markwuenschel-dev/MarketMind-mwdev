from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from pysrc.meta.curriculum import CurriculumSampler, CurriculumSamplerConfig
from pysrc.meta.reptile_trainer import ReptileTrainer
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER
from pysrc.meta_learning.reports.meta_validity_report import validate_meta_validity_report_keys


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(regime_class: str, i: int) -> MetaTask:
    day = datetime(2024, 6, 1, tzinfo=UTC) + timedelta(days=i * 3)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(6))
    query = tuple((day + timedelta(days=20 + j)).isoformat() for j in range(4))
    emb = np.full(4, 0.01 * float(i), dtype=np.float32)
    return MetaTask(
        task_id=f"int-{regime_class}-{i}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0=support[0],
        t1=(day + timedelta(days=40)).isoformat(),
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=_mask(),
        signal_set_version="1",
        signal_ids_hash="sha256:inttest",
        horizon=1,
        active_k=1,
        regime_embedding=emb,
    )


def _pool() -> list[MetaTask]:
    out: list[MetaTask] = []
    for i, b in enumerate(REGIME_CLASS_ORDER):
        out.append(_task(b, i))
    for j in range(24):
        out.append(_task("crisis" if j % 3 == 0 else "bull", 100 + j))
    return out


def test_full_epoch_pass_and_json_roundtrip() -> None:
    pool = _pool()
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    cfg = CurriculumSamplerConfig(
        batch_size=16,
        crisis_floor_fraction=0.10,
        bucket_minimums=mins,
        seed=999,
    )

    def _trainer() -> ReptileTrainer:
        samp = CurriculumSampler(pool, config=cfg)
        return ReptileTrainer(
            ReptileTrainerConfig(task_failure_abort_threshold=10), samp, seed=4242
        )

    theta = np.random.default_rng(4242).standard_normal(MAX_SIGNALS).astype(
        np.float32
    ) * np.float32(0.02)
    r1 = _trainer().run_batch(theta_meta=theta, seed=4242)
    r2 = _trainer().run_batch(theta_meta=theta, seed=4242)
    assert r1.meta_validity_report["overall_result"] == "PASS"
    assert r1.meta_validity_report == r2.meta_validity_report
    assert r1.theta_meta.shape == (MAX_SIGNALS,)
    ilg = r1.meta_validity_report["inner_loop_gain"]
    assert isinstance(ilg, dict)
    assert set(ilg.keys()) == {"mean_query_ic", "harvey_t", "by_regime_class"}
    assert r1.inner_loop_gain is not None
    assert math.isfinite(r1.inner_loop_gain)
    raw = json.dumps(r1.meta_validity_report)
    back = json.loads(raw)
    assert back["schema_version"] == "v1"
    assert back["theta_day_prime_promoted"] is True
    assert back["inner_loop_gain"]["mean_query_ic"] == ilg["mean_query_ic"]


def test_crisis_floor_early_abort_integration() -> None:
    from pysrc.meta.curriculum import CurriculumBatch

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
            c = Counter(t.regime_class for t in self._tasks)
            bc = {b: int(c.get(b, 0)) for b in REGIME_CLASS_ORDER}
            return CurriculumBatch(
                tasks=self._tasks,
                phase="bootstrap",
                bucket_counts=bc,
                priority_alpha=0.6,
                importance_beta=0.4,
                importance_weights=(),
            )

    tasks = tuple(_task("bull", i) for i in range(12))
    tr = ReptileTrainer(ReptileTrainerConfig(), _FakeSampler(tasks), seed=1)
    res = tr.run_batch(theta_meta=np.zeros(MAX_SIGNALS, dtype=np.float32), seed=1)
    assert res.meta_validity_report["overall_result"] == "FAIL"
    assert "INSUFFICIENT_CRISIS_TASKS" in res.meta_validity_report["fail_reasons"]


def test_example_artifact_is_valid_json() -> None:
    path = Path("artifacts/phase_ii/mlc3/meta_validity_report_example.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v1"
    assert data["overall_result"] == "PASS"
    assert data["reporting_gate"] == "MLC3_SCAFFOLD"
    assert data["theta_day_prime_promoted"] is True
    ig = data["inner_loop_gain"]
    assert isinstance(ig, dict)
    assert set(ig.keys()) == {"mean_query_ic", "harvey_t", "by_regime_class"}
    validate_meta_validity_report_keys(data)

"""Unit tests for :mod:`pysrc.meta.w1_baseline_incumbent`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")

from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.w1_baseline_errors import W1BaselineEvidenceError
from pysrc.meta.w1_baseline_incumbent import XGBoostIncumbentBaseline, XGBoostIncumbentConfig


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _task(*, pit_ok: bool = True) -> MetaTask:
    emb = np.zeros(64, dtype=np.float32)
    support = tuple(f"2024-01-0{i}T00:00:00+00:00" for i in range(1, 7))
    pit = support[-1]
    if pit_ok:
        query = tuple(f"2024-01-{10 + i}T00:00:00+00:00" for i in range(4))
    else:
        query = (pit, "2024-01-07T00:00:00+00:00")
    return MetaTask(
        task_id="inc-test-1",
        regime_id="trend_hi__vol_hi__bocpd_stable",
        regime_class="high_vol",
        t0=support[0],
        t1="2024-01-20T00:00:00+00:00",
        pit_boundary=pit,
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=_mask(),
        signal_set_version="1",
        signal_ids_hash="sha256:inc",
        horizon=1,
        active_k=1,
        regime_embedding=emb,
    )


@pytest.mark.determinism("d1")
def test_xgb_incumbent_fit_and_predict_shape(deterministic_seed: int) -> None:
    _ = deterministic_seed
    t = _task()
    m = XGBoostIncumbentBaseline(
        XGBoostIncumbentConfig(n_estimators=8, max_depth=2, random_state=1)
    )
    m.fit_for_training_tasks([t], rng_seed=42)
    scores = m.predict_scores(t, fold_index=0, rng_seed=42)
    assert scores.shape == (len(t.query_set),)


@pytest.mark.determinism("d1")
def test_pit_violation_query_before_pit(deterministic_seed: int) -> None:
    _ = deterministic_seed
    t = _task(pit_ok=False)
    m = XGBoostIncumbentBaseline()
    m.fit_for_training_tasks([_task()], rng_seed=1)
    with pytest.raises(W1BaselineEvidenceError, match="PIT violation"):
        m.predict_scores(t, fold_index=0, rng_seed=1)


@pytest.mark.determinism("d1")
def test_challenger_proxy_scores_length(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.w1_baseline_incumbent import challenger_proxy_scores_for_tasks

    tasks = (_task(), _task())
    v = challenger_proxy_scores_for_tasks(tasks, fold_index=0, rng_seed=5)
    assert v.shape == (2,)

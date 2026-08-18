"""Unit tests for governed W1 linear meta-head trainer + checkpoint emission."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.python.unit.meta.test_task_generator import _Encoder

from pysrc.meta.curriculum import CurriculumSampler, CurriculumSamplerConfig
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.w1_governed_csv_dataview import W1GovernedCsvDataView
from pysrc.meta.w1_governed_learned_checkpoint_emit import emit_w1_learned_checkpoint_v2
from pysrc.meta.w1_governed_meta_head_trainer import (
    TRAINED_BY_W1_GOVERNED_META_HEAD_V1,
    W1GovernedMetaHeadTrainConfig,
    W1GovernedMetaHeadTrainFailure,
    W1GovernedMetaHeadTrainSuccess,
    fit_w1_governed_meta_head,
)
from pysrc.meta.w1_real_task_pool import (
    W1RealPoolConfig,
    W1RealTaskPoolOutcome,
    build_w1_real_task_pool,
    w1_support_mean_log_return_net_from_pool,
)
from pysrc.meta.w1_reptile_trained_meta_allocator_adapter import ReptileTrainedMetaAllocatorAdapter
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

_FIXTURE_CSV = (
    Path(__file__).resolve().parents[3] / "fixtures" / "w1" / "spy_daily_close_fixture.csv"
)


class _Universe:
    def __call__(self, knowledge_date: date) -> tuple[str, ...]:
        _ = knowledge_date
        return ("SPY",)


def _small_pool() -> tuple[W1RealTaskPoolOutcome, W1RealPoolConfig]:
    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    dv = W1GovernedCsvDataView.from_csv(_FIXTURE_CSV, symbol="SPY")
    t0 = pd.Timestamp("2010-01-04T00:00:00Z")
    t1 = pd.Timestamp("2015-12-31T00:00:00Z")
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
    n = len(list(pd.bdate_range(t0.normalize(), t1.normalize(), freq="C")))
    returns = np.zeros(n, dtype=np.float64)
    if n > 1:
        i = np.arange(1, n, dtype=np.float64)
        returns[1:] = np.sin(i / 15.0) * 0.021 + np.cos(i / 31.0) * 0.019
    pool_cfg = W1RealPoolConfig(
        data_view=dv,
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=101,
        bucket_minimums=mins,
        start_ts="2010-01-04T00:00:00Z",
        end_ts="2015-12-31T00:00:00Z",
        evidence_mode="unit_fixture",
        test_fixture_log_returns=returns,
    )
    return build_w1_real_task_pool(pool_cfg), pool_cfg


@pytest.mark.determinism("d1")
def test_fit_fails_closed_without_support_targets(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pool_out, _ = _small_pool()
    tasks_sorted = tuple(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)))
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    rtc = ReptileTrainerConfig()
    samp = CurriculumSampler(
        tasks_sorted,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks_sorted),
            crisis_floor_fraction=float(rtc.crisis_floor_pct),
            bucket_minimums=mins,
            seed=707,
        ),
    )
    out = fit_w1_governed_meta_head(
        sampler=samp,
        reptile_config=rtc,
        support_targets_by_task_id={},
        head_cfg=W1GovernedMetaHeadTrainConfig(),
        seed=1,
    )
    assert isinstance(out, W1GovernedMetaHeadTrainFailure)
    assert out.reasons[0] == "INSUFFICIENT_SUPPORT_TARGETS"


@pytest.mark.determinism("d1")
def test_fit_emit_adapter_roundtrip(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    pool_out, pool_cfg = _small_pool()
    assert pool_out.data_fingerprint is not None
    data_fp = str(pool_out.data_fingerprint)
    tasks_sorted = tuple(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)))
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    rtc = ReptileTrainerConfig()
    samp = CurriculumSampler(
        tasks_sorted,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks_sorted),
            crisis_floor_fraction=float(rtc.crisis_floor_pct),
            bucket_minimums=mins,
            seed=808,
        ),
    )
    targets = w1_support_mean_log_return_net_from_pool(pool_out)
    fit_out = fit_w1_governed_meta_head(
        sampler=samp,
        reptile_config=rtc,
        support_targets_by_task_id=targets,
        head_cfg=W1GovernedMetaHeadTrainConfig(outer_lr=0.06, n_gradient_steps=120),
        seed=42,
    )
    assert isinstance(fit_out, W1GovernedMetaHeadTrainSuccess)
    ck = tmp_path / "ck.json"
    emit_w1_learned_checkpoint_v2(
        ck,
        weights=[float(x) for x in fit_out.weights.tolist()],
        model_state_hash=fit_out.model_state_hash,
        trainer_config_hash=fit_out.trainer_config_hash,
        training_task_pool_hash=pool_out.task_pool_hash,
        training_data_fingerprint=data_fp,
        training_splits_fingerprint="sha256:" + "d" * 64,
        signal_set_version=str(pool_cfg.signal_set_version),
        feature_encoder_contract_version="w1_task_regime_one_hot_v1",
        code_version=fit_out.code_version,
        created_at_utc="2026-04-24T12:00:00Z",
        trained_by_runner=TRAINED_BY_W1_GOVERNED_META_HEAD_V1,
        training_run_id=fit_out.training_run_id,
    )
    adapter = ReptileTrainedMetaAllocatorAdapter(
        ck,
        expected_signal_set_version=str(pool_cfg.signal_set_version),
        expected_training_task_pool_hash=pool_out.task_pool_hash,
        expected_training_data_fingerprint=data_fp,
        expected_training_splits_fingerprint="sha256:" + "d" * 64,
    )
    assert adapter.governed_checkpoint_lineage_verified is True
    assert adapter.model_state_hash == fit_out.model_state_hash

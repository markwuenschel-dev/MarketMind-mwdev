"""Unit tests for :mod:`pysrc.meta.w1_real_task_pool`."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.python.unit.meta.test_task_generator import _Encoder

from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler
from pysrc.meta.w1_governed_csv_dataview import W1GovernedCsvDataView
from pysrc.meta.w1_real_task_pool import (
    W1RealPoolBuildError,
    W1RealPoolConfig,
    build_w1_real_task_pool,
    w1_support_mean_log_return_net_from_pool,
)
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

_FIXTURE_CSV = (
    Path(__file__).resolve().parents[3] / "fixtures" / "w1" / "spy_daily_close_fixture.csv"
)


def _csv_data_view() -> W1GovernedCsvDataView:
    return W1GovernedCsvDataView.from_csv(_FIXTURE_CSV, symbol="SPY")


def _fixture_log_returns_for_range(*, start_ts: str, end_ts: str) -> np.ndarray:
    t0 = pd.Timestamp(start_ts)
    t1 = pd.Timestamp(end_ts)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
    n = len(list(pd.bdate_range(t0.normalize(), t1.normalize(), freq="C")))
    out = np.zeros(n, dtype=np.float64)
    if n > 1:
        i = np.arange(1, n, dtype=np.float64)
        out[1:] = np.sin(i / 15.0) * 0.021 + np.cos(i / 31.0) * 0.019
    return out


class _Universe:
    def __call__(self, d: date) -> tuple[str, ...]:
        return ("SPY",)


def _tiny_mins() -> dict[str, int]:
    # Floors of 1 keep the fixture path fast while still exercising all buckets.
    return dict.fromkeys(REGIME_CLASS_ORDER, 1)


@pytest.mark.determinism("d1")
def test_build_w1_real_task_pool_stable_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    pool_cfg = W1RealPoolConfig(
        data_view=_csv_data_view(),
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=99,
        start_ts="2010-01-04T00:00:00Z",
        end_ts="2015-12-31T00:00:00Z",
        bucket_minimums=_tiny_mins(),
        evidence_mode="unit_fixture",
        test_fixture_log_returns=_fixture_log_returns_for_range(
            start_ts="2010-01-04T00:00:00Z",
            end_ts="2015-12-31T00:00:00Z",
        ),
    )
    a = build_w1_real_task_pool(pool_cfg)
    b = build_w1_real_task_pool(pool_cfg)
    assert a.task_pool_hash == b.task_pool_hash
    assert len(a.tasks) >= 5
    assert a.diagnostics["total_tasks"] == len(a.tasks)
    assert a.data_fingerprint is not None
    assert a.data_fingerprint.startswith("sha256:")
    assert len(a.task_query_targets_net) == len(a.tasks)
    assert len(a.task_support_mean_log_return_net) == len(a.tasks)
    sup = w1_support_mean_log_return_net_from_pool(a)
    assert set(sup) == {t.task_id for t in a.tasks}
    query_targets = dict(a.task_query_targets_net)
    task_windows = {(t.t0, t.t1, t.pit_boundary) for t in a.tasks}
    support_windows = {(t.support_set[0], t.support_set[-1]) for t in a.tasks}
    query_windows = {(t.query_set[0], t.query_set[-1]) for t in a.tasks}
    assert len(task_windows) > 1
    assert len(support_windows) > 1
    assert len(query_windows) > 1
    assert np.std([float(query_targets[str(t.task_id)]) for t in a.tasks], ddof=1) > 0.0
    assert np.std([float(sup[str(t.task_id)]) for t in a.tasks], ddof=1) > 0.0
    assert len({float(query_targets[str(t.task_id)]) for t in a.tasks}) >= 2
    assert len({float(sup[str(t.task_id)]) for t in a.tasks}) >= 2


@pytest.mark.determinism("d1")
def test_manifest_projection_roundtrip_length(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.w1_real_task_pool import w1_meta_tasks_as_manifest_inputs

    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    pool_cfg = W1RealPoolConfig(
        data_view=_csv_data_view(),
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=3,
        bucket_minimums=_tiny_mins(),
        start_ts="2010-01-04T00:00:00Z",
        end_ts="2015-12-31T00:00:00Z",
        evidence_mode="unit_fixture",
        test_fixture_log_returns=_fixture_log_returns_for_range(
            start_ts="2010-01-04T00:00:00Z",
            end_ts="2015-12-31T00:00:00Z",
        ),
    )
    out = build_w1_real_task_pool(pool_cfg)
    rows = w1_meta_tasks_as_manifest_inputs(out.tasks)
    assert len(rows) == len(out.tasks)


@pytest.mark.determinism("d1")
def test_build_raises_when_bucket_floors_impossible(deterministic_seed: int) -> None:
    _ = deterministic_seed
    bcfg = BOCPDConfig(cold_start_burn_in=400, vol_window=10, trend_window=15)
    pool_cfg = W1RealPoolConfig(
        data_view=_csv_data_view(),
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=1,
        bucket_minimums=dict.fromkeys(REGIME_CLASS_ORDER, 50),
        start_ts="2014-01-02T00:00:00Z",
        end_ts="2014-12-31T00:00:00Z",
        max_calendar_bars=120,
    )
    with pytest.raises(W1RealPoolBuildError, match="bucket floors") as excinfo:
        build_w1_real_task_pool(pool_cfg)
    details = excinfo.value.details or {}
    assert "missing_regime_buckets" in details
    assert isinstance(details["missing_regime_buckets"], list)


@pytest.mark.determinism("d1")
def test_governed_real_forbids_fixture_returns(deterministic_seed: int) -> None:
    _ = deterministic_seed
    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    pool_cfg = W1RealPoolConfig(
        data_view=_csv_data_view(),
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=1,
        bucket_minimums=_tiny_mins(),
        start_ts="2010-01-04T00:00:00Z",
        end_ts="2015-12-31T00:00:00Z",
        evidence_mode="governed_real",
        test_fixture_log_returns=_fixture_log_returns_for_range(
            start_ts="2010-01-04T00:00:00Z",
            end_ts="2015-12-31T00:00:00Z",
        ),
    )
    with pytest.raises(W1RealPoolBuildError, match="test_fixture_log_returns is forbidden"):
        build_w1_real_task_pool(pool_cfg)


@pytest.mark.determinism("d1")
def test_build_requires_historical_bounds(deterministic_seed: int) -> None:
    _ = deterministic_seed
    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    pool_cfg = W1RealPoolConfig(
        data_view=_csv_data_view(),
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=1,
        bucket_minimums=_tiny_mins(),
        start_ts="",
        end_ts="2015-12-31T00:00:00Z",
    )
    with pytest.raises(W1RealPoolBuildError, match="start_ts"):
        build_w1_real_task_pool(pool_cfg)

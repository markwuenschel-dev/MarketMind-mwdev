"""Unit tests for RegimeLabeler (compositional regime_id + 5-class projection)."""

from __future__ import annotations

import numpy as np
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import (
    RegimeLabeler,
    annualized_log_rv_from_returns,
    validate_regime_id,
)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_5_class_projection_crisis_requires_severity() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    assert lb.project_regime_class("hi", "hi", "cp", severity_flag=True) == "crisis"
    assert lb.project_regime_class("hi", "hi", "cp", severity_flag=False) == "high_vol"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_5_class_projection_full_truth_table() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    trends = ("hi", "lo", "flat")
    vols = ("hi", "med", "lo")
    bstates = ("stable", "transition", "cp")
    for tr in trends:
        for vo in vols:
            for bs in bstates:
                rc_s = lb.project_regime_class(tr, vo, bs, severity_flag=True)
                rc_n = lb.project_regime_class(tr, vo, bs, severity_flag=False)
                rid = lb.compute_regime_id(tr, vo, bs)
                assert validate_regime_id(rid)
                if vo == "hi":
                    assert rc_s == "crisis"
                    assert rc_n == "high_vol"
                elif tr == "lo":
                    assert rc_s == "bear"
                    assert rc_n == "bear"
                elif tr == "hi":
                    assert rc_s == "bull"
                    assert rc_n == "bull"
                else:
                    assert rc_s == "sideways"
                    assert rc_n == "sideways"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_bocpd_gated_reference_legacy_crisis() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    assert lb.project_regime_class_bocpd_gated_reference("hi", "hi", "cp") == "crisis"
    assert lb.project_regime_class_bocpd_gated_reference("hi", "hi", "stable") == "high_vol"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_flag_pit_safe_on_suffix() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=2, crisis_vol_score_percentile=90.0)
    lb = RegimeLabeler(cfg)
    xs = np.arange(30, dtype=np.float64)
    a = lb.compute_severity_flag_vol_score_raw(xs, 20)
    xs_alt = xs.copy()
    xs_alt[25:] = 999.0
    b = lb.compute_severity_flag_vol_score_raw(xs_alt, 20)
    assert a == b


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_flag_raises_data_precondition_when_history_too_short() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=0)
    lb = RegimeLabeler(cfg)
    xs = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(DataPreconditionError, match="out of range"):
        lb.compute_severity_flag_vol_score_raw(xs, 5)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_flag_raises_data_precondition_when_pit_negative() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=0)
    lb = RegimeLabeler(cfg)
    xs = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(DataPreconditionError, match="non-negative"):
        lb.compute_severity_flag_vol_score_raw(xs, -1)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_flag_false_during_cold_window() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=10)
    lb = RegimeLabeler(cfg)
    xs = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    assert lb.compute_severity_flag_vol_score_raw(xs, 5) is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_id_format() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    rid = lb.compute_regime_id("flat", "med", "stable")
    assert validate_regime_id(rid)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_expanding_tercile_pit_safe() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=5)
    lb = RegimeLabeler(cfg)
    xs = np.arange(30, dtype=np.float64)
    a = lb.compute_vol_regime(xs, 20)
    xs_alt = xs.copy()
    xs_alt[25:] = 999.0
    b = lb.compute_vol_regime(xs_alt, 20)
    assert a == b


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_expanding_tercile_deterministic() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    xs = np.sin(np.linspace(0, 3.0, 80)).astype(np.float64)
    assert lb.compute_vol_regime(xs, 50) == lb.compute_vol_regime(xs.copy(), 50)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_trend_regime_respects_epsilon() -> None:
    cfg = BOCPDConfig(trend_window=10, trend_flat_epsilon=0.05)
    lb = RegimeLabeler(cfg)
    r = np.full(20, 0.001, dtype=np.float64)
    assert lb.compute_trend_regime(r, 19) == "flat"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_trend_regime_sign() -> None:
    cfg = BOCPDConfig(trend_window=5, trend_flat_epsilon=0.001)
    lb = RegimeLabeler(cfg)
    up = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    assert lb.compute_trend_regime(up, 6) == "hi"
    down = -up
    assert lb.compute_trend_regime(down, 6) == "lo"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_vol_regime_ordering() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=0)
    lb = RegimeLabeler(cfg)
    xs = np.concatenate([np.zeros(25, dtype=np.float64), np.full(8, 50.0, dtype=np.float64)])
    assert lb.compute_vol_regime(xs, 24) == "lo"
    assert lb.compute_vol_regime(xs, 31) == "hi"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_trend_short_window_returns_flat() -> None:
    cfg = BOCPDConfig(trend_window=20)
    lb = RegimeLabeler(cfg)
    r = np.ones(10, dtype=np.float64)
    assert lb.compute_trend_regime(r, 5) == "flat"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_trend_slice_shorter_than_window_returns_flat() -> None:
    cfg = BOCPDConfig(trend_window=10)
    lb = RegimeLabeler(cfg)
    r = np.zeros(5, dtype=np.float64)
    assert lb.compute_trend_regime(r, 9) == "flat"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_vol_expanding_too_few_points_returns_med() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=0)
    lb = RegimeLabeler(cfg)
    xs = np.array([1.0, 2.0], dtype=np.float64)
    assert lb.compute_vol_regime(xs, 1) == "med"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_vol_empty_history_returns_med() -> None:
    lb = RegimeLabeler(BOCPDConfig())
    xs = np.array([], dtype=np.float64)
    assert lb.compute_vol_regime(xs, 0) == "med"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_log_rv_computation() -> None:
    r = np.full(21, 0.01, dtype=np.float64)
    v = annualized_log_rv_from_returns(r, 20, 21)
    assert np.isfinite(v)
    expected = float(np.log(np.sqrt(252.0 / 21.0 * float(np.sum(r**2)))))
    assert abs(v - expected) < 1e-9

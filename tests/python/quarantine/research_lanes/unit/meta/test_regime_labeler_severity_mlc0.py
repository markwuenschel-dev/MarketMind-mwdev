"""MLC-0 · Unit tests for :meth:`RegimeLabeler.compute_severity_flag_vol_score_raw`.

Covers brief §5 Step 7 acceptance for the severity_flag path:
- Uses only strictly prior values (expanding window)
- Returns ``False`` during ``cold_start_burn_in`` burn-in
- Current bar excluded from the percentile reference set
- Threshold rising with the current value crosses severity (p90 default)
- Projection: ``crisis`` requires ``vol_hi AND severity_flag``; BOCPD is
  not the crisis primitive
"""

from __future__ import annotations

import numpy as np
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_false_during_burn_in() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=20)
    lb = RegimeLabeler(cfg)
    hist = np.linspace(0.1, 1.0, 50, dtype=np.float64)
    assert lb.compute_severity_flag_vol_score_raw(hist, 10) is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_true_when_current_exceeds_percentile() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=5, crisis_vol_score_percentile=90.0)
    lb = RegimeLabeler(cfg)
    hist = np.concatenate(
        [
            np.zeros(20, dtype=np.float64),
            np.full(5, 10.0, dtype=np.float64),
        ]
    )
    # current index 24 sees a large value; strictly prior window [5:24] ends with 10s too
    assert lb.compute_severity_flag_vol_score_raw(hist, 24) is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_current_bar_excluded_from_reference() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=5, crisis_vol_score_percentile=90.0)
    lb = RegimeLabeler(cfg)
    # past is flat zero; current is 100; p90 of past is 0; 100 >= 0 → True
    hist = np.zeros(30, dtype=np.float64)
    hist[-1] = 100.0
    assert lb.compute_severity_flag_vol_score_raw(hist, 29) is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_raises_on_negative_index() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    with pytest.raises(DataPreconditionError):
        lb.compute_severity_flag_vol_score_raw(np.ones(10, dtype=np.float64), -1)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_raises_on_oob_index() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    with pytest.raises(DataPreconditionError):
        lb.compute_severity_flag_vol_score_raw(np.ones(10, dtype=np.float64), 10)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_returns_false_when_past_set_is_empty() -> None:
    # burn_in == pit_boundary_idx: strict-prior window is empty → False
    cfg = BOCPDConfig(cold_start_burn_in=5)
    lb = RegimeLabeler(cfg)
    hist = np.linspace(0, 1, 20, dtype=np.float64)
    assert lb.compute_severity_flag_vol_score_raw(hist, 5) is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_id_format_matches_compositional_grammar() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    for trend in ("hi", "lo", "flat"):
        for vol in ("hi", "med", "lo"):
            for bocpd in ("stable", "transition", "cp"):
                rid = lb.compute_regime_id(trend, vol, bocpd)
                assert rid == f"trend_{trend}__vol_{vol}__bocpd_{bocpd}"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_class_crisis_requires_vol_hi_and_severity() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    # BOCPD alone (without severity) cannot produce crisis
    assert lb.project_regime_class("hi", "hi", "cp", severity_flag=False) == "high_vol"
    assert lb.project_regime_class("hi", "hi", "cp", severity_flag=True) == "crisis"
    # vol != hi + severity is still not crisis
    assert lb.project_regime_class("hi", "med", "cp", severity_flag=True) == "bull"

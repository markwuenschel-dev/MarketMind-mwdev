"""Compositional regime_id strings + 5-class projection (Architecture Vision §4.2)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta_learning.regime_vocabulary import (
    is_valid_compositional_regime_id,
    project_regime_class_bocpd_reference,
    project_regime_class_extended_ablation,
)
from pysrc.meta_learning.regime_vocabulary import (
    project_regime_class as project_regime_class_canonical,
)


class RegimeLabeler:
    """Computes compositional regime_id and derived regime_class."""

    def __init__(self, config: BOCPDConfig) -> None:
        self._config = config

    def compute_trend_regime(
        self, returns: NDArray[np.float64], pit_boundary_idx: int
    ) -> Literal["hi", "lo", "flat"]:
        """Trailing trend_window-day log return; bucketed by epsilon (PIT-safe slice end)."""
        w = self._config.trend_window
        if pit_boundary_idx < w - 1:
            return "flat"
        r = returns[pit_boundary_idx - w + 1 : pit_boundary_idx + 1]
        if r.size < w:
            return "flat"
        log_ret = float(np.sum(np.log1p(r)))
        eps = self._config.trend_flat_epsilon
        if log_ret > eps:
            return "hi"
        if log_ret < -eps:
            return "lo"
        return "flat"

    def compute_vol_regime(
        self, log_rv_history: NDArray[np.float64], pit_boundary_idx: int
    ) -> Literal["hi", "med", "lo"]:
        """
        Expanding-window terciles of log-RV up to pit_boundary_idx (inclusive).

        Uses only indices [cold_start_end, pit_boundary_idx] per RG-09 expanding policy.
        """
        cold = self._config.cold_start_burn_in
        xs = np.asarray(log_rv_history[: pit_boundary_idx + 1], dtype=np.float64).ravel()
        if xs.size == 0:
            return "med"
        exp = xs[cold : pit_boundary_idx + 1] if pit_boundary_idx >= cold else xs
        if exp.size < 3:
            return "med"
        p33, p67 = np.percentile(exp, [100 / 3, 200 / 3], method="linear")
        v = float(xs[pit_boundary_idx])
        if v <= p33:
            return "lo"
        if v >= p67:
            return "hi"
        return "med"

    def compute_regime_id(
        self,
        trend: Literal["hi", "lo", "flat"],
        vol: Literal["hi", "med", "lo"],
        bocpd_state: Literal["stable", "transition", "cp"],
    ) -> str:
        """Returns 'trend_{x}__vol_{y}__bocpd_{z}'."""
        return f"trend_{trend}__vol_{vol}__bocpd_{bocpd_state}"

    def compute_severity_flag_vol_score_raw(
        self,
        vol_score_history: NDArray[np.float64],
        pit_boundary_idx: int,
    ) -> bool:
        """
        PIT-safe severity for MLN-02-AMD-01: current vol_score_raw vs p-percentile of
        strict past history [cold_start_burn_in, pit_boundary_idx). RG09-V12 default p=90.

        Threshold semantics (MLC-0 authority):

        - Expanding window over strictly prior values — the bar at
          ``pit_boundary_idx`` is **excluded** from the percentile
          reference set.
        - Window starts after ``cold_start_burn_in``.  During burn-in
          (``pit_boundary_idx < cold_start_burn_in``) this function
          **returns False** — severity is not defined without a stable
          past reference, and those labels must not be used as gate
          evidence without explicit separation (AQ-04 cold-start rule).
        - Default percentile is p90 (``BOCPDConfig.crisis_vol_score_percentile``).
          ⚑ VALIDATE — do **not** tune against Anti-Goodhart holdouts
          (MLN-02-AMD-01 §5.1); calibration happens via MLN-07 threshold
          governance, not from observed outputs.
        """
        if pit_boundary_idx < 0:
            raise DataPreconditionError(
                f"pit_boundary_idx={pit_boundary_idx} must be non-negative",
                details={"pit_boundary_idx": pit_boundary_idx},
            )
        hist = np.asarray(vol_score_history, dtype=np.float64).ravel()
        hist_len = int(hist.size)
        if pit_boundary_idx >= hist_len:
            raise DataPreconditionError(
                f"pit_boundary_idx={pit_boundary_idx} out of range for "
                f"vol_score_history length={hist_len}",
                details={"pit_boundary_idx": pit_boundary_idx, "history_length": hist_len},
            )

        cold = self._config.cold_start_burn_in
        pct = self._config.crisis_vol_score_percentile
        exp = hist[: pit_boundary_idx + 1]
        if pit_boundary_idx < cold or exp.size == 0:
            return False
        current = float(exp[pit_boundary_idx])
        past = exp[cold:pit_boundary_idx]
        if past.size < 1:
            return False
        thresh = float(np.percentile(past, pct, method="linear"))
        return current >= thresh

    def project_regime_class(
        self,
        trend: Literal["hi", "lo", "flat"],
        vol: Literal["hi", "med", "lo"],
        bocpd_state: Literal["stable", "transition", "cp"],
        *,
        severity_flag: bool,
    ) -> Literal["bull", "bear", "sideways", "high_vol", "crisis"]:
        """
        Delegates to :func:`pysrc.meta_learning.regime_vocabulary.project_regime_class`.
        """
        return project_regime_class_canonical(trend, vol, bocpd_state, severity_flag=severity_flag)

    @staticmethod
    def project_regime_class_bocpd_gated_reference(
        trend: Literal["hi", "lo", "flat"],
        vol: Literal["hi", "med", "lo"],
        bocpd_state: Literal["stable", "transition", "cp"],
    ) -> Literal["bull", "bear", "sideways", "high_vol", "crisis"]:
        """II-0A reference; see :func:`pysrc.meta_learning.regime_vocabulary.project_regime_class_bocpd_reference`."""
        return project_regime_class_bocpd_reference(trend, vol, bocpd_state)

    def project_regime_class_extended(
        self,
        trend: Literal["hi", "lo", "flat"],
        vol: Literal["hi", "med", "lo"],
        bocpd_state: Literal["stable", "transition", "cp"],
    ) -> Literal["bull", "bear", "sideways", "high_vol", "crisis"]:
        """Ablation diagnostic; see :func:`pysrc.meta_learning.regime_vocabulary.project_regime_class_extended_ablation`."""
        return project_regime_class_extended_ablation(trend, vol, bocpd_state)


def validate_regime_id(regime_id: str) -> bool:
    """True iff ``regime_id`` matches compositional Level-1 grammar (MLN-02)."""
    return is_valid_compositional_regime_id(regime_id)


def annualized_log_rv_from_returns(
    returns: NDArray[np.float64], pit_idx: int, vol_window: int
) -> float:
    """
    log(annualized realized vol) over [pit_idx - vol_window + 1, pit_idx] inclusive.

    Annualization: sqrt(252/vol_window * sum(r^2)) (daily simple returns).
    """
    r = returns[pit_idx - vol_window + 1 : pit_idx + 1]
    rv = np.sqrt(252.0 / vol_window * float(np.sum(r**2)))
    return float(np.log(max(rv, 1e-12)))

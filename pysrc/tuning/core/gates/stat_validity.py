"""Statistical validity gate: DSR, PBO, and Harvey t-stat checks."""

from __future__ import annotations

import math

try:
    from scipy.special import erfinv as _erfinv  # type: ignore[import-untyped]

    def _erfinv_fn(x: float) -> float:
        return float(_erfinv(x))
except ImportError:  # pragma: no cover

    def _erfinv_fn(x: float) -> float:
        raise RuntimeError("scipy is required for deflated_sharpe_ratio; install scipy")


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio correcting for multiple testing (Bailey & López de Prado 2014)."""
    gamma = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if n_trials > 1:
        e_max = (1.0 - 0.5772156649) * _erfinv_fn(1.0 - 1.0 / n_trials) + 0.5772156649 * _erfinv_fn(
            1.0 - 1.0 / (n_trials * math.e)
        )
    else:
        e_max = 0.0
    denom = math.sqrt(max(gamma, 0.0) / n_obs) if n_obs > 0 else 1e-9
    return (sharpe - e_max) / denom


def passes_dsr_gate(dsr: float, threshold: float = 0.0) -> bool:
    """Return True if DSR exceeds threshold."""
    return dsr > threshold


def passes_harvey_tstat(t_stat: float, min_t: float = 3.0) -> bool:
    """Return True if Harvey t-statistic meets minimum threshold."""
    return t_stat >= min_t


__all__ = ["deflated_sharpe_ratio", "passes_dsr_gate", "passes_harvey_tstat"]

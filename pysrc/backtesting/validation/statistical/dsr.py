# py/backtesting/validation/statistical/dsr.py
"""
Deflated Sharpe Ratio (DSR), Minimum Track Record Length (minTRL),
and block-bootstrap Sharpe confidence intervals.

References:
    Bailey & López de Prado (2014) "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting, and Non-Normality"
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr  # standard normal CDF — faster than stats.norm.cdf

from pysrc.core.errors import BaseError
from pysrc.core.runtime.optional_imports import optional_import
from pysrc.ops.mm_logkit import get_logger
from pysrc.ops.observability import instrument

LOG = get_logger(__name__)

# Optional Polars
pl = optional_import("polars")


# ---------------------------------------------------------------------------
# Exception hierarchy (matches stat_tests.py pattern)
# ---------------------------------------------------------------------------


class DSRError(BaseError):
    """Base for all DSR/validity computation errors."""


class DSRDataError(DSRError):
    """Input data is structurally invalid for DSR computation."""


class DSRComputationError(DSRError):
    """Numerical computation failed during DSR/bootstrap."""


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _to_returns(x: Any) -> np.ndarray:
    """Coerce various array-likes to a clean 1-D float64 numpy array."""
    if isinstance(x, pd.Series) or pl and isinstance(x, pl.Series):
        arr = x.to_numpy(dtype=np.float64)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise DSRDataError(
                "DataFrame input must have exactly one column",
                details={"n_cols": x.shape[1]},
            )
        arr = x.iloc[:, 0].to_numpy(dtype=np.float64)
    elif isinstance(x, np.ndarray):
        arr = x.astype(np.float64).ravel()
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=np.float64)
    else:
        raise DSRDataError(
            "Unsupported input type for returns",
            details={"type": type(x).__name__},
        )

    arr = arr[np.isfinite(arr)]  # strip NaN / Inf
    if arr.size < 10:
        raise DSRDataError(
            "Insufficient observations after removing non-finite values",
            details={"n_obs": arr.size, "min_required": 10},
        )
    return arr


def _annualised_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualised Sharpe Ratio from a return array."""
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-12:
        raise DSRComputationError(
            "Return series has near-zero volatility; Sharpe undefined",
            details={"std": float(sigma)},
        )
    return float(mu / sigma * math.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def compute_dsr(
    returns: Any,
    n_trials: int = 1,
    periods_per_year: int = 252,
    benchmark_sr: float = 0.0,
) -> dict[str, Any]:
    """
    Compute the Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    The DSR corrects the observed SR for:
      - Selection bias across n_trials strategy variants
      - Non-normality (skewness + excess kurtosis of returns)

    Args:
        returns:          Daily (or per-period) return series.
        n_trials:         Number of strategy variants evaluated before selecting
                          this one. DSR = SR when n_trials = 1 (no correction).
                          Pass the actual trial count for honest reporting.
        periods_per_year: Trading periods per year (252 for daily, 52 weekly, 12 monthly).
        benchmark_sr:     Minimum acceptable SR threshold (default 0.0).

    Returns:
        Dict with keys: sharpe_ratio, dsr, p_value, n_trials, skewness,
        excess_kurtosis, gate_result ("PASS" / "FAIL").

    Gate rule (Appendix H): FAIL if p_value > 0.05.
    """
    arr = _to_returns(returns)
    n = arr.size

    if n_trials < 1:
        raise DSRDataError("n_trials must be >= 1", details={"n_trials": n_trials})

    if n_trials == 1:
        LOG.warning(
            "dsr_n_trials_is_one",
            note="DSR is unadjusted for multiple testing. Pass n_trials > 1 for honest correction.",
        )

    # --- Observed Sharpe (annualised) ---
    sr_obs = _annualised_sharpe(arr, periods_per_year)

    # --- Higher moments ---
    skew = float(stats.skew(arr))
    kurt_excess = float(stats.kurtosis(arr, fisher=True))  # excess kurtosis

    # --- Expected maximum SR under H0 (from n_trials independent tests) ---
    # E[max SR] ≈ (1 - γ·E_Z) · Φ⁻¹(1 - 1/n_trials) + γ·φ(Φ⁻¹(1 - 1/n_trials))
    # Simplified form used by Bailey & LdP (2014), equation (8):
    if n_trials == 1:
        sr_expected_max = benchmark_sr
    else:
        euler_gamma = 0.5772156649
        z = stats.norm.ppf(1.0 - 1.0 / n_trials)
        phi_z = stats.norm.pdf(z)
        sr_expected_max = (1.0 - euler_gamma) * z + euler_gamma * phi_z

    # --- DSR statistic (z-score of SR vs expected maximum) ---
    # Variance of SR estimator (Mertens 2002 / Christie 2005):
    # Var(SR_hat) ≈ (1/T) * (1 + 0.5*SR² - skew*SR + (kurt_excess/4)*SR²)  [per-period]
    # Annualised: multiply by periods_per_year, then sqrt for std
    sr_per_period = sr_obs / math.sqrt(periods_per_year)
    var_sr = (1.0 / n) * (
        1.0 + 0.5 * sr_per_period**2 - skew * sr_per_period + (kurt_excess / 4.0) * sr_per_period**2
    )
    if var_sr <= 0:
        raise DSRComputationError(
            "SR variance is non-positive; cannot compute DSR",
            details={"var_sr": var_sr, "n": n, "sr_per_period": sr_per_period},
        )
    std_sr = math.sqrt(var_sr * periods_per_year)  # back to annualised scale

    dsr_z = (sr_obs - sr_expected_max) / std_sr
    # P(observed SR or better under null): low p-value => significant => PASS
    p_value = float(1.0 - ndtr(dsr_z))

    gate = "PASS" if p_value <= 0.05 else "FAIL"

    LOG.info(
        "dsr_computed",
        sharpe_ratio=round(sr_obs, 4),
        dsr=round(dsr_z, 4),
        p_value=round(p_value, 4),
        n_trials=n_trials,
        gate=gate,
    )

    return {
        "sharpe_ratio": round(sr_obs, 6),
        "dsr": round(dsr_z, 6),
        "p_value": round(p_value, 6),
        "n_trials": n_trials,
        "skewness": round(skew, 6),
        "excess_kurtosis": round(kurt_excess, 6),
        "gate_result": gate,
    }


# ---------------------------------------------------------------------------
# Minimum Track Record Length
# ---------------------------------------------------------------------------


def compute_min_trl(
    returns: Any,
    target_confidence: float = 0.95,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Compute the Minimum Track Record Length (minTRL) for the observed SR
    to be statistically significant at the given confidence level.

    Args:
        returns:           Return series.
        target_confidence: Confidence level (0.95 = 95% CI).
        benchmark_sr:      Minimum acceptable SR (annualised). Default 0.0.
        periods_per_year:  252 for daily, 52 weekly, 12 monthly.

    Returns:
        Dict with keys: years_needed, years_available, observed_sr,
        target_confidence, gate_result ("PASS" / "WARN").

    Gate rule (Appendix H): WARN if years_available < years_needed.
    """
    arr = _to_returns(returns)
    n = arr.size
    years_available = n / periods_per_year

    sr_obs = _annualised_sharpe(arr, periods_per_year)
    skew = float(stats.skew(arr))
    kurt_excess = float(stats.kurtosis(arr, fisher=True))

    z_conf = stats.norm.ppf(target_confidence)

    # minTRL formula (Bailey & LdP 2014, equation (13)):
    # n* = (z_α / (SR_obs - SR_min))² * (1 - skew·SR_obs + (kurt/4)·SR_obs²) * periods_per_year
    sr_per_period = sr_obs / math.sqrt(periods_per_year)
    benchmark_sr_per_period = benchmark_sr / math.sqrt(periods_per_year)
    sr_excess = sr_per_period - benchmark_sr_per_period

    if abs(sr_excess) < 1e-10:
        # SR equals benchmark exactly — minTRL is infinite
        years_needed = float("inf")
    else:
        moment_correction = 1.0 - skew * sr_per_period + (kurt_excess / 4.0) * sr_per_period**2
        moment_correction = max(moment_correction, 0.01)  # numerical floor
        n_star = (z_conf / sr_excess) ** 2 * moment_correction
        years_needed = n_star / periods_per_year

    gate = "PASS" if years_available >= years_needed else "WARN"

    LOG.info(
        "min_trl_computed",
        years_needed=round(years_needed, 2) if math.isfinite(years_needed) else "inf",
        years_available=round(years_available, 2),
        observed_sr=round(sr_obs, 4),
        gate=gate,
    )

    return {
        "observed_sr": round(sr_obs, 6),
        "years_needed": round(years_needed, 4) if math.isfinite(years_needed) else None,
        "years_available": round(years_available, 4),
        "target_confidence": target_confidence,
        "gate_result": gate,
    }


# ---------------------------------------------------------------------------
# Block Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------


@instrument(name="bootstrap_sharpe_ci", measure_latency=True)
def compute_bootstrap_ci(
    returns: Any,
    n_resamples: int = 10_000,
    block_size: int | None = None,
    periods_per_year: int = 252,
    ci_levels: tuple[float, ...] = (0.95, 0.99),
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Block-bootstrap confidence intervals for the annualised Sharpe Ratio.

    Block bootstrap preserves the autocorrelation structure of returns,
    which is essential for financial return series.

    Args:
        returns:         Return series.
        n_resamples:     Bootstrap replications (default 10,000 per Appendix H).
        block_size:      Block length. Defaults to ceil(n^(1/3)) — optimal for
                         stationary series (Politis & Romano 1994).
        periods_per_year: 252 daily / 52 weekly / 12 monthly.
        ci_levels:       Confidence levels for intervals (default 95% and 99%).
        random_state:    Seed for reproducibility.

    Returns:
        Dict with CI bounds, n_resamples, block_size, and gate_result.

    Gate rules (Appendix H):
        WARN if 95% CI lower bound < 0.
        FAIL if 95% CI lower bound < -0.5.
    """
    arr = _to_returns(returns)
    n = arr.size

    # Default block size: cube-root rule
    if block_size is None:
        block_size = max(1, int(math.ceil(n ** (1.0 / 3.0))))

    if block_size >= n:
        raise DSRDataError(
            "block_size must be smaller than the number of observations",
            details={"block_size": block_size, "n_obs": n},
        )

    rng = np.random.default_rng(random_state)

    # --- Build block indices ---
    n_blocks = math.ceil(n / block_size)
    max_start = n - block_size  # last valid block start index

    # Pre-compute all block starts for all resamples at once (vectorised)
    all_starts = rng.integers(0, max_start + 1, size=(n_resamples, n_blocks))

    # --- Resample and compute SR for each bootstrap sample ---
    boot_srs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        # Concatenate blocks then trim to n
        indices = np.concatenate([np.arange(s, s + block_size) for s in all_starts[i]])[:n]
        boot_returns = arr[indices]
        mu = boot_returns.mean()
        sigma = boot_returns.std(ddof=1)
        if sigma < 1e-12:
            boot_srs[i] = 0.0
        else:
            boot_srs[i] = mu / sigma * math.sqrt(periods_per_year)

    # --- Compute percentile CIs ---
    ci_results: dict[str, Any] = {}
    gate = "PASS"

    for level in ci_levels:
        alpha = 1.0 - level
        lower = float(np.percentile(boot_srs, 100 * alpha / 2))
        upper = float(np.percentile(boot_srs, 100 * (1 - alpha / 2)))
        pct = int(round(level * 100))
        ci_results[f"lower_{pct}"] = round(lower, 6)
        ci_results[f"upper_{pct}"] = round(upper, 6)

        # Gate logic on 95% CI only (Appendix H)
        if pct == 95:
            if lower < -0.5:
                gate = "FAIL"
            elif lower < 0.0 and gate != "FAIL":
                gate = "WARN"

    LOG.info(
        "bootstrap_ci_computed",
        n_resamples=n_resamples,
        block_size=block_size,
        gate=gate,
        **{k: round(v, 4) for k, v in ci_results.items()},
    )

    return {
        **ci_results,
        "n_resamples": n_resamples,
        "block_size": block_size,
        "gate_result": gate,
    }

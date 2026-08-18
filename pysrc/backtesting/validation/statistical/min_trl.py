"""
py/backtesting/validation/statistical/min_trl.py

Standalone public API for the Minimum Track Record Length (minTRL)
calculation.  The core formula lives in dsr.py; this module re-exports
it under a stable name and adds a typed result wrapper so callers can
rely on attribute access rather than dict key lookups.

Reference: Bailey & López de Prado (2014), equation (13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pysrc.backtesting.validation.statistical.dsr import compute_min_trl as _compute_min_trl


@dataclass
class MinTRLResult:
    """Typed wrapper around the minTRL computation result.

    Attributes:
        years_needed:       Track record length required for significance.
        years_available:    Actual track record length in the sample.
        observed_sr:        Annualised Sharpe ratio of the observed series.
        target_confidence:  Confidence level used (e.g. 0.95).
        gate_result:        ``"PASS"`` if ``years_available >= years_needed``,
                            otherwise ``"WARN"``.
    """

    years_needed: float
    years_available: float
    observed_sr: float
    target_confidence: float
    gate_result: str  # "PASS" | "WARN"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MinTRLResult:
        """Construct from the raw dict returned by ``compute_min_trl``."""
        return cls(
            years_needed=d["years_needed"],
            years_available=d["years_available"],
            observed_sr=d.get("observed_sr", float("nan")),
            target_confidence=d["target_confidence"],
            gate_result=d["gate_result"],
        )


def compute_min_trl(
    returns: Any,
    *,
    target_confidence: float = 0.95,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> MinTRLResult:
    """Compute the Minimum Track Record Length and return a typed result.

    Delegates to the underlying implementation in ``dsr.compute_min_trl``
    and wraps the result in :class:`MinTRLResult` for typed access.

    Args:
        returns:            Return series (array-like or Polars Series).
        target_confidence:  Confidence level (default 0.95 = 95 %).
        benchmark_sr:       Minimum acceptable annualised SR (default 0.0).
        periods_per_year:   252 daily / 52 weekly / 12 monthly.

    Returns:
        :class:`MinTRLResult` with ``gate_result`` set to ``"PASS"`` or
        ``"WARN"``.
    """
    raw = _compute_min_trl(
        returns,
        target_confidence=target_confidence,
        benchmark_sr=benchmark_sr,
        periods_per_year=periods_per_year,
    )
    return MinTRLResult.from_dict(raw)

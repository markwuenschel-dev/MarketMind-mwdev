"""Cost-stress testing: apply transaction-cost multipliers to returns."""

from __future__ import annotations

import pandas as pd


def apply_cost_stress(
    returns: pd.Series,
    positions: pd.DataFrame,
    stress_bps: float,
) -> pd.Series:
    """Subtract transaction costs (in bps) from a returns series."""
    cost_per_unit = stress_bps / 10_000.0
    turnover = positions.diff().abs().sum(axis=1)
    costs = turnover * cost_per_unit
    return returns - costs


__all__ = ["apply_cost_stress"]

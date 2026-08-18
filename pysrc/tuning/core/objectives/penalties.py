"""Penalty functions applied to objective scores for constraint satisfaction."""

from __future__ import annotations

import pandas as pd


def turnover_penalty(returns: pd.Series, positions: pd.DataFrame, cap: float) -> float:
    """Return 0 if turnover <= cap, else a negative penalty proportional to excess."""
    from pysrc.tuning.core.objectives.metrics import turnover_ratio

    to = turnover_ratio(positions)
    return 0.0 if to <= cap else -(to - cap)


def drawdown_penalty(returns: pd.Series, max_allowed: float) -> float:
    """Return 0 if drawdown is within limit, else a negative penalty."""
    from pysrc.tuning.core.objectives.metrics import max_drawdown

    mdd = max_drawdown(returns)
    return 0.0 if mdd >= max_allowed else (mdd - max_allowed)


__all__ = ["turnover_penalty", "drawdown_penalty"]

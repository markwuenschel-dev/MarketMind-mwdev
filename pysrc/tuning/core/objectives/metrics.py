"""Core financial metrics used in objective scoring."""

from __future__ import annotations

import math

import pandas as pd


def sharpe_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio; returns NaN if std is zero."""
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma == 0 or math.isnan(sigma):
        return float("nan")
    return mu / sigma * math.sqrt(ann_factor)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative float)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def turnover_ratio(positions: pd.DataFrame) -> float:
    """Mean absolute position change per period, normalised by total abs positions."""
    delta = positions.diff().abs().sum(axis=1)
    total = positions.abs().sum(axis=1).replace(0, float("nan"))
    return float(delta.div(total).mean())


def calmar_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualised return divided by absolute max drawdown."""
    ann_ret = float(returns.mean()) * ann_factor
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("nan")
    return ann_ret / mdd


__all__ = ["sharpe_ratio", "max_drawdown", "turnover_ratio", "calmar_ratio"]

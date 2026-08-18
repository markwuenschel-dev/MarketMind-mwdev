"""Pure label-generation logic for supervised tuning tasks."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

__all__ = ["LabelType", "forward_return_labels", "sign_labels", "excess_return_labels"]

LabelType = Literal["forward_return", "sign", "quantile", "excess_return"]


def forward_return_labels(
    prices: pd.Series[float],
    horizon: int,
) -> pd.Series[float]:
    """Compute forward log-returns over *horizon* bars; NaN at tail."""
    return cast("pd.Series[float]", np.log(prices.shift(-horizon) / prices))


def sign_labels(
    prices: pd.Series[float],
    horizon: int,
) -> pd.Series[float]:
    """Return +1/-1 sign of forward log-return."""
    fwd = forward_return_labels(prices, horizon)
    return cast("pd.Series[float]", np.sign(fwd))


def excess_return_labels(
    prices: pd.Series[float],
    benchmark: pd.Series[float],
    horizon: int,
) -> pd.Series[float]:
    """Compute excess forward log-return over a benchmark."""
    asset = forward_return_labels(prices, horizon)
    bench = forward_return_labels(benchmark, horizon)
    return asset - bench

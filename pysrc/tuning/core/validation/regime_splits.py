"""Regime-conditioned splits: group fold observations by detected regimes."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from pysrc.tuning.core.tasks.regime_projection import RegimeSegment


def regime_conditioned_splits(
    index: pd.DatetimeIndex,
    segments: tuple[RegimeSegment, ...],
    bar_duration: timedelta,
) -> dict[str, pd.DatetimeIndex]:
    """Return a dict mapping regime label -> DatetimeIndex of observations."""
    result: dict[str, list[pd.Timestamp]] = {}
    for ts in index:
        dt = ts.to_pydatetime()
        for seg in segments:
            if seg.start <= dt < seg.end:
                result.setdefault(seg.label, []).append(ts)
                break
    return {k: pd.DatetimeIndex(v) for k, v in result.items()}


__all__ = ["regime_conditioned_splits"]

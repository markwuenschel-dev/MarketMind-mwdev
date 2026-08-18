"""Embargo logic: enforce a gap between training and test periods."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def apply_embargo(
    train_end: datetime,
    embargo_periods: int,
    bar_duration: timedelta,
) -> datetime:
    """Return the earliest safe test-start date after an embargo gap."""
    return train_end + embargo_periods * bar_duration


def trim_to_embargo(
    index: pd.DatetimeIndex,
    safe_start: datetime,
) -> pd.DatetimeIndex:
    """Drop index entries before safe_start."""
    return index[index >= pd.Timestamp(safe_start)]


__all__ = ["apply_embargo", "trim_to_embargo"]

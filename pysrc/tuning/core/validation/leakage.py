"""Point-in-time leakage checks for validation splits."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


class LeakageDetectedError(ValueError):
    """Raised when future information is detected in a training split."""


def assert_no_future_in_train(
    train_idx: pd.DatetimeIndex,
    test_start: datetime,
) -> None:
    """Raise LeakageDetectedError if any training timestamp is >= test_start."""
    leaking = train_idx[train_idx >= pd.Timestamp(test_start)]
    if len(leaking):
        raise LeakageDetectedError(
            f"{len(leaking)} training observations on or after test_start {test_start}"
        )


__all__ = ["LeakageDetectedError", "assert_no_future_in_train"]

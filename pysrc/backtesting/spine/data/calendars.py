from __future__ import annotations

from datetime import datetime


def is_trading_day(ts: datetime) -> bool:
    return ts.weekday() < 5

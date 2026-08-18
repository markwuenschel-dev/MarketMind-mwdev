from __future__ import annotations

from datetime import datetime
from typing import Any


class InvariantViolation(ValueError):
    pass


def assert_fill_timestamps_within_window(
    fills: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> None:
    for fill in fills:
        timestamp = datetime.fromisoformat(str(fill["timestamp"]).replace("Z", "+00:00"))
        if timestamp < start or timestamp > end:
            raise InvariantViolation("Fill timestamp falls outside the allowed window.")


def assert_position_cash_directionality(
    before_cash: float, after_cash: float, fills: list[dict[str, Any]]
) -> None:
    net_cash_change = sum(
        (-1 if str(fill.get("side", "BUY")).upper() == "BUY" else 1)
        * float(fill.get("quantity", 0.0))
        * float(fill.get("price", 0.0))
        for fill in fills
    )
    actual_change = after_cash - before_cash
    if round(actual_change, 8) != round(net_cash_change, 8):
        raise InvariantViolation("Cash movement does not match fill directionality.")

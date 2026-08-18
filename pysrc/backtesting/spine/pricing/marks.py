from __future__ import annotations

from pysrc.backtesting.contracts.types import LedgerSnapshot


def mark_to_market(snapshot: LedgerSnapshot, prices: dict[str, float]) -> float:
    position_value = sum(
        position.quantity * prices.get(position.symbol, position.average_price)
        for position in snapshot.positions
    )
    return snapshot.cash + position_value

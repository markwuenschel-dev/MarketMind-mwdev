from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pysrc.backtesting.contracts.registry import register_ledger
from pysrc.backtesting.contracts.types import LedgerPosition, LedgerSnapshot
from pysrc.backtesting.spine.ledger.positions import CashBook, PositionBook, PositionLot


@dataclass
class SimpleLedger:
    opening_cash: float = 0.0

    def apply(
        self,
        fills: list[dict[str, Any]],
        corporate_actions: list[dict[str, Any]] | None,
    ) -> LedgerSnapshot:
        cash = CashBook(balance=self.opening_cash)
        positions = PositionBook()
        timestamp = "1970-01-01T00:00:00+00:00"

        for fill in fills:
            quantity = float(fill.get("quantity", 0.0))
            price = float(fill.get("price", 0.0))
            side = str(fill.get("side", "BUY")).upper()
            symbol = str(fill.get("symbol", "UNKNOWN"))
            timestamp = str(fill.get("timestamp", timestamp))
            signed_quantity = quantity if side == "BUY" else -quantity
            prior = positions.lots.get(
                symbol, PositionLot(symbol=symbol, quantity=0.0, average_price=price)
            )
            new_quantity = prior.quantity + signed_quantity
            if side == "BUY":
                cash = CashBook(balance=cash.balance - (quantity * price))
                avg_price = (
                    ((prior.quantity * prior.average_price) + (quantity * price)) / new_quantity
                    if new_quantity != 0.0
                    else price
                )
            else:
                cash = CashBook(balance=cash.balance + (quantity * price))
                avg_price = prior.average_price
            positions.lots[symbol] = PositionLot(
                symbol=symbol, quantity=new_quantity, average_price=avg_price
            )

        serialized_positions = [
            LedgerPosition(
                symbol=lot.symbol, quantity=lot.quantity, average_price=lot.average_price
            )
            for lot in positions.lots.values()
        ]
        return LedgerSnapshot(
            timestamp=timestamp, cash=cash.balance, positions=serialized_positions
        )


register_ledger("ledger.simple", lambda: SimpleLedger())

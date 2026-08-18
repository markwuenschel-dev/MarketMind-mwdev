"""Ledger vs broker reconciliation schema (PDR-003 Wave 3 stub)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PositionDiff:
    symbol: str
    ledger_qty: float
    broker_qty: float
    delta: float


@dataclass(frozen=True, slots=True)
class CashDiff:
    ledger_cash: float
    broker_cash: float
    delta: float


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    """Canonical diff payload for internal ledger vs broker state."""

    schema_version: str
    as_of_bar: str
    position_diffs: tuple[PositionDiff, ...]
    cash_diff: CashDiff | None
    has_mismatch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of_bar": self.as_of_bar,
            "has_mismatch": self.has_mismatch,
            "position_diffs": [
                {
                    "symbol": item.symbol,
                    "ledger_qty": item.ledger_qty,
                    "broker_qty": item.broker_qty,
                    "delta": item.delta,
                }
                for item in self.position_diffs
            ],
            "cash_diff": (
                {
                    "ledger_cash": self.cash_diff.ledger_cash,
                    "broker_cash": self.cash_diff.broker_cash,
                    "delta": self.cash_diff.delta,
                }
                if self.cash_diff is not None
                else None
            ),
        }


def compare_ledger_to_broker(
    *,
    ledger_positions: Mapping[str, float],
    broker_positions: Mapping[str, float],
    ledger_cash: float,
    broker_cash: float,
    as_of_bar: str,
    tolerance: float = 1e-6,
) -> ReconciliationDiff:
    """Compare ledger and broker snapshots; flag mismatches beyond ``tolerance``."""

    symbols = sorted(set(ledger_positions) | set(broker_positions))
    position_diffs: list[PositionDiff] = []
    for symbol in symbols:
        ledger_qty = float(ledger_positions.get(symbol, 0.0))
        broker_qty = float(broker_positions.get(symbol, 0.0))
        delta = ledger_qty - broker_qty
        if abs(delta) > tolerance:
            position_diffs.append(
                PositionDiff(
                    symbol=symbol,
                    ledger_qty=ledger_qty,
                    broker_qty=broker_qty,
                    delta=delta,
                )
            )
    cash_delta = ledger_cash - broker_cash
    cash_diff = CashDiff(ledger_cash=ledger_cash, broker_cash=broker_cash, delta=cash_delta)
    has_mismatch = bool(position_diffs) or abs(cash_delta) > tolerance
    return ReconciliationDiff(
        schema_version="reconciliation_diff.v1",
        as_of_bar=as_of_bar,
        position_diffs=tuple(position_diffs),
        cash_diff=cash_diff,
        has_mismatch=has_mismatch,
    )


__all__ = ["CashDiff", "PositionDiff", "ReconciliationDiff", "compare_ledger_to_broker"]

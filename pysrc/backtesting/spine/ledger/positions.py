from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PositionLot:
    symbol: str
    quantity: float
    average_price: float


@dataclass(frozen=True)
class CashBook:
    balance: float = 0.0


@dataclass(frozen=True)
class PositionBook:
    lots: dict[str, PositionLot] = field(default_factory=dict)

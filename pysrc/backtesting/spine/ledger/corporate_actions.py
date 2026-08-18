from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitAction:
    symbol: str
    ratio: float


@dataclass(frozen=True)
class DividendAction:
    symbol: str
    amount_per_share: float


@dataclass(frozen=True)
class SymbolChangeAction:
    old_symbol: str
    new_symbol: str

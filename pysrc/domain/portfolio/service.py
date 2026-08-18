"""Portfolio domain service.

Typed contract for portfolio construction and rebalancing operations.
Integrates with risk service for constraint validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd


@dataclass(frozen=True, slots=True)
class Position:
    """Immutable position representation.

    Attributes:
        symbol: Ticker symbol
        quantity: Position size (negative for short)
        entry_price: Weighted average entry price
    """

    symbol: str
    quantity: Decimal
    entry_price: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state.

    Attributes:
        timestamp: Snapshot timestamp
        positions: Mapping of symbol to position
        cash: Available cash balance
        base_currency: Portfolio denomination currency
    """

    timestamp: pd.Timestamp
    positions: Mapping[str, Position]
    cash: Decimal
    base_currency: str


class ConstraintValidator(Protocol):
    """Protocol for portfolio constraint validation.

    TODO: Integrate with risk service for unified constraint checking.
    """

    def validate(self, snapshot: PortfolioSnapshot, proposed: Mapping[str, Decimal]) -> bool:
        """Validate proposed allocation against constraints."""
        ...


class PortfolioService(ABC):
    """Abstract portfolio service contract.

    Implementations handle portfolio construction, rebalancing,
    and allocation logic with deterministic outcomes.

    TODO: Registry hook for strategy-specific portfolio implementations.
    TODO: Factory integration for service instantiation.
    """

    @abstractmethod
    def construct(
        self,
        signals: pd.DataFrame,
        capital: Decimal,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> PortfolioSnapshot:
        """Construct portfolio from signal dataframe.

        Args:
            signals: Signal strength dataframe with symbol index
            capital: Available capital for allocation
            as_of: Point-in-time boundary for prices

        Returns:
            Constructed portfolio snapshot
        """
        ...

    @abstractmethod
    def rebalance(
        self,
        current: PortfolioSnapshot,
        target_weights: Mapping[str, Decimal],
        *,
        constraints: Sequence[ConstraintValidator] | None = None,
    ) -> PortfolioSnapshot:
        """Rebalance portfolio to target weights.

        Args:
            current: Current portfolio state
            target_weights: Target allocation weights (sum to 1.0)
            constraints: Optional sequence of constraint validators

        Returns:
            Rebalanced portfolio snapshot
        """
        ...

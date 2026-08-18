"""Risk domain service.

Typed contract for risk measurement and limit validation.
Integrates with portfolio service for exposure calculation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from pysrc.domain.portfolio.service import PortfolioSnapshot


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    """Immutable risk measurement container.

    Attributes:
        var_95: 95% Value at Risk
        cvar_95: 95% Conditional VaR (expected shortfall)
        max_drawdown: Maximum drawdown percentage
        beta: Market beta
        volatility: Annualized volatility
    """

    var_95: Decimal
    cvar_95: Decimal
    max_drawdown: Decimal
    beta: Decimal
    volatility: Decimal


@dataclass(frozen=True, slots=True)
class RiskLimit:
    """Risk limit definition.

    Attributes:
        metric: Risk metric name
        threshold: Limit threshold
        hard: If True, breach raises exception; else warning
    """

    metric: str
    threshold: Decimal
    hard: bool = True


class RiskService(ABC):
    """Abstract risk service contract.

    Implementations handle risk calculation with statistical
    rigor and point-in-time data boundaries.

    TODO: Registry hook for risk model selection.
    TODO: Integration with gate validation for limit enforcement.
    """

    @abstractmethod
    def calculate(
        self, portfolio: PortfolioSnapshot, *, lookback: pd.Timedelta | None = None
    ) -> RiskMetrics:
        """Calculate risk metrics for portfolio.

        Args:
            portfolio: Current portfolio snapshot
            lookback: Historical window for calculation

        Returns:
            Computed risk metrics
        """
        ...

    @abstractmethod
    def validate_limits(
        self,
        portfolio: PortfolioSnapshot,
        limits: Sequence[RiskLimit],
        *,
        as_of: pd.Timestamp | None = None,
    ) -> Mapping[str, bool]:
        """Validate portfolio against risk limits.

        Args:
            portfolio: Portfolio to validate
            limits: Sequence of risk limits to check
            as_of: Point-in-time boundary for data

        Returns:
            Mapping of metric name to pass/fail status
        """
        ...

    @abstractmethod
    def exposure(
        self,
        portfolio: PortfolioSnapshot,
        *,
        by_sector: bool = False,
        by_factor: bool = False,
    ) -> pd.DataFrame:
        """Calculate portfolio exposure breakdown.

        Args:
            portfolio: Portfolio to analyze
            by_sector: Include sector aggregation
            by_factor: Include factor exposure

        Returns:
            Exposure dataframe with breakdown dimensions
        """
        ...

"""Backtesting public API.

Typed contracts for backtest execution, result retrieval,
and analysis with deterministic outcomes and artifact tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from decimal import Decimal
    from pathlib import Path
    from typing import Any

    import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    """Backtest execution specification.

    Attributes:
        strategy_id: Strategy to backtest
        start_date: Backtest start date (inclusive)
        end_date: Backtest end date (inclusive)
        initial_capital: Starting capital amount
        config_overrides: Optional parameter overrides
    """

    strategy_id: str
    start_date: str
    end_date: str
    initial_capital: Decimal
    config_overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest execution result.

    Attributes:
        spec: Original backtest specification
        equity_curve: Daily equity values
        trades: Executed trade records
        metrics: Performance metrics dictionary
        artifact_path: Path to stored result artifact
    """

    spec: BacktestSpec
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Mapping[str, float]
    artifact_path: Path


class BacktestAPI(ABC):
    """Abstract backtest API coordinator.

    Implementations provide deterministic backtest execution
    with mechanical and statistical validation.

    TODO: Registry hook for engine selection.
    TODO: Factory integration for API instantiation.
    """

    @abstractmethod
    def run(self, spec: BacktestSpec) -> BacktestResult:
        """Execute backtest for given specification.

        Args:
            spec: Backtest execution specification

        Returns:
            Complete backtest result with metrics
        """
        ...

    @abstractmethod
    def load(self, run_id: str) -> BacktestResult:
        """Load historical backtest result from artifact.

        Args:
            run_id: Artifact registry run identifier

        Returns:
            Reconstructed backtest result
        """
        ...

    @abstractmethod
    def compare(
        self,
        results: Sequence[BacktestResult],
        *,
        benchmark: str | None = None,
    ) -> pd.DataFrame:
        """Compare multiple backtest results.

        Args:
            results: Backtest results to compare
            benchmark: Optional benchmark ticker for relative metrics

        Returns:
            Comparison dataframe with aligned metrics
        """
        ...

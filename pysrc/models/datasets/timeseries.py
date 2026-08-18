"""Time-series dataset construction for ML training.

Provides typed contracts for building training/validation datasets
from point-in-time DataViews with deterministic windowing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from typing import Protocol

    import pandas as pd


T = TypeVar("T")


class SampleWindow(Protocol):
    """Protocol defining a time-series sample window contract."""

    lookback: int
    horizon: int
    step: int


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Immutable window configuration for time-series sampling.

    Attributes:
        lookback: Historical lookback window size
        horizon: Forward prediction horizon
        step: Stride between consecutive samples
    """

    lookback: int
    horizon: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")


class DatasetBuilder[T](ABC):
    """Abstract base for time-series dataset builders.

    TODO: Registry hook for builder selection based on model type.
    TODO: Planner integration for batch size and memory estimation.
    """

    @abstractmethod
    def build(
        self,
        data: pd.DataFrame,
        config: WindowConfig,
        *,
        as_of: str | None = None,
    ) -> T:
        """Build dataset from point-in-time data.

        Args:
            data: Input DataFrame with DatetimeIndex
            config: Window configuration for sampling
            as_of: Optional point-in-time boundary for leakage prevention

        Returns:
            Model-specific dataset representation
        """
        ...

"""Contracts for market-data source adapters.

New source adapters should import from this module instead of ``base.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import polars as pl


class DataSource(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Fetch historical market data."""

    @abstractmethod
    async def get_realtime(
        self,
        symbol: str,
        *,
        interval: float = 60.0,
    ) -> AsyncIterator[pl.DataFrame]:
        """Stream real-time market data."""

    async def __aenter__(self) -> DataSource:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        _ = (exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        """Release any source-owned resources."""

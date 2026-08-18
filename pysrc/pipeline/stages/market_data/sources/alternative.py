"""
py/pipeline/stages/market_data/sources/alternative.py

Stubs for alternative-data sources: satellite imagery, weather, and
social-sentiment feeds.  Each class extends the base DataSource /
APIDataSource interfaces defined in sources/contracts.py and sources/runtime.pysrc.

These replace the classes that previously lived in py/data/alternative_data.pysrc.
Migrate any imports from that module to this one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import polars as pl

from pysrc.pipeline.stages.market_data.sources.contracts import DataSource
from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource


class AlternativeDataSource(DataSource):
    """Abstract base for all alternative-data sources.

    Sub-classes must implement ``get_historical`` and ``get_realtime``.
    """

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        raise NotImplementedError(f"{type(self).__name__}.get_historical() is not implemented.")

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        raise NotImplementedError(f"{type(self).__name__}.get_realtime() is not implemented.")
        # Satisfy the AsyncIterator type; never reached.
        yield  # type: ignore[misc]


class SatelliteDataSource(AlternativeDataSource, APIDataSource):
    """Stub for satellite imagery / crop / shipping data providers.

    Phase 0: raises NotImplementedError.  Replace with real API calls
    when a provider contract is established.
    """

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        raise NotImplementedError("SatelliteDataSource.get_historical() is not yet implemented.")

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        raise NotImplementedError("SatelliteDataSource.get_realtime() is not yet implemented.")
        yield  # type: ignore[misc]


class WeatherDataSource(AlternativeDataSource, APIDataSource):
    """Stub for weather / meteorological data providers.

    Phase 0: raises NotImplementedError.  Replaces ``WeatherSource``
    from the deleted py/data/alternative_data.pysrc.
    """

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        raise NotImplementedError("WeatherDataSource.get_historical() is not yet implemented.")

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        raise NotImplementedError("WeatherDataSource.get_realtime() is not yet implemented.")
        yield  # type: ignore[misc]


class SocialSentimentDataSource(AlternativeDataSource, APIDataSource):
    """Stub for social-media / news-sentiment data providers.

    Phase 0: raises NotImplementedError.  Replaces ``TwitterSource`` and
    the sentiment leg of ``ESGSource`` from the deleted module.
    """

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        raise NotImplementedError(
            "SocialSentimentDataSource.get_historical() is not yet implemented."
        )

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        raise NotImplementedError(
            "SocialSentimentDataSource.get_realtime() is not yet implemented."
        )
        yield  # type: ignore[misc]


__all__ = [
    "AlternativeDataSource",
    "SatelliteDataSource",
    "WeatherDataSource",
    "SocialSentimentDataSource",
]

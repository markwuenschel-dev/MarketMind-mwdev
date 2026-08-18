from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, date, datetime, time
from typing import Any, Protocol, runtime_checkable

import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

try:
    from fredapi import Fred  # type: ignore[import-untyped]
except ImportError:
    Fred = None  # type: ignore[assignment]

try:
    from .registry import register_source
except ImportError:

    def register_source(_name: str):
        def _wrap(cls):
            return cls

        return _wrap


def _utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


@runtime_checkable
class FREDVintageSeam(Protocol):
    def apply(self, frame: pl.DataFrame, *, retrieval_time: datetime) -> pl.DataFrame:
        """Attach vintage-aware valid_time and knowledge_time columns to a FRED frame."""


class FREDApproximationStub:
    """Phase I approximation: use observation date as valid_time and retrieval time as knowledge_time until ALFRED replaces this in Phase II."""

    def apply(self, frame: pl.DataFrame, *, retrieval_time: datetime) -> pl.DataFrame:
        return frame.with_columns(
            pl.col("date").cast(pl.Date).alias("valid_time"),
            pl.lit(retrieval_time).alias("knowledge_time"),
        )


@register_source("fred")
class FREDSource(DataSource):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        fred_client: Any | None = None,
        vintage_seam: FREDVintageSeam | None = None,
        retrieval_clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(config)
        if fred_client is not None:
            self.fred = fred_client
        elif Fred is not None:
            self.fred = Fred(api_key=config["api_key"])
        else:
            raise ImportError("fredapi is required unless a fred_client is provided")
        self.vintage_seam = vintage_seam or FREDApproximationStub()
        self._retrieval_clock = retrieval_clock or (lambda: datetime.now(UTC))

    def _normalize_series(
        self, symbol: str, series: Any, *, retrieval_time: datetime
    ) -> pl.DataFrame:
        if series.empty:
            raise DataFetchError(f"No historical data found for {symbol}")
        pd_df = series.rename_axis("date").reset_index(name="value")
        frame = pl.from_pandas(pd_df, rechunk=False).with_columns(
            pl.col("date").cast(pl.Date),
            pl.lit(symbol).alias("symbol"),
        )
        frame = self.vintage_seam.apply(frame, retrieval_time=retrieval_time)
        required_columns = ["symbol", "date", "value", "valid_time", "knowledge_time"]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise DataFetchError(f"FRED source is missing required columns: {missing}")
        return frame.select(required_columns).sort("date")

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        if not symbol.strip():
            raise DataFetchError("FRED source requires a non-empty symbol")
        try:
            start_date = date.fromisoformat(str(start)[:10])
            end_date = date.fromisoformat(str(end)[:10])
            if start_date > end_date:
                raise DataFetchError("FRED source requires start <= end")
            retrieval_time = _utc_midnight(self._retrieval_clock().astimezone(UTC).date())
            series = await asyncio.to_thread(
                self.fred.get_series,
                symbol.strip(),
                observation_start=start_date.isoformat(),
                observation_end=end_date.isoformat(),
            )
            frame = self._normalize_series(symbol.strip(), series, retrieval_time=retrieval_time)
            if eager:
                return frame
            return frame.lazy()
        except DataFetchError:
            raise
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            raise DataFetchError(f"Failed to fetch historical data for {symbol}: {exc}") from exc

    async def get_realtime(
        self, symbol: str, *, interval: float = 3600.0
    ) -> AsyncIterator[pl.DataFrame]:
        while True:
            try:
                latest = await asyncio.to_thread(self.fred.get_series_latest_release, symbol)
                retrieval_time = _utc_midnight(self._retrieval_clock().astimezone(UTC).date())
                yield pl.DataFrame(
                    {
                        "timestamp": [datetime.now(UTC)],
                        "symbol": [symbol],
                        "value": [latest.iloc[0]],
                        "valid_time": [retrieval_time.date()],
                        "knowledge_time": [retrieval_time],
                    }
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise DataFetchError(f"Failed to fetch real-time data for {symbol}: {exc}") from exc
            await asyncio.sleep(interval)

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

try:
    from pysrc.pipeline.core.pipeline_core_registry import register_source
except ImportError:

    def register_source(_name: str):
        def _noop(cls):
            return cls

        return _noop


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


@register_source("yahoo")
class YahooFinanceSource(DataSource):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        fetch_clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(config or {})
        self._fetch_clock = fetch_clock or (lambda: datetime.now(UTC))

    def _download_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf  # type: ignore[import-untyped]

        ticker = yf.Ticker(symbol)
        return ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

    def _normalize_frame(
        self,
        raw: pd.DataFrame,
        *,
        symbol: str,
        fetch_day: date,
    ) -> pl.DataFrame:
        if raw.empty:
            raise DataFetchError(f"No historical data found for {symbol}")

        pdf = raw.reset_index()
        pdf = pdf.rename(
            columns={
                "Date": "date",
                "index": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if "date" not in pdf.columns:
            raise DataFetchError("Yahoo fetcher response is missing a date column")

        frame = pl.from_pandas(pdf, include_index=False).with_columns(
            pl.col("date").cast(pl.Date),
            pl.lit(symbol).alias("symbol"),
            pl.col("date").cast(pl.Date).alias("valid_time"),
            pl.lit(_utc_midnight(fetch_day)).alias("knowledge_time"),
        )
        required_columns = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "valid_time",
            "knowledge_time",
        ]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise DataFetchError(f"Yahoo fetcher response is missing columns: {missing}")
        if frame.filter(pl.col("knowledge_time").dt.date() < pl.col("valid_time")).height > 0:
            raise DataFetchError("Yahoo fetcher produced rows where knowledge_time < valid_time")
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
            raise DataFetchError("Yahoo fetcher requires a non-empty symbol")
        start_date = _parse_iso_date(start)
        end_date = _parse_iso_date(end)
        if start_date > end_date:
            raise DataFetchError("Yahoo fetcher requires start <= end")

        fetch_day = self._fetch_clock().astimezone(UTC).date()
        raw = await asyncio.to_thread(self._download_history, symbol.strip(), start_date, end_date)
        frame = self._normalize_frame(raw, symbol=symbol.strip(), fetch_day=fetch_day)
        if eager:
            return frame
        return frame.lazy()

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        _ = (symbol, interval)
        raise NotImplementedError("YahooFinanceSource does not implement real-time streaming")

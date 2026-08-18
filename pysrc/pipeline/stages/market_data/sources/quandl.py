import asyncio
from collections.abc import AsyncIterator

import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource
from pysrc.pipeline.stages.market_data.sources.runtime import async_retry

# prefer the real registry; use specific ImportError handling only
try:
    from pysrc.pipeline.stages.market_data.sources.registry import register_source  # type: ignore
except ImportError:
    try:
        from pysrc.pipeline.stages.market_data.sources import register_source  # type: ignore
    except ImportError:
        # minimal no-op so smoke-imports don't fail if registry isn't wired yet
        def register_source(name: str):
            def deco(cls):
                cls.__source_name__ = name  # hint for downstream tooling

                return cls

            return deco


@register_source("quandl")
class QuandlSource(DataSource):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config["api_key"]
        self.base_url = "https://www.quandl.com/api/v3/datasets"

    @async_retry(attempts=3, multiplier=1, min_delay=2, max_delay=60)
    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        dataset_code = symbol.replace("/", "_")

        url = f"{self.base_url}/{dataset_code}/data.json"
        params = {"api_key": self.api_key, "start_date": start, "end_date": end}
        data = await self._request(url, params=params)
        if "dataset_data" not in data or not data["dataset_data"]["data"]:
            raise DataFetchError(f"No historical data found for {symbol}")
        columns = data["dataset_data"]["column_names"]
        rows = data["dataset_data"]["data"]
        df = pl.DataFrame(rows, schema=columns).with_columns(
            pl.col("Date").str.to_datetime(format="%Y-%m-%d").alias("timestamp")
        )
        lf = df.drop("Date").lazy().sort("timestamp")
        if eager:
            return lf.collect()
        return lf

    async def get_realtime(
        self, symbol: str, *, interval: float = 300.0
    ) -> AsyncIterator[pl.DataFrame]:
        while True:
            try:
                dataset_code = symbol.replace("/", "_")
                url = f"{self.base_url}/{dataset_code}/data.json"
                params = {"api_key": self.api_key, "limit": 1}
                data = await self._request(url, params=params)
                if "dataset_data" not in data or not data["dataset_data"]["data"]:
                    raise DataFetchError(f"No real-time data found for {symbol}")
                columns = data["dataset_data"]["column_names"]
                row = data["dataset_data"]["data"][0]
                df = pl.DataFrame([row], schema=columns).with_columns(
                    pl.col("Date").str.to_datetime(format="%Y-%m-%d").alias("timestamp")
                )
                yield df.drop("Date")
            except Exception as e:
                raise DataFetchError(f"Failed to fetch real-time data for {symbol}: {e}")
            await asyncio.sleep(interval)

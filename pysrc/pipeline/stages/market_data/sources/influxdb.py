# py/pipeline/stages/market_data/sources/influxdb.py
import asyncio
from collections.abc import AsyncIterator

import polars as pl
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

from . import register_source


@register_source("influxdb")
class InfluxDBSource(DataSource):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = InfluxDBClient(
            url=config["url"], token=config["token"], org=config.get("org", "default_org")
        )
        self.query_api: QueryApi = self.client.query_api()
        self.bucket = config.get("bucket", "default_bucket")
        self.measurement = config.get("measurement", "market_data")

    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        try:
            flux_query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: {start}, stop: {end})
                |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
            '''
            pd_df = await asyncio.to_thread(self.query_api.query_data_frame, flux_query)
            if pd_df.empty:
                raise DataFetchError(f"No historical data found for {symbol}")
            df = (
                pl.from_pandas(pd_df, rechunk=False)
                .rename({"_time": "timestamp"})
                .with_columns(pl.col("timestamp").cast(pl.Datetime))
            )
            lf = df.lazy().sort("timestamp")
            if eager:
                return lf.collect()
            return lf
        except Exception as e:
            raise DataFetchError(f"Failed to fetch historical data for {symbol}: {e}")

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        while True:
            try:
                flux_query = f'''
                    from(bucket: "{self.bucket}")
                    |> range(start: -1m)
                    |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
                    |> filter(fn: (r) => r["symbol"] == "{symbol}")
                    |> last()
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
                '''
                pd_df = await asyncio.to_thread(self.query_api.query_data_frame, flux_query)
                if not pd_df.empty:
                    df = (
                        pl.from_pandas(pd_df, rechunk=False)
                        .rename({"_time": "timestamp"})
                        .with_columns(pl.col("timestamp").cast(pl.Datetime))
                    )
                    yield df
            except Exception as e:
                raise DataFetchError(f"Failed to fetch real-time data for {symbol}: {e}")
            await asyncio.sleep(interval)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

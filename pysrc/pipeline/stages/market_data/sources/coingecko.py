# py/pipeline/stages/market_data/sources/coingecko.py
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource, async_retry

from . import register_source


@register_source("coingecko")
class CoinGeckoSource(APIDataSource):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.vs_currency = config.get("vs_currency", "usd")
        self.coin_map = config.get(
            "coin_map", {"BTC": "bitcoin", "ETH": "ethereum", "USDT": "tether"}
        )
        self.rate_limit = config.get("rate_limit", 60.0 / 50)  # ~50 calls/min

    @async_retry(attempts=3, multiplier=1, min_delay=2, max_delay=60)
    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        if symbol not in self.coin_map:
            raise DataFetchError(f"Symbol {symbol} not supported")
        coin_id = self.coin_map[symbol]
        start_unix = int(datetime.fromisoformat(start).timestamp())
        end_unix = int(datetime.fromisoformat(end).timestamp())
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": self.vs_currency, "from": start_unix, "to": end_unix}
        data = await self._request(url, params=params)
        if "prices" not in data or not data["prices"]:
            raise DataFetchError(f"No historical data found for {symbol}")
        df_prices = (
            pl.DataFrame(data["prices"], schema=["timestamp_ms", "price"])
            .with_columns((pl.col("timestamp_ms") / 1000).cast(pl.Int64).alias("timestamp_unix"))
            .with_columns(pl.from_epoch("timestamp_unix").alias("timestamp"))
            .drop("timestamp_ms", "timestamp_unix")
        )
        df_volumes = (
            pl.DataFrame(data["total_volumes"], schema=["timestamp_ms", "volume"])
            .with_columns((pl.col("timestamp_ms") / 1000).cast(pl.Int64).alias("timestamp_unix"))
            .with_columns(pl.from_epoch("timestamp_unix").alias("timestamp"))
            .drop("timestamp_ms", "timestamp_unix")
        )
        df = df_prices.join(df_volumes, on="timestamp", how="inner")
        lf = df.lazy().sort("timestamp")
        if eager:
            return lf.collect()
        return lf

    async def get_realtime(
        self, symbol: str, *, interval: float = None
    ) -> AsyncIterator[pl.DataFrame]:
        interval = interval or self.rate_limit
        if symbol not in self.coin_map:
            raise DataFetchError(f"Symbol {symbol} not supported")
        coin_id = self.coin_map[symbol]
        url = f"{self.base_url}/simple/price"
        params = {"ids": coin_id, "vs_currencies": self.vs_currency}
        while True:
            try:
                data = await self._request(url, params=params)
                if coin_id not in data:
                    raise DataFetchError(f"No real-time data found for {symbol}")
                price = data[coin_id][self.vs_currency]
                yield pl.DataFrame({"timestamp": [datetime.now()], "price": [price]})
            except Exception as e:
                raise DataFetchError(f"Failed to fetch real-time data for {symbol}: {e}")
            await asyncio.sleep(interval)

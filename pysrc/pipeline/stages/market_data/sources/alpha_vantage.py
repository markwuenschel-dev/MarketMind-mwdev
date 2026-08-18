# py/pipeline/stages/market_data/sources/alpha_vantage.py
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource, async_retry

from . import register_source


@register_source("alpha_vantage")
class AlphaVantageSource(APIDataSource):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config["api_key"]
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = config.get("rate_limit", 60.0 / 5)  # e.g., 5 calls/min free tier

    @async_retry(attempts=3, multiplier=1, min_delay=2, max_delay=60)
    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",
        }
        data = await self._request(self.base_url, params=params)
        time_series = data.get("Time Series (Daily)")
        if not time_series:
            raise DataFetchError(f"No historical data found for {symbol}")
        records = [
            {
                "timestamp": k,
                "open": float(v["1. open"]),
                "high": float(v["2. high"]),
                "low": float(v["3. low"]),
                "close": float(v["4. close"]),
                "volume": int(v["6. volume"]),
                "adjusted_close": float(v["5. adjusted close"]),
                "dividend_amount": float(v["7. dividend amount"]),
                "split_coefficient": float(v["8. split coefficient"]),
            }
            for k, v in time_series.items()
        ]
        df = pl.DataFrame(records).with_columns(
            pl.col("timestamp").str.to_datetime(format="%Y-%m-%d")
        )
        df = df.filter(
            pl.col("timestamp").is_between(
                datetime.fromisoformat(start), datetime.fromisoformat(end)
            )
        )
        if df.is_empty():
            raise DataFetchError(f"No historical data found for {symbol}")
        lf = df.lazy().sort("timestamp")
        if eager:
            return lf.collect()
        return lf

    async def get_realtime(
        self, symbol: str, *, interval: float = None
    ) -> AsyncIterator[pl.DataFrame]:
        interval = interval or self.rate_limit
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key}
        while True:
            try:
                data = await self._request(self.base_url, params=params)
                quote = data.get("Global Quote")
                if not quote:
                    raise DataFetchError(f"No real-time data found for {symbol}")
                price = float(quote["05. price"])
                yield pl.DataFrame(
                    {
                        "timestamp": [datetime.now()],
                        "open": [price],
                        "high": [price],
                        "low": [price],
                        "close": [price],
                        "volume": [int(quote["06. volume"]) if "06. volume" in quote else 0],
                    }
                )
            except Exception as e:
                raise DataFetchError(f"Failed to fetch real-time data for {symbol}: {e}")
            await asyncio.sleep(interval)

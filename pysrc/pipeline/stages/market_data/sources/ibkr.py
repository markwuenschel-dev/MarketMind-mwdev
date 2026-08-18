# py/pipeline/stages/market_data/sources/ibkr.py
import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import polars as pl

try:
    from ib_insync import IB, BarDataList, Contract, RealTimeBar, Stock, util
except ImportError:
    # Dummy classes for smoke tests
    IB = BarDataList = Contract = RealTimeBar = Stock = util = None

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

from . import register_source


@register_source("ikbr")
class IBKRSource(DataSource):
    def __init__(self, config: dict):
        super().__init__(config)
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 7497)
        self.client_id = config.get("client_id", 1)
        self.default_bar_size = config.get("bar_size", "1 day")
        self.what_to_show = config.get("what_to_show", "TRADES")
        self.use_rth = config.get("use_rth", True)
        self.ib = None
        self.queue_maxsize = config.get("queue_maxsize", 100)

    async def __aenter__(self):
        util.patchAsyncio()
        self.ib = IB()
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ib:
            self.ib.disconnect()

    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        if not self.ib:
            raise RuntimeError("Use IBKRSource as async context manager")
        try:
            duration, bar_size, end_dt = self._to_ib_window(start, end)
            contract = Stock(symbol, "SMART", "USD")
            bars: BarDataList = await self.ib.reqHistoricalDataAsync(
                contract,
                endDatetime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=self.what_to_show,
                useRTH=self.use_rth,
                formatDate=1,
            )
            if not bars:
                raise DataFetchError(f"No historical data found for {symbol}")
            pd_df = util.df(bars)
            df = (
                pl.from_pandas(pd_df, rechunk=False)
                .rename({"date": "timestamp"})
                .with_columns(pl.col("timestamp").cast(pl.Datetime))
            )
            lf = (
                df.lazy()
                .filter(
                    pl.col("timestamp").is_between(
                        datetime.fromisoformat(start), datetime.fromisoformat(end)
                    )
                )
                .sort("timestamp")
            )
            if eager:
                return lf.collect()
            return lf
        except Exception as e:
            raise DataFetchError(f"Failed to fetch historical data for {symbol}: {e}")

    async def get_realtime(
        self, symbol: str, *, interval: float = 5.0
    ) -> AsyncIterator[pl.DataFrame]:
        if not self.ib:
            raise RuntimeError("Use IBKRSource as async context manager")
        contract: Contract = Stock(symbol, "SMART", "USD")
        q: asyncio.Queue[RealTimeBar] = asyncio.Queue(maxsize=self.queue_maxsize)

        def on_bar_update(bars: BarDataList, has_new_bar: bool):
            if has_new_bar:
                try:
                    asyncio.get_event_loop_policy().get_event_loop().call_soon_threadsafe(
                        q.put_nowait, bars[-1]
                    )
                except asyncio.queues.QueueFull:
                    pass  # Drop if full for backpressure

        bars = await self.ib.reqRealTimeBarsAsync(
            contract, barSize=5, whatToShow=self.what_to_show, useRTH=self.use_rth
        )
        bars.updateEvent += on_bar_update
        try:
            while True:
                rtbar: RealTimeBar = await q.get()
                yield self._rtbar_to_df(rtbar, symbol)
                await asyncio.sleep(interval)  # Additional back-pressure
        finally:
            self.ib.cancelRealTimeBars(bars)

    def _rtbar_to_df(self, rtbar: RealTimeBar, symbol: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": [rtbar.time],
                "open": [rtbar.open],
                "high": [rtbar.high],
                "low": [rtbar.low],
                "close": [rtbar.close],
                "volume": [rtbar.volume],
                "wap": [rtbar.average],
                "count": [rtbar.count],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime))

    def _to_ib_window(self, start: str, end: str):
        s_dt = datetime.fromisoformat(start).astimezone(UTC)
        e_dt = datetime.fromisoformat(end).astimezone(UTC)
        delta = e_dt - s_dt
        days = delta.days + (delta.seconds / 86400)
        if days > 365 * 2:
            return "2 Y", "1 day", e_dt
        elif days > 365:
            return "1 Y", "1 day", e_dt
        elif days > 30:
            return f"{int(days)} D", "1 hour", e_dt
        elif days > 1:
            return f"{int(days)} D", "5 mins", e_dt
        else:
            return "1 D", "5 secs", e_dt

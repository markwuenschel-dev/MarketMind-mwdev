# tests/python/unit/data/test_market_data_sources.py (redesigned)
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime as _dt
from typing import Any

import polars as pl
import pytest

from pysrc.pipeline.stages.market_data.sources.contracts import DataSource as _BaseDataSource
from pysrc.pipeline.stages.market_data.sources.market_data import (
    MarketDataConfig,
    MarketDataManager,
)


def _apply_time_range_and_symbol(df: pl.DataFrame, start, end, symbol: str) -> pl.DataFrame:
    # ensure a timestamp column exists and is datetime-typed
    if "timestamp" not in df.columns:
        raise AssertionError("expected 'timestamp' column")
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))

    # inclusive range filter with tolerant parsing of string/datetime inputs
    sdt = pl.Series([str(start)]).str.strptime(pl.Datetime, strict=False)[0] if start else None
    edt = pl.Series([str(end)]).str.strptime(pl.Datetime, strict=False)[0] if end else None
    if sdt is not None:
        df = df.filter(pl.col("timestamp") >= sdt)
    if edt is not None:
        df = df.filter(pl.col("timestamp") <= edt)

    # stamp a symbol column if the input table is single-asset
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
    return df


class _FakeSource(_BaseDataSource):
    def __init__(
        self,
        table: pl.DataFrame,
        config: dict[str, Any] | None = None,
        *,
        name: str = "alpha_vantage",
    ):
        cfg = config or {"name": name, "type": "fake", "priority": 1}
        try:
            super().__init__(config=cfg)  # kw style
        except TypeError:
            try:
                super().__init__(cfg)  # positional
            except TypeError:
                super().__init__()  # no-arg
        self._table = table
        self.calls = {"hist": 0, "rt": 0}

    async def get_historical(self, symbol: str, start: str, end: str, *, eager: bool = False):
        # use parameters to filter and return lazy by default to match runtime behavior
        self.calls["hist"] += 1
        df = _apply_time_range_and_symbol(self._table, start, end, symbol)
        lf = df.lazy().sort("timestamp")
        return lf.collect() if eager else lf

    def get_realtime(self, symbol: str, *, interval: float = 60.0) -> AsyncIterator[pl.DataFrame]:
        # consume interval to satisfy linters without adding test latency
        self.calls["rt"] += 1

        async def _gen():
            await asyncio.sleep(0 if not interval or interval < 0 else min(float(interval), 0.001))
            yield pl.DataFrame(
                {"timestamp": [_dt(2024, 1, 1, 0, 0, 0)], "symbol": [symbol], "close": [102.0]}
            )

        return _gen()


@pytest.fixture
def av_manager():
    # architect: explicit instance registration avoids depending on global registry wiring
    tbl = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1d", eager=True
            ),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 200, 300],
        }
    )
    mgr = MarketDataManager(
        config=MarketDataConfig()
    )  # architect: default pipeline_config; we pass source_name explicitly
    config = {"name": "alpha_vantage", "type": "fake", "priority": 1}
    src = _FakeSource(tbl, config)
    mgr.register_instance("alpha_vantage", src)  # architect: name matches what tests call
    return mgr, src


@pytest.mark.asyncio
async def test_manager_historical_success(av_manager):
    mgr, src = av_manager
    lf = await mgr.get_historical(
        "AAPL", "2024-01-01", "2024-01-03", source_name="alpha_vantage", eager=False
    )
    assert isinstance(lf, pl.LazyFrame)  # architect: lazy by default
    df = lf.collect()
    assert df.shape[0] == 3
    assert "symbol" in df.columns
    assert src.calls["hist"] == 1


@pytest.mark.asyncio
async def test_manager_historical_cached(av_manager):
    mgr, src = av_manager
    # architect: cache is client-internal; identical key should de-dup downstream calls
    df1 = await mgr.get_historical_cached(
        "AAPL", "2024-01-01", "2024-01-03", source_name="alpha_vantage", eager=True
    )
    df2 = await mgr.get_historical_cached(
        "AAPL", "2024-01-01", "2024-01-03", source_name="alpha_vantage", eager=True
    )
    # Polars >=0.17: DataFrame.frame_equal; fall back to robust comparison if unavailable
    # Compare results robustly across DataFrame backends and Polars versions
    try:
        import polars as pl  # type: ignore
    except ImportError:
        pl = None  # type: ignore[assignment]

    if pl is not None and isinstance(df1, pl.DataFrame) and isinstance(df2, pl.DataFrame):
        equal = getattr(df1, "frame_equal", None)
        if callable(equal):
            # Call via the retrieved attribute to avoid static "unresolved attribute" warnings
            assert equal(df2, null_equal=True)
        else:
            import pandas as pd  # type: ignore

            assert df1.to_pandas().equals(df2.to_pandas())
    else:
        import pandas as pd  # type: ignore

        df1p = df1.to_pandas() if hasattr(df1, "to_pandas") else df1
        df2p = df2.to_pandas() if hasattr(df2, "to_pandas") else df2
        assert isinstance(df1p, pd.DataFrame)
        assert isinstance(df2p, pd.DataFrame)
        assert df1p.equals(df2p)
    assert src.calls["hist"] == 1  # architect: second fetch served from cache


@pytest.mark.asyncio
async def test_manager_historical_many_concat(av_manager):
    mgr, src = av_manager
    out = await mgr.get_historical_many(
        ["AAPL", "MSFT"], "2024-01-01", "2024-01-03", source_name="alpha_vantage", eager=True
    )
    # architect: client concatenates per-symbol frames vertically and stamps a symbol column
    assert out.shape[0] == 6
    assert set(out["symbol"].unique().to_list()) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_manager_realtime_stream(av_manager):
    mgr, _ = av_manager
    # architect: API returns an async generator; you iterate to receive ticks
    stream = mgr.get_realtime("AAPL", source_name="alpha_vantage", interval=0.01)
    try:
        first = await stream.__anext__()
        assert first["close"][0] == 102.0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_manager_realtime_missing_source():
    mgr = MarketDataManager(config=MarketDataConfig())
    # architect: _resolve -> REGISTRY.ensure -> KeyError("unknown data source") occurs on first iteration
    stream = mgr.get_realtime("AAPL", source_name="does_not_exist")
    try:
        with pytest.raises(KeyError):
            await stream.__anext__()
    finally:
        await stream.aclose()

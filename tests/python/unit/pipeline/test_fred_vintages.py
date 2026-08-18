from __future__ import annotations

from contextlib import aclosing
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest

from pysrc.data.dataview import DataView
from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources import fred as fred_module
from pysrc.pipeline.stages.market_data.sources.fred import (
    FREDApproximationStub,
    FREDSource,
    FREDVintageSeam,
)

pytestmark = pytest.mark.net


class _FakeFredClient:
    def get_series(self, symbol: str, observation_start: str, observation_end: str) -> pd.Series:
        _ = (symbol, observation_start, observation_end)
        return pd.Series([3.1, 3.2], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))

    def get_series_latest_release(self, symbol: str) -> pd.Series:
        _ = symbol
        return pd.Series([3.14], index=pd.to_datetime(["2024-03-01"]))


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_output_has_valid_time() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    df = await source.get_historical("CPIAUCSL", "2024-01-01", "2024-02-28", eager=True)
    assert "valid_time" in df.columns


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_uses_default_retrieval_clock_when_none() -> None:
    """FREDSource uses datetime.now(utc) when retrieval_clock is not provided."""
    source = FREDSource({"api_key": "test"}, fred_client=_FakeFredClient())
    df = await source.get_historical("X", "2024-01-01", "2024-02-28", eager=True)
    assert "knowledge_time" in df.columns
    assert not df.is_empty()


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_output_has_knowledge_time() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    df = await source.get_historical("CPIAUCSL", "2024-01-01", "2024-02-28", eager=True)
    assert "knowledge_time" in df.columns


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_registers_into_dataview() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    df = await source.get_historical("CPIAUCSL", "2024-01-01", "2024-02-28", eager=True)
    dataview = DataView()
    pandas_df = df.with_columns(pl.col("knowledge_time").dt.date()).to_pandas()
    dataview.register_source(pandas_df)
    snapshot = dataview.as_of(
        symbols=["CPIAUCSL"],
        fields=["value"],
        knowledge_date=df["knowledge_time"][0].date(),
    )
    assert not snapshot.empty


@pytest.mark.determinism("d1")
def test_fred_seam_is_explicit() -> None:
    assert FREDApproximationStub.__name__ == "FREDApproximationStub"
    assert "Phase I approximation" in (FREDApproximationStub.__doc__ or "")
    assert "ALFRED" in (FREDApproximationStub.__doc__ or "")


@pytest.mark.determinism("d1")
def test_fred_seam_interface_replaceable() -> None:
    stub = FREDApproximationStub()
    assert isinstance(stub, FREDVintageSeam)


class _EmptyFredClient:
    def get_series(self, symbol: str, observation_start: str, observation_end: str) -> pd.Series:
        _ = (symbol, observation_start, observation_end)
        return pd.Series(dtype=float)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_empty_series_raises() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_EmptyFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    with pytest.raises(DataFetchError, match="No historical data found"):
        await source.get_historical("X", "2024-01-01", "2024-02-28", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_empty_symbol_raises() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    with pytest.raises(DataFetchError, match="non-empty symbol"):
        await source.get_historical("", "2024-01-01", "2024-02-28", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_start_after_end_raises() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    with pytest.raises(DataFetchError, match="start <= end"):
        await source.get_historical("X", "2024-02-01", "2024-01-15", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_returns_lazy_when_not_eager() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    result = await source.get_historical("CPIAUCSL", "2024-01-01", "2024-02-28", eager=False)
    assert isinstance(result, pl.LazyFrame)
    assert not result.collect().is_empty()


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_realtime_yields_frames() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    count = 0
    async with aclosing(source.get_realtime("CPIAUCSL", interval=0.01)) as stream:
        async for df in stream:
            assert isinstance(df, pl.DataFrame)
            assert "value" in df.columns
            count += 1
            if count >= 2:
                break


class _FailingFredClient:
    def get_series(self, symbol: str, observation_start: str, observation_end: str) -> pd.Series:
        raise ValueError("api error")

    def get_series_latest_release(self, symbol: str) -> pd.Series:
        raise AttributeError("no such method")


class _TypeErrorFredClient:
    def get_series(self, symbol: str, observation_start: str, observation_end: str) -> pd.Series:
        raise TypeError("invalid type")

    def get_series_latest_release(self, symbol: str) -> pd.Series:
        return pd.Series([1.0], index=pd.to_datetime(["2024-03-01"]))


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_get_historical_fetch_error_wrapped() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FailingFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    with pytest.raises(DataFetchError, match="Failed to fetch historical"):
        await source.get_historical("X", "2024-01-01", "2024-02-28", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_realtime_fetch_error_wrapped() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FailingFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    with pytest.raises(DataFetchError, match="Failed to fetch real-time"):
        async with aclosing(source.get_realtime("X", interval=0.01)) as stream:
            async for _ in stream:
                break


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_get_historical_type_error_wrapped() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_TypeErrorFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
    )
    with pytest.raises(DataFetchError, match="Failed to fetch historical"):
        await source.get_historical("X", "2024-01-01", "2024-02-28", eager=True)


class _BadSeamStub:
    def apply(self, frame: pl.DataFrame, *, retrieval_time: datetime) -> pl.DataFrame:
        return frame.select(["symbol", "date", "value"])


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fred_vintage_seam_missing_columns_raises() -> None:
    source = FREDSource(
        {"api_key": "test"},
        fred_client=_FakeFredClient(),
        retrieval_clock=lambda: datetime(2024, 3, 10, tzinfo=UTC),
        vintage_seam=_BadSeamStub(),
    )
    with pytest.raises(DataFetchError, match="missing required columns"):
        await source.get_historical("X", "2024-01-01", "2024-02-28", eager=True)


@pytest.mark.determinism("d1")
def test_fred_init_raises_when_fredapi_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """FREDSource.__init__ raises ImportError when fredapi is not installed and no fred_client."""
    monkeypatch.setattr(fred_module, "Fred", None)
    with pytest.raises(ImportError, match="fredapi is required unless a fred_client is provided"):
        FREDSource({"api_key": "x"})


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_fred_init_uses_fred_from_config_when_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FREDSource.__init__ uses Fred(api_key=config['api_key']) when fred_client not passed (lines 60-61)."""
    if fred_module.Fred is None:
        pytest.skip("fredapi not installed")
    fake = _FakeFredClient()
    mock_fred_class = MagicMock(return_value=fake)
    monkeypatch.setattr(fred_module, "Fred", mock_fred_class)
    source = FREDSource({"api_key": "test_key"})
    assert source.fred is fake
    mock_fred_class.assert_called_once_with(api_key="test_key")
    df = await source.get_historical("X", "2024-01-01", "2024-02-28", eager=True)
    assert not df.is_empty()

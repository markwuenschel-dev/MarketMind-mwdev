from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest

from pysrc.core.errors import DataFetchError
from pysrc.data.dataview import DataView
from pysrc.pipeline.stages.market_data.sources.yahoo_fetcher import YahooFinanceSource

pytestmark = pytest.mark.net


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1000000, 1100000],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_emits_required_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(source, "_download_history", lambda _symbol, _start, _end: _history_frame())
    df = await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)
    assert {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "valid_time",
        "knowledge_time",
    } <= set(df.columns)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_valid_time_is_date(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(source, "_download_history", lambda _symbol, _start, _end: _history_frame())
    df = await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)
    assert df["valid_time"].to_list() == df["date"].to_list()


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_knowledge_time_gte_valid_time(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(source, "_download_history", lambda _symbol, _start, _end: _history_frame())
    df = await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)
    assert all(
        knowledge.date() >= valid_time
        for knowledge, valid_time in zip(
            df["knowledge_time"].to_list(), df["valid_time"].to_list(), strict=False
        )
    )


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_registers_into_dataview(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(source, "_download_history", lambda _symbol, _start, _end: _history_frame())
    df = await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)
    dataview = DataView()
    pandas_df = df.with_columns(pl.col("knowledge_time").dt.date()).to_pandas()
    dataview.register_source(pandas_df)
    snapshot = dataview.as_of(
        symbols=["SPY"],
        fields=["close"],
        knowledge_date=df["knowledge_time"][0].date(),
    )
    assert not snapshot.empty


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_empty_symbol_list_raises() -> None:
    source = YahooFinanceSource()
    with pytest.raises(DataFetchError):
        await source.get_historical("", "2024-01-01", "2024-01-31", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_invalid_date_range_raises() -> None:
    source = YahooFinanceSource()
    with pytest.raises(DataFetchError):
        await source.get_historical("SPY", "2024-02-01", "2024-01-31", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_empty_history_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(source, "_download_history", lambda _s, _start, _end: pd.DataFrame())
    with pytest.raises(DataFetchError, match="No historical data found"):
        await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_missing_date_column_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    raw = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1e6]},
        index=pd.Index([0], name="x"),
    )
    monkeypatch.setattr(source, "_download_history", lambda _s, _start, _end: raw)
    with pytest.raises(DataFetchError, match="missing a date column"):
        await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_returns_lazy_when_not_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(source, "_download_history", lambda _symbol, _start, _end: _history_frame())
    result = await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=False)
    assert isinstance(result, pl.LazyFrame)
    assert not result.collect().is_empty()


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_realtime_raises_not_implemented() -> None:
    source = YahooFinanceSource()
    with pytest.raises(NotImplementedError, match="does not implement real-time"):
        await source.get_realtime("SPY", interval=60.0)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_missing_required_columns_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    raw = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    monkeypatch.setattr(source, "_download_history", lambda _s, _start, _end: raw)
    with pytest.raises(DataFetchError, match="missing columns"):
        await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_knowledge_time_before_valid_time_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1e6],
        },
        index=pd.to_datetime(["2024-01-15"]),
    )
    monkeypatch.setattr(source, "_download_history", lambda _s, _start, _end: raw)
    with pytest.raises(DataFetchError, match="knowledge_time < valid_time"):
        await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_fetcher_download_history_via_yfinance_ticker() -> None:
    """Cover _download_history path (yfinance.Ticker and .history) when yfinance is available."""
    try:
        import yfinance  # noqa: F401
    except ImportError:
        pytest.skip("yfinance not installed")
    source = YahooFinanceSource(fetch_clock=lambda: datetime(2024, 1, 10, tzinfo=UTC))
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = _history_frame()
        df = await source.get_historical("SPY", "2024-01-01", "2024-01-31", eager=True)
    assert not df.is_empty()
    assert "symbol" in df.columns

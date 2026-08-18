"""Tests for market_data.sources.runtime (APIDataSource)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource

pytestmark = [pytest.mark.determinism("d1"), pytest.mark.net]


class _ConcreteAPIDataSource(APIDataSource):
    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        raise NotImplementedError

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        raise NotImplementedError
        yield  # type: ignore[misc]


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_api_data_source_request_returns_json() -> None:
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={"key": "value"})
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.closed = False
    mock_session.close = AsyncMock(return_value=None)
    with patch(
        "pysrc.pipeline.stages.market_data.sources.runtime.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        src = _ConcreteAPIDataSource({})
    try:
        data = await src._request("https://api.example.com/data", timeout=1)
        assert data == {"key": "value"}
    finally:
        await src.close()


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_api_data_source_context_manager_closes_session() -> None:
    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock(return_value=None)
    with patch(
        "pysrc.pipeline.stages.market_data.sources.runtime.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        src = _ConcreteAPIDataSource({})
    async with src:
        assert not src.session.closed
    assert mock_session.close.await_count >= 1


@pytest.mark.asyncio
@pytest.mark.determinism("d1")
async def test_api_data_source_close_idempotent() -> None:
    class SessionState:
        closed: bool = False

    state = SessionState()
    mock_session = MagicMock()
    mock_session.closed = False

    async def close_session() -> None:
        state.closed = True
        mock_session.closed = True

    mock_session.close = AsyncMock(side_effect=close_session)
    with patch(
        "pysrc.pipeline.stages.market_data.sources.runtime.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        src = _ConcreteAPIDataSource({})
    await src.close()
    assert state.closed
    await src.close()
    assert mock_session.close.await_count == 1

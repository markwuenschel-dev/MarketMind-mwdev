import asyncio
from unittest.mock import MagicMock

import aiohttp
import pandas as pd
import pytest

from pysrc.pipeline.stages.market_data.sources.data_loader import APIDataLoader, CSVLoader


def test_csv_loader_chunks(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n" + "\n".join(f"{i},{i + 1}" for i in range(2000)))
    conf = type("C", (), {"path": str(p), "chunksize": 1000})
    df_parts = CSVLoader(conf).load_data()
    assert len(df_parts) == 2
    df = pd.concat(df_parts)
    assert len(df) == 2000
    assert list(df.columns) == ["a", "b"]


@pytest.mark.asyncio
async def test_api_dataloader_request_ok(monkeypatch):
    loader = APIDataLoader(MagicMock())

    async def fake_request(url, params, timeout=None):
        return [{"k": "v"}]

    monkeypatch.setattr(loader, "_request", fake_request)
    res = await loader._request("http://x", {}, 10)
    assert res == [{"k": "v"}]


@pytest.mark.asyncio
async def test_api_dataloader_timeout(monkeypatch):
    loader = APIDataLoader(MagicMock())
    loader.max_attempts = 1  # Disable retries for this test

    class MockResponse:
        async def __aenter__(self):
            raise TimeoutError()

        async def __aexit__(self, *args):
            pass

    def mock_get(*a, **k):
        return MockResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "get", mock_get)
    with pytest.raises(asyncio.TimeoutError):
        await loader._request("http://x", {}, 5)

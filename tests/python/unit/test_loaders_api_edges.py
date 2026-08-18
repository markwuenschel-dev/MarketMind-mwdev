# tests/python/unit/test_loaders_api_edges.py
import asyncio
from unittest.mock import MagicMock

import pytest

from pysrc.core.errors import DataFetchError
from pysrc.pipeline.stages.market_data.sources.data_loader import (
    APIDataLoader,
    InfluxDBLoader,
    build_loader,
)
from tests.python.infra.aio import AioHTTPMock


@pytest.mark.asyncio
async def test_api_non_200_raises(monkeypatch):
    http = AioHTTPMock()
    http.register_text("GET", "https://x/api", "boom", status=500)
    http.patch(monkeypatch, target="aiohttp.ClientSession")
    loader = APIDataLoader(MagicMock())
    with pytest.raises(DataFetchError):
        await loader._request("https://x/api", {}, timeout=5)


@pytest.mark.asyncio
async def test_api_retry_backoff(monkeypatch):
    # Make .get raise Timeout twice, then succeed
    calls = {"n": 0}

    def _get(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            # Create an async context manager that raises TimeoutError on entry
            class FailingCtx:
                async def __aenter__(self):
                    raise TimeoutError()

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return False

            return FailingCtx()

        class MockResponse:
            def __init__(self):
                self.status = 200

            async def json(self):
                return {"ok": True}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return MockResponse()

    # Patch ClientSession.get at the import site used by the canonical loader.
    import pysrc.pipeline.stages.market_data.sources.data_loader as dl

    monkeypatch.setattr(dl.aiohttp.ClientSession, "get", _get, raising=True)

    sleeper = []

    async def fake_sleep(t):
        sleeper.append(t)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep, raising=False)

    loader = APIDataLoader(MagicMock(max_attempts=3, retry_strategy="exponential"))
    out = await loader._request("https://any", {}, timeout=1)
    assert out == {"ok": True}
    assert sleeper
    assert sorted(set(sleeper)) == [1]


def test_build_loader_maps_known_types():
    # Adjust keys/classes to whatever your build_loader supports today
    cases = {
        "influxdb": InfluxDBLoader,
        # "twitter": TwitterLoader,
        # "fred": FREDLoader,
        # ...
    }
    for key, cls in cases.items():
        ins = build_loader(key)
        assert isinstance(ins, cls)


def test_influxdb_query_invocation(monkeypatch):
    fake_client = MagicMock()
    fake_client.query_api.return_value.query.return_value = []  # empty result ok
    from pysrc.pipeline.stages.market_data.sources.data_loader import InfluxDBLoader

    loader = InfluxDBLoader(
        MagicMock(
            url="http://localhost",
            token="x",
            org="o",
            bucket="b",
            query='from(bucket: "b") |> range(start: -1h)',
        )
    )
    monkeypatch.setattr(loader, "_make_client", lambda: fake_client)
    _ = loader.load_data()
    fake_client.query_api.return_value.query.assert_called()

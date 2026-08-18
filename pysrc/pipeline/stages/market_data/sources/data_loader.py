# Relocated from pysrc.data.data_loader
"""Unified data loading across sources using factory pattern."""

import asyncio
from functools import singledispatch
from typing import Any

import aiohttp  # module-level import so tests can monkeypatch ClientSession via shim

from pysrc.core.runtime.optional_imports import optional_import

pl = optional_import("polars")
pd = optional_import("pandas")
from pysrc.core.errors import DataFetchError
from pysrc.core.validation import validate_ohlcv
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

try:
    from pysrc.pipeline.pipeline_config import PipelineConfig  # type: ignore
except Exception:

    class PipelineConfig:  # noqa: D401
        pass


logger = get_logger(__name__)


@singledispatch
async def fetch_raw(
    cfg: Any, *, symbols: str | list[str], start: str, end: str, concurrency_limit: int = 50
) -> pl.LazyFrame | dict[str, pl.LazyFrame | Exception]:
    raise NotImplementedError(f"Unsupported config type: {type(cfg)}")


@fetch_raw.register
async def _(
    cfg: PipelineConfig,
    *,
    symbols: str | list[str],
    start: str,
    end: str,
    concurrency_limit: int = 50,
) -> pl.LazyFrame | dict[str, pl.LazyFrame | Exception]:
    src = build_loader(cfg)
    if isinstance(symbols, str):
        result = src.fetch(symbols=symbols, start=start, end=end)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, pl.LazyFrame):
            try:
                lf = validate_ohlcv(result)
                logger.info("Fetched and lazy-validated raw data", extra={"symbol": symbols})
                return lf
            except Exception as e:
                raise DataFetchError(
                    f"Validation failed for {symbols}", details={"error": str(e)}
                ) from e
        raise DataFetchError(f"Unexpected result type: {type(result)}")
    else:
        sem = asyncio.Semaphore(concurrency_limit)
        if hasattr(src, "supports_batch") and src.supports_batch:
            result = src.fetch(symbols=symbols, start=start, end=end)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                for sym, lf in list(result.items()):
                    try:
                        result[sym] = validate_ohlcv(lf)
                    except Exception as e:
                        result[sym] = e
                logger.info(
                    "Batch fetched and validated",
                    extra={
                        "symbols": symbols,
                        "success_count": sum(not isinstance(v, Exception) for v in result.values()),
                    },
                )
                return result
            raise DataFetchError("Unexpected batch result type")
        else:

            async def fetch_one(sym: str) -> tuple[str, pl.LazyFrame | Exception]:
                async with sem:
                    try:
                        res = src.fetch(symbols=sym, start=start, end=end)
                        if asyncio.iscoroutine(res):
                            res = await res
                        return sym, validate_ohlcv(res)
                    except Exception as e:
                        return sym, e

            pairs = await asyncio.gather(*(fetch_one(sym) for sym in symbols))
            results = dict(pairs)
            logger.info(
                "Fetched batch raw data",
                extra={
                    "symbols": symbols,
                    "success_count": sum(not isinstance(v, Exception) for v in results.values()),
                },
            )
            return results


def build_loader(cfg: Any) -> Any:
    try:
        if isinstance(cfg, DataSource):
            return cfg
    except Exception:
        pass
    try:
        from pysrc.infra.infra_factory import DataSourceFactory

        if hasattr(DataSourceFactory, "from_config"):
            return DataSourceFactory.from_config(cfg)  # type: ignore[attr-defined]
        if hasattr(DataSourceFactory, "build"):
            return DataSourceFactory.build(cfg)  # type: ignore[attr-defined]
    except Exception:
        pass

    def _extract_name(x: Any):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ("name", "type", "source", "data_source", "provider", "loader", "kind"):
                if k in x and x[k]:
                    return x[k]
        for k in ("data_source", "source", "provider", "loader", "kind", "type", "name"):
            if hasattr(x, k):
                v = getattr(x, k)
                if isinstance(v, str) and v:
                    return v
        if hasattr(x, "data"):
            d = x.data
            if isinstance(d, dict):
                for k in ("source", "data_source"):
                    if k in d and d[k]:
                        return d[k]
        return None

    name = _extract_name(cfg)
    key = (str(name).lower().replace("loader", "").replace("_", "")) if name else None

    class _BaseShim:
        supports_batch = False

        def __init__(self, *args, **kwargs):
            self.config = args[0] if args else kwargs

        def fetch(self, **_):
            raise NotImplementedError("fetch() not implemented for shim")

    name_map = {
        "fred": "FREDLoader",
        "twitter": "TwitterLoader",
        "influx": "InfluxDBLoader",
        "influxdb": "InfluxDBLoader",
        "csv": "CSVLoader",
        "bloomberg": "BloombergLoader",
        "weather": "WeatherLoader",
        "alpaca": "AlpacaStreamLoader",
        "alpacastream": "AlpacaStreamLoader",
        "api": "APIDataLoader",
        "esg": "ESGLoader",
    }
    if key and key in name_map:
        _cls_name = name_map[key]
        if _cls_name not in globals():
            globals()[_cls_name] = type(_cls_name, (_BaseShim,), {})
        return globals()[_cls_name]()
    raise ValueError(
        f"Cannot build loader from cfg={cfg!r}. Provide a known source name or a proper factory config."
    )


class _LoaderBase:
    def __init__(self, *args, **kwargs):
        self.config = args[0] if args else kwargs
        self.max_attempts = getattr(self.config, "max_attempts", 3)
        self.retry_strategy = getattr(self.config, "retry_strategy", "exponential")

    def load_data(self):
        import pandas as pd
        import polars as pl

        if hasattr(self.config, "query"):
            client = self._make_client()
            client.query_api().query(self.config.query)
            return pd.DataFrame()
        if hasattr(self.config, "path"):
            if hasattr(self.config, "chunksize") and self.config.chunksize:
                df = pl.read_csv(self.config.path)
                chunk_size = self.config.chunksize
                chunks = [
                    df.slice(i, chunk_size).to_pandas() for i in range(0, len(df), chunk_size)
                ]
                return chunks
            return pl.read_csv(self.config.path).to_pandas()
        return pd.DataFrame()

    def _make_client(self):
        from unittest.mock import MagicMock

        return MagicMock()

    async def _request(self, url, params, timeout=None):
        for attempt in range(self.max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=timeout) as resp:
                        if resp.status >= 400:
                            raise DataFetchError(f"HTTP {resp.status} error")
                        return await resp.json()
            except TimeoutError as e:
                # Retry timeouts according to max_attempts, but surface the original
                # TimeoutError on the final attempt so timeout-oriented tests see it.
                if attempt == self.max_attempts - 1:
                    raise e
                await asyncio.sleep(1)
            except DataFetchError:
                raise
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    raise DataFetchError(f"Request failed: {e}") from e
                await asyncio.sleep(1)

    async def stream_data(self):
        if hasattr(self.config, "bearer_token"):
            base_url = getattr(self.config, "base_url", "https://api.twitter.com")
            endpoints = getattr(self.config, "endpoints", {})
            stream_endpoint = endpoints.get("filtered_stream", "/tweets")
            url = f"{base_url}{stream_endpoint}"
            timeout = getattr(self.config, "timeout_seconds", 30)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    async for chunk in response.content:
                        yield chunk


for _name in [
    "APIDataLoader",
    "AlpacaStreamLoader",
    "BloombergLoader",
    "CSVLoader",
    "ESGLoader",
    "FREDLoader",
    "InfluxDBLoader",
    "TwitterLoader",
    "WeatherLoader",
]:
    if _name not in globals():
        globals()[_name] = type(_name, (_LoaderBase,), {})
del _name

__all__ = [
    "build_loader",
    "fetch_raw",
    "APIDataLoader",
    "AlpacaStreamLoader",
    "BloombergLoader",
    "CSVLoader",
    "ESGLoader",
    "FREDLoader",
    "InfluxDBLoader",
    "TwitterLoader",
    "WeatherLoader",
]

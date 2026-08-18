# Relocated from pysrc.data.market_data
import asyncio
import inspect
import time
from collections import OrderedDict
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Hashable,
    Iterable,
    Mapping,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    Any,
    TypeVar,
    cast,
    overload,
)

import polars as pl

from pysrc.core.errors import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

T = TypeVar("T")


class AsyncLRU:
    def __init__(self, maxsize: int = 512, ttl: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._inflight: dict[Hashable, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) > self.ttl

    def _evict(self) -> None:
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    async def get_or_set(self, key: Hashable, coro_factory: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            hit = self._cache.get(key)
            if hit and not self._expired(hit[0]):
                self._cache.move_to_end(key)
                return hit[1]
            fut = self._inflight.get(key)
            if fut is None:
                fut = asyncio.get_event_loop().create_future()
                self._inflight[key] = fut
                do_compute = True
            else:
                do_compute = False
        if not do_compute:
            return await fut
        try:
            val = await coro_factory()
            async with self._lock:
                self._cache[key] = (time.time(), val)
                self._evict()
                if not fut.done():
                    fut.set_result(val)
                if key in self._inflight:
                    del self._inflight[key]
            return val
        except Exception as e:
            async with self._lock:
                if not fut.done():
                    fut.set_exception(e)
                if key in self._inflight:
                    del self._inflight[key]
            raise


async def _retry[T](
    coro_factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.25,
    transient: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except transient as e:
            last = e
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2**i))
    assert last is not None
    raise last


@overload
async def _to_async_iter(streamish: AsyncIterable[pl.DataFrame]) -> AsyncIterator[pl.DataFrame]: ...
@overload
async def _to_async_iter(
    streamish: Awaitable[AsyncIterable[pl.DataFrame]],
) -> AsyncIterator[pl.DataFrame]: ...
async def _to_async_iter(streamish: Any) -> AsyncIterator[pl.DataFrame]:
    if inspect.isawaitable(streamish):
        streamish = await streamish
    if not hasattr(streamish, "__aiter__"):
        raise TypeError("expected AsyncIterable for realtime stream")
    return cast(AsyncIterator[pl.DataFrame], streamish.__aiter__())


@asynccontextmanager
async def _maybe_enter(src: DataSource):
    aenter = getattr(src, "__aenter__", None)
    aexit = getattr(src, "__aexit__", None)
    if callable(aenter) and callable(aexit):
        async with src:
            yield src
    else:
        yield src


class SourceRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[DataSource]] = {}
        self._instances: dict[str, DataSource] = {}

    def register(self, name: str, cls: type[DataSource]) -> None:
        self._classes[name.lower()] = cls

    def names(self) -> list[str]:
        return sorted(set(list(self._classes.keys()) + list(self._instances.keys())))

    def get(self, name: str) -> DataSource | None:
        return self._instances.get(name.lower())

    def create(self, name: str, cfg: Mapping[str, Any]) -> DataSource:
        key = name.lower()
        cls = self._classes.get(key)
        if cls is None:
            raise KeyError(f"unknown data source: {name}")
        inst = cls(dict(cfg))
        self._instances[key] = inst
        return inst

    def ensure(self, name: str, cfg: Mapping[str, Any]) -> DataSource:
        inst = self.get(name)
        return inst if inst else self.create(name, cfg)


REGISTRY = SourceRegistry()


def register_source(name: str) -> Callable[[type[DataSource]], type[DataSource]]:
    def _inner(cls: type[DataSource]) -> type[DataSource]:
        REGISTRY.register(name, cls)
        return cls

    return _inner


@register_source("composite")
class CompositeSource(DataSource):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__({})
        self._names: list[str] = list(cfg.get("sources", []))
        self._cfg_overrides: Mapping[str, Any] = cfg.get("overrides", {})
        if not self._names:
            raise DataFetchError("CompositeSource requires non-empty 'sources' list")

    def _resolve_all(self) -> list[DataSource]:
        return [REGISTRY.ensure(n, self._cfg_overrides.get(n, {})) for n in self._names]

    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame:
        async def one(src: DataSource) -> pl.LazyFrame | None:
            try:
                async with _maybe_enter(src):
                    lf = await src.get_historical(symbol, start, end, eager=False)
                    return lf if isinstance(lf, pl.LazyFrame) else lf.lazy()
            except Exception:
                return None

        lfs = [
            p for p in await asyncio.gather(*[one(s) for s in self._resolve_all()]) if p is not None
        ]
        if not lfs:
            raise DataFetchError(f"CompositeSource: all subsources failed for {symbol}")
        cols = set(lfs[0].schema.keys())
        for lf in lfs[1:]:
            cols &= set(lf.schema.keys())
        if not cols:
            raise DataFetchError("CompositeSource: no common columns across subsources")
        out = pl.concat([lf.select(sorted(cols)) for lf in lfs], how="vertical_relaxed")
        return out.collect() if eager else out

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        sources = self._resolve_all()
        streams: list[AsyncIterator[pl.DataFrame]] = []
        for s in sources:
            async with _maybe_enter(s):
                streamish = s.get_realtime(symbol, interval=interval)
                streams.append(await _to_async_iter(streamish))
        q: asyncio.Queue[pl.DataFrame | None] = asyncio.Queue()

        async def pump(gen: AsyncIterator[pl.DataFrame]) -> None:
            try:
                async for df in gen:
                    await q.put(df)
            finally:
                await q.put(None)

        tasks = [asyncio.create_task(pump(g)) for g in streams]
        finished = 0
        try:
            while finished < len(tasks):
                item = await q.get()
                if item is None:
                    finished += 1
                else:
                    yield item
        finally:
            for t in tasks:
                t.cancel()


@register_source("failover")
class FailoverSource(DataSource):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__({})
        self._order: list[str] = list(cfg.get("order", []))
        if not self._order:
            raise DataFetchError("FailoverSource requires non-empty 'order' list")

    async def get_historical(
        self, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame:
        last: BaseException | None = None
        for name in self._order:
            try:
                src = REGISTRY.ensure(name, {})
                async with _maybe_enter(src):
                    lf = await src.get_historical(symbol, start, end, eager=False)
                lf = lf if isinstance(lf, pl.LazyFrame) else lf.lazy()
                return lf.collect() if eager else lf
            except Exception as e:
                last = e
                continue
        raise DataFetchError(f"FailoverSource: all sources failed for {symbol}") from last

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        for name in self._order:
            try:
                src = REGISTRY.ensure(name, {})
                async with _maybe_enter(src):
                    streamish = src.get_realtime(symbol, interval=interval)
                    stream = await _to_async_iter(streamish)
                    async for df in stream:
                        yield df
                return
            except Exception:
                continue


@dataclass
class MarketDataConfig:
    default_source: str = "file"
    cache_maxsize: int = 512
    cache_ttl: float = 300.0
    retry_attempts: int = 3
    retry_base_delay: float = 0.25
    fanout_concurrency: int = 8
    source_configs: Mapping[str, Mapping[str, Any]] | None = None


class MarketDataManager:
    def __init__(self, *, config: MarketDataConfig | None = None) -> None:
        if isinstance(config, dict):
            cfg = dict(config)
            # Compatibility: run_cfg often has "sources" list; map to source_configs by name
            if "sources" in cfg and "source_configs" not in cfg:
                sources = cfg.get("sources") or []
                cfg["source_configs"] = {
                    s.get("name_for_registry") or s.get("type") or str(i): s
                    for i, s in enumerate(sources)
                    if isinstance(s, dict)
                }
            self.cfg = MarketDataConfig(
                default_source=cfg.get("default_source", "file"),
                cache_maxsize=cfg.get("cache_maxsize", 512),
                cache_ttl=cfg.get("cache_ttl", 300.0),
                retry_attempts=cfg.get("retry_attempts", 3),
                retry_base_delay=cfg.get("retry_base_delay", 0.25),
                fanout_concurrency=cfg.get("fanout_concurrency", 8),
                source_configs=cfg.get("source_configs"),
            )
        else:
            self.cfg = config or MarketDataConfig()
        self._cache = AsyncLRU(
            getattr(self.cfg, "cache_maxsize", 512),
            getattr(self.cfg, "cache_ttl", 300.0),
        )
        self._instances: dict[str, DataSource] = {}
        self._src_cfg: Mapping[str, Mapping[str, Any]] = (
            getattr(self.cfg, "source_configs", None) or {}
        )

    def register_instance(self, name: str, src: DataSource) -> None:
        self._instances[name.lower()] = src

    def _resolve(self, name: str | None) -> DataSource:
        key = (name or self.cfg.default_source).lower()
        inst = self._instances.get(key) or REGISTRY.get(key)
        if inst:
            self._instances[key] = inst
            return inst
        cfg = self._src_cfg.get(key, {})
        inst = REGISTRY.ensure(key, cfg or {})
        self._instances[key] = inst
        return inst

    async def _fetch_hist(
        self, src: DataSource, symbol: str, start: str, end: str, *, eager: bool = False
    ) -> pl.LazyFrame:
        async def _call() -> pl.LazyFrame:
            async with _maybe_enter(src):
                lf = await src.get_historical(symbol, start, end, eager=False)
            lf = lf if isinstance(lf, pl.LazyFrame) else lf.lazy()
            return lf.collect() if eager else lf

        return await _retry(
            _call, attempts=self.cfg.retry_attempts, base_delay=self.cfg.retry_base_delay
        )

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        source_name: str | None = None,
        eager: bool = False,
    ) -> pl.LazyFrame:
        src = self._resolve(source_name)
        return await self._fetch_hist(src, symbol, start, end, eager=eager)

    async def get_historical_cached(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        source_name: str | None = None,
        eager: bool = False,
    ) -> pl.LazyFrame:
        key = ("hist", source_name or self.cfg.default_source, symbol, start, end, eager)
        return await self._cache.get_or_set(
            key,
            lambda: self.get_historical(symbol, start, end, source_name=source_name, eager=eager),
        )

    async def get_historical_many(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        *,
        source_name: str | None = None,
        eager: bool = False,
        use_cache: bool = True,
    ) -> pl.LazyFrame:
        sem = asyncio.Semaphore(self.cfg.fanout_concurrency)

        async def one(sym: str) -> pl.LazyFrame:
            async with sem:
                lf = await (
                    self.get_historical_cached(
                        sym, start, end, source_name=source_name, eager=eager
                    )
                    if use_cache
                    else self.get_historical(sym, start, end, source_name=source_name, eager=eager)
                )
                return lf.with_columns(pl.lit(sym).alias("symbol"))

        lfs = await asyncio.gather(*[one(s) for s in symbols])
        return pl.concat(lfs, how="vertical_relaxed")

    async def get_historical_auto(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        candidates: list[str] | None = None,
        eager: bool = False,
    ) -> pl.LazyFrame:
        order = candidates or [self.cfg.default_source] + [
            n for n in REGISTRY.names() if n != self.cfg.default_source
        ]
        last: BaseException | None = None
        for name in order:
            try:
                return await self.get_historical(symbol, start, end, source_name=name, eager=eager)
            except Exception as e:
                last = e
                continue
        raise DataFetchError(f"all sources failed for {symbol}: {order}") from last

    async def get_realtime(
        self, symbol: str, *, source_name: str | None = None, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        src = self._resolve(source_name)
        async with _maybe_enter(src):
            streamish: Any = src.get_realtime(symbol, interval=interval)
            stream = await _to_async_iter(streamish)
            async for chunk in stream:
                yield chunk

    def close(self) -> None:
        for s in list(self._instances.values()):
            try:
                close = getattr(s, "close", None)
                if inspect.iscoroutinefunction(close):
                    asyncio.create_task(close())
                elif callable(close):
                    close()
            except Exception:
                pass


def build_client_from_config(cfg: Mapping[str, Any]) -> MarketDataManager:
    mcfg = MarketDataConfig(
        default_source=cfg.get("default_source", "file"),
        cache_maxsize=cfg.get("cache_maxsize", 512),
        cache_ttl=cfg.get("cache_ttl", 300.0),
        retry_attempts=cfg.get("retry_attempts", 3),
        retry_base_delay=cfg.get("retry_base_delay", 0.25),
        fanout_concurrency=cfg.get("fanout_concurrency", 8),
        source_configs=cfg.get("source_configs", {}),
    )
    return MarketDataManager(config=mcfg)

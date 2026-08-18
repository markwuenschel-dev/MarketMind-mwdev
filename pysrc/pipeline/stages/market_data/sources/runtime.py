"""Runtime helpers for network-backed market-data sources."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar, cast

import aiohttp

from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

T = TypeVar("T")


def async_retry(
    *,
    attempts: int,
    multiplier: float,
    min_delay: float,
    max_delay: float,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorate(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapped(*args: Any, **kwargs: Any) -> T:
            delay = min_delay
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == attempts:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(max_delay, max(min_delay, delay * multiplier * 2))
            raise RuntimeError("unreachable retry state")

        return cast(Callable[..., Awaitable[T]], wrapped)

    return decorate


class APIDataSource(DataSource):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.session = aiohttp.ClientSession()

    @async_retry(attempts=3, multiplier=1, min_delay=2, max_delay=60)
    async def _request(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 10,
    ) -> dict[str, Any]:
        async with self.session.get(
            url, params=params, headers=headers, timeout=timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        _ = (exc_type, exc_val, exc_tb)
        await self.close()

    async def close(self) -> None:
        if not self.session.closed:
            await self.session.close()

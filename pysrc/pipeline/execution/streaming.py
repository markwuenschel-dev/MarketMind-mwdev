# py/pipeline/execution/streaming.py
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pysrc.core.runtime.optional_imports import optional_import

pl = optional_import("polars")
pd = optional_import("pandas")


class _Sentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<_SENTINEL>"


# A single, process-unique sentinel instance
_SENTINEL: Final = _Sentinel()

# Make sure all required classes are exported
__all__ = ["_SENTINEL", "StreamingPipeline"]


def _to_polars_df(item: Any) -> pl.DataFrame:
    if pl is None:
        raise RuntimeError("polars is not available")
    if isinstance(item, pl.DataFrame):
        return item
    if pd is not None and isinstance(item, pd.DataFrame):
        return pl.from_pandas(item)
    if isinstance(item, dict):
        # Ensure consistent integer types by converting large integers to Int64
        normalized_dict = {}
        for k, v in item.items():
            if isinstance(v, int):
                # Keep integers within Int64 range
                if v > 2**63 - 1:
                    v = 2**63 - 1
                elif v < -(2**63):
                    v = -(2**63)
            normalized_dict[k] = v
        return pl.DataFrame([normalized_dict])
    # Fallback: try constructing from sequence of dicts
    if isinstance(item, Iterable):
        try:
            # Normalize integers in sequences as well
            normalized_items = []
            for item_elem in list(item):
                if isinstance(item_elem, dict):
                    normalized_dict = {}
                    for k, v in item_elem.items():
                        if isinstance(v, int):
                            if v > 2**63 - 1:
                                v = 2**63 - 1
                            elif v < -(2**63):
                                v = -(2**63)
                        normalized_dict[k] = v
                    normalized_items.append(normalized_dict)
                else:
                    normalized_items.append(item_elem)
            return pl.DataFrame(normalized_items)  # may raise, that's ok
        except Exception:
            pass
    raise TypeError(f"Cannot convert {type(item)!r} to pl.DataFrame")


def _to_pandas_df(item: Any) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is not available")
    if isinstance(item, pd.DataFrame):
        return item
    if pl is not None and isinstance(item, pl.DataFrame):
        return item.to_pandas()
    if isinstance(item, dict):
        return pd.DataFrame([item])
    if isinstance(item, Iterable):
        try:
            return pd.DataFrame(list(item))
        except Exception:
            pass
    raise TypeError(f"Cannot convert {type(item)!r} to pd.DataFrame")


@dataclass
class _Config:
    engine: str = "polars"  # "polars" | "pandas"
    queue_size: int = 100
    batch_size: int = 0  # 0 means "no batching" (emit per-item)
    batch_timeout_s: float = 0.0  # only used when batch_size > 0


class StreamingPipeline:
    def __init__(self, steps: Sequence[Any], config: dict[str, Any] | None = None):
        self.steps = list(steps)
        cfg = config or {}
        self.config = _Config(
            engine=cfg.get("engine", "polars"),
            queue_size=int(cfg.get("queue_size", 100)),
            batch_size=int(cfg.get("batch_size", 0)),
            batch_timeout_s=float(cfg.get("batch_timeout_s", 0.0)),
        )

    async def _producer(self, data_stream: AsyncIterator[Any], in_q: asyncio.Queue[Any]) -> None:
        try:
            async for item in data_stream:
                await in_q.put(item)
        except asyncio.CancelledError:
            # Don't try to put sentinel if we're cancelled
            raise
        except Exception:
            # For other exceptions, still try to put sentinel
            with contextlib.suppress(BaseException):
                await in_q.put(_SENTINEL)
            raise
        else:
            # Normal completion - always signal completion
            with contextlib.suppress(BaseException):
                await in_q.put(_SENTINEL)

    async def _apply_steps(self, data: Any, context: Any) -> Any:
        for step in self.steps:
            # tests call step.execute(data, context)
            result = step.execute(data, context)
            # Handle both sync and async execute methods
            if hasattr(result, "__await__"):
                data = await result
            else:
                data = result
        return data

    async def _consume_batched(
        self, context: Any, in_q: asyncio.Queue[Any], out_q: asyncio.Queue[Any]
    ) -> None:
        batch_size = self.config.batch_size
        timeout = self.config.batch_timeout_s if batch_size > 0 else 0.0
        engine = self.config.engine

        # Local accumulation for current batch
        items: list[Any] = []

        async def flush_if_any() -> None:
            if not items:
                return
            try:
                if engine == "polars":
                    if pl is None:
                        raise RuntimeError(
                            "polars is not available but engine='polars' was requested"
                        )
                    # Normalize to a list of polars DataFrames and vertically concat
                    frames = [_to_polars_df(x) for x in items]
                    try:
                        df = pl.concat(frames, how="vertical") if len(frames) > 1 else frames[0]
                    except pl.SchemaError:
                        # Handle schema mismatches by processing items individually
                        for frame in frames:
                            payload = await self._apply_steps(frame, context)
                            if isinstance(payload, pl.LazyFrame):
                                payload = payload.collect()
                            await out_q.put(payload)
                        return

                    payload = await self._apply_steps(df, context)
                    # Ensure we never emit LazyFrame
                    if isinstance(payload, pl.LazyFrame):
                        payload = payload.collect()
                    await out_q.put(payload)
                else:
                    if pd is None:
                        raise RuntimeError(
                            "pandas is not available but engine='pandas' was requested"
                        )
                    frames = [_to_pandas_df(x) for x in items]
                    pdf = (
                        pd.concat(frames, axis=0, ignore_index=True)
                        if len(frames) > 1
                        else frames[0]
                    )
                    payload = await self._apply_steps(pdf, context)
                    await out_q.put(payload)
            except Exception as e:
                # Put the exception on the output queue to be propagated
                await out_q.put(e)
                raise  # Re-raise to break the consumer loop
            finally:
                items.clear()

        while True:
            try:
                if timeout > 0:
                    item = await asyncio.wait_for(in_q.get(), timeout=timeout)
                else:
                    item = await in_q.get()
            except TimeoutError:
                # timeout => flush partial batch (if any) and continue waiting
                try:
                    await flush_if_any()
                except Exception:
                    break  # Exception already put on queue by flush_if_any
                continue

            if item is _SENTINEL:
                # Upstream completed: flush any remainder and exit
                try:
                    await flush_if_any()
                    await out_q.put(_SENTINEL)
                except Exception:
                    # Exception already put on queue by flush_if_any
                    pass
                break

            if batch_size <= 0:
                # Per-item mode
                try:
                    if engine == "polars":
                        df = _to_polars_df(item)
                        payload = await self._apply_steps(df, context)
                        if isinstance(payload, pl.LazyFrame):
                            payload = payload.collect()
                        await out_q.put(payload)
                    else:
                        pdf = _to_pandas_df(item)
                        payload = await self._apply_steps(pdf, context)
                        await out_q.put(payload)
                except Exception as e:
                    # Put the exception on the output queue to be propagated
                    await out_q.put(e)
                    break
            else:
                # Batched mode
                items.append(item)
                if len(items) >= batch_size:
                    try:
                        await flush_if_any()
                    except Exception:
                        break  # Exception already put on queue by flush_if_any

    async def run(self, data_stream: AsyncIterator[Any], context: Any) -> AsyncIterator[Any]:
        """Run producer/consumer and yield processed results as they become available."""
        in_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.config.queue_size)
        out_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.config.queue_size)

        producer_task = asyncio.create_task(self._producer(data_stream, in_q))
        consumer_task = asyncio.create_task(self._consume_batched(context, in_q, out_q))

        try:
            while True:
                # Check if either task has an exception
                if producer_task.done():
                    exc = producer_task.exception()
                    if exc is not None:
                        raise exc

                if consumer_task.done():
                    exc = consumer_task.exception()
                    if exc is not None:
                        raise exc

                result = await out_q.get()
                if result is _SENTINEL:
                    break
                if isinstance(result, Exception):
                    raise result
                yield result
        finally:
            # Ensure tasks are cancelled if still running
            for t in (producer_task, consumer_task):
                if not t.done():
                    t.cancel()

            # Wait for tasks to finish with timeout to avoid hanging
            done, pending = await asyncio.wait(
                [producer_task, consumer_task], timeout=1.0, return_when=asyncio.ALL_COMPLETED
            )

            # Check for exceptions in completed tasks
            for task in done:
                if task.exception() is not None and not isinstance(
                    task.exception(), asyncio.CancelledError
                ):
                    raise task.exception()

            # Force cancel any still pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass  # Ignore other exceptions during cleanup

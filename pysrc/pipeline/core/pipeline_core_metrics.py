# py/pipeline/core/pipeline_core_metrics.py
from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Callable
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.ops.observability import get_metrics  # your v2 module


# ---- Domain exceptions (precise) ------------------------------------------------
class PipelineMetricsError(Exception): ...


class MLflowUnavailable(PipelineMetricsError): ...


# ---- Instruments (lazy, via observability) -------------------------------------
_m = get_metrics()  # returns MetricsManager or None (e.g., tests)
_meter = _m.meter if _m else None

# Counters/Histograms/Gauges are obtained from Meter lazily; names remain stable
_STEP_TIME = (
    _m.histogram("pipeline_step_execution_ms", "Wall time spent in pipeline steps (ms)", "ms")
    if _m
    else None
)
_ERRORS = _m.counter("pipeline_errors_total", "Number of errors in pipeline") if _m else None
_STR_LAT = (
    _m.histogram("streaming_step_latency_ms", "Latency per cleaning step (ms)", "ms")
    if _m
    else None
)
_BUF_LEN = (
    _m.meter.create_observable_gauge(
        "streaming_buffer_length", callbacks=[], description="Current length of streaming buffer"
    )
    if _m
    else None
)
_DATA_VOL = (
    _m.meter.create_observable_gauge(
        "data_volume_processed", callbacks=[], description="Records processed"
    )
    if _m
    else None
)

# Back-compat exports expected by pysrc.pipeline.core.__init__
ERROR_COUNTER = _ERRORS
STEP_EXECUTION_TIME = _STEP_TIME

# Optional MLflow (LLM-ready)
mlflow = optional_import("mlflow")  # None if not installed/disabled


class AsyncMLflowLogger:
    """
    Contract:
      - Non-blocking logging using thread pool offloads.
      - Precise exceptions: raises MLflowUnavailable when MLflow is not available.
      - Stable method set so tests assert behavior, not implementation.
    """

    def __init__(self) -> None:
        if not mlflow:
            raise MLflowUnavailable("mlflow is not available")

    async def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        loop = asyncio.get_running_loop()

        def _log() -> None:
            mlflow.log_metrics(metrics, step=step)  # may raise ValueError/RuntimeError

        await loop.run_in_executor(None, _log)

    async def log_params(self, params: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()

        def _log() -> None:
            mlflow.log_params(params)  # may raise ValueError/RuntimeError

        await loop.run_in_executor(None, _log)

    async def set_experiment(self, name: str) -> None:
        loop = asyncio.get_running_loop()

        def _set() -> None:
            mlflow.set_experiment(name)

        await loop.run_in_executor(None, _set)

    async def start_run(
        self,
        *,
        run_name: str | None = None,
        nested: bool = True,
        tags: dict[str, str] | None = None,
    ) -> Any:
        # Return the active run so callers can attach IDs if needed.
        loop = asyncio.get_running_loop()

        def _start():
            return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)

        return await loop.run_in_executor(None, _start)

    async def end_run(self) -> None:
        loop = asyncio.get_running_loop()

        def _end() -> None:
            mlflow.end_run()

        await loop.run_in_executor(None, _end)

    async def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        loop = asyncio.get_running_loop()

        def _log() -> None:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

        await loop.run_in_executor(None, _log)


# Backwards-compatible helper kept for existing call sites
async def _mlflow_log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if not mlflow:
        raise MLflowUnavailable("mlflow is not available")
    logger = AsyncMLflowLogger()
    await logger.log_metrics(metrics, step=step)


# ---- Contract-driven decorator (sync/async) ------------------------------------
def track_step_execution(step_name: str, *, stage: str = "unknown", engine: str = "unknown"):
    """
    Contract:
      - Record latency (ms) and error counter per step/stage/engine.
      - Works for sync and async callables.
      - Keeps labels stable to protect metric cardinality.
    """

    def deco(fn: Callable[..., Any]):
        labels = {"step": step_name, "stage": stage, "engine": engine}

        async def _run_async(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            except (ValueError, TypeError, RuntimeError):
                if _m and _ERRORS:
                    _m.record_counter(_ERRORS, 1, labels=labels)
                raise
            finally:
                if _m and _STEP_TIME:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    _m.record_histogram(_STEP_TIME, dt_ms, labels=labels)

        def _run_sync(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            except (ValueError, TypeError, RuntimeError):
                if _m and _ERRORS:
                    _m.record_counter(_ERRORS, 1, labels=labels)
                raise
            finally:
                if _m and _STEP_TIME:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    _m.record_histogram(_STEP_TIME, dt_ms, labels=labels)

        return _run_async if inspect.iscoroutinefunction(fn) else _run_sync

    return deco


# ---- Streaming infra (expose gauges via callbacks) ---------------------------
# Expose user-provided callables to report buffer length and volume.
# Call `wire_streaming_observers(get_len, get_volume)` once at startup.
def wire_streaming_observers(
    get_buffer_len: Callable[[], int] | None, get_processed_volume: Callable[[], int] | None
) -> None:
    if not _m or not _meter:
        return

    callbacks = []
    if get_buffer_len:

        def _buf_len(_):
            try:
                return [(get_buffer_len(),)]  # Observation(value) form, but compat shim:
            except (ValueError, TypeError, RuntimeError):
                return []

        callbacks.append(
            lambda _opts: (
                [_m.metrics.Observation(get_buffer_len(), {"what": "buffer"})]
                if hasattr(_m, "metrics")
                else []
            )
        )

    if get_processed_volume:
        callbacks.append(
            lambda _opts: (
                [_m.metrics.Observation(get_processed_volume(), {"what": "volume"})]
                if hasattr(_m, "metrics")
                else []
            )
        )

    # Replace gauge callbacks atomically
    if _BUF_LEN:
        with contextlib.suppress(IndexError, ValueError, TypeError):
            _meter.create_observable_gauge("streaming_buffer_length", callbacks=[callbacks[0]])
    if _DATA_VOL and len(callbacks) > 1:
        with contextlib.suppress(ValueError, TypeError):
            _meter.create_observable_gauge("data_volume_processed", callbacks=[callbacks[1]])

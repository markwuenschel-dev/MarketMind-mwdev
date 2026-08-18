# py/pipeline/stages/cleaning/core/pipeline_core_metrics.py
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.ops.observability import get_metrics


class CleaningMetricsError(Exception): ...


class MLflowUnavailable(CleaningMetricsError): ...


_m = get_metrics()
_meter = _m.meter if _m else None

# Instruments (names preserved from the original cleaning metrics file)
# streaming_step_latency → histogram (ms)
_STEP_LAT = (
    _m.histogram("streaming_step_latency_ms", "Latency per cleaning step (ms)", "ms")
    if _m
    else None
)

# Gauges become observable gauges fed by callbacks registered at boot
# streaming_latency, buffer_length, data_volume
if _m and _meter:
    try:
        _meter.create_observable_gauge(
            "streaming_cleaner_latency", callbacks=[], description="Latency of streaming cleaner"
        )
        _meter.create_observable_gauge(
            "streaming_buffer_length",
            callbacks=[],
            description="Current length of streaming buffer",
        )
        _meter.create_observable_gauge(
            "data_volume_processed", callbacks=[], description="Number of records processed"
        )
    except (ValueError, TypeError):
        pass

mlflow = optional_import("mlflow")


class AsyncMLflowLogger:
    """
    Non-blocking MLflow logger for cleaning pipelines.

    Contract:
      - Constructor raises MLflowUnavailable if MLflow is not present.
      - All operations are offloaded to a thread pool (do not block event loop).
      - Uses precise, explicit exception propagation from MLflow (ValueError/RuntimeError).
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
            mlflow.log_params(params)

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


# Backward-compatible helper
async def log_metric_mlflow(key: str, value: Any) -> None:
    if not mlflow:
        raise MLflowUnavailable("mlflow is not available")
    logger = AsyncMLflowLogger()
    await logger.log_metrics({key: float(value)})


def record_step_latency_ms(value_ms: float, *, step: str = "unknown") -> None:
    if _m and _STEP_LAT:
        _m.record_histogram(_STEP_LAT, float(value_ms), labels={"step": step})


def wire_cleaner_observers(
    *,
    get_latency: Callable[[], float] | None,
    get_buffer_len: Callable[[], int] | None,
    get_data_volume: Callable[[], int] | None,
) -> None:
    if not _m or not _meter:
        return
    if get_latency:
        with contextlib.suppress(ValueError, TypeError):
            _meter.create_observable_gauge(
                "streaming_cleaner_latency",
                callbacks=[
                    lambda _opts: [_m.metrics.Observation(get_latency(), {"what": "latency"})]
                ]
                if hasattr(_m, "metrics")
                else [],
            )
    if get_buffer_len:
        with contextlib.suppress(ValueError, TypeError):
            _meter.create_observable_gauge(
                "streaming_buffer_length",
                callbacks=[
                    lambda _opts: [_m.metrics.Observation(get_buffer_len(), {"what": "buffer"})]
                ]
                if hasattr(_m, "metrics")
                else [],
            )
    if get_data_volume:
        with contextlib.suppress(ValueError, TypeError):
            _meter.create_observable_gauge(
                "data_volume_processed",
                callbacks=[
                    lambda _opts: [_m.metrics.Observation(get_data_volume(), {"what": "volume"})]
                ]
                if hasattr(_m, "metrics")
                else [],
            )

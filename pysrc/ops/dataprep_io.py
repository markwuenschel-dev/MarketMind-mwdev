# I/O coordination for dataprep: stage guard, timeouts, retries, workers, memory/GPU metrics.
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import

psutil = optional_import("psutil")
pynvml = optional_import("pynvml")


def maybe_mem_info(_ctx: Any = None) -> dict[str, Any]:
    """Collect memory metrics; used by stage timing."""
    if psutil is None:
        return {}
    try:
        vm = psutil.virtual_memory()
        out: dict[str, Any] = {
            "vm_total": int(getattr(vm, "total", 0)),
            "vm_used": int(getattr(vm, "used", 0)),
            "vm_available": int(getattr(vm, "available", 0)),
        }
        try:
            proc = psutil.Process()
            rss = int(getattr(proc.memory_info(), "rss", 0))
            out["rss_mb"] = int(rss // (1024 * 1024))
            total = out.get("vm_total") or 0
            out["mem_pct"] = float((rss / total) * 100.0) if total else 0.0
        except (AttributeError, OSError, psutil.NoSuchProcess):
            pass
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                n = int(pynvml.nvmlDeviceGetCount())
                out["gpu_count"] = n
                used = 0
                for i in range(n):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    used += int(getattr(m, "used", 0))
                out["gpu_mem_used"] = used
                pynvml.nvmlShutdown()
            except (AttributeError, OSError):
                pass
        return out
    except (AttributeError, OSError, ValueError):
        return {}


def resolve_workers(val: int | str) -> int:
    """Resolve worker count from config value (int or 'auto')."""
    if isinstance(val, int):
        return max(1, val)
    if isinstance(val, str) and val.lower() == "auto":
        try:
            cores = os.cpu_count() or 4
            return max(1, cores - 1)
        except (ValueError, TypeError):
            return 4
    try:
        return int(val)
    except (ValueError, TypeError):
        return 1


def call_with_timeout(fn: Callable[[], Any], timeout_s: int | None) -> Any:
    """Run fn in a thread with optional timeout."""
    if timeout_s is None or timeout_s <= 0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut: Future = ex.submit(fn)
        return fut.result(timeout=timeout_s)


def stage_with_guard(
    name: str,
    fn: Callable[[], Any],
    *,
    timeout_s: int | None = None,
    run_cfg: Mapping[str, Any],
    run_id: str | None,
    metrics: dict[str, Any],
    logger: Any,
    ConfigError: type,
    DataPrepError: type,
    get_metrics_fn: Callable,
) -> Any:
    """Execute stage with retry, timeout, and timing; append to metrics['stages']."""
    attempts = int(run_cfg.get("error_handling", {}).get("retry_policy", {}).get("max_attempts", 1))
    backoff0 = int(
        run_cfg.get("error_handling", {}).get("retry_policy", {}).get("initial_backoff_seconds", 1)
    )
    backoff_max = int(
        run_cfg.get("error_handling", {}).get("retry_policy", {}).get("max_backoff_seconds", 8)
    )
    attempt = 0
    last_exc: BaseException | None = None
    t_start = time.perf_counter()
    while attempt < max(1, attempts):
        attempt += 1
        try:
            out = call_with_timeout(fn, timeout_s)
            duration = time.perf_counter() - t_start
            rec = {"name": name, "duration_s": duration, "attempts": attempt, **maybe_mem_info()}
            metrics.setdefault("stages", []).append(rec)
            logger.info(
                "stage complete",
                extra={
                    "run_id": run_id,
                    "stage": name,
                    "duration_s": duration,
                    "attempts": attempt,
                    **maybe_mem_info(),
                },
            )
            return out
        except Exception as exc:
            if isinstance(exc, ConfigError):
                raise
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_s = min(backoff_max, backoff0 * (2 ** (attempt - 1)))
            try:
                mm = get_metrics_fn()
                if mm and hasattr(mm, "histogram") and hasattr(mm, "record_histogram"):
                    h = mm.histogram("dataprep_stage_retry", "Stage retry backoff", "s")
                    mm.record_histogram(h, sleep_s, labels={"stage": name, "attempt": attempt})
            except (AttributeError, TypeError, ValueError):
                pass
            logger.warning(
                "stage retry",
                extra={
                    "run_id": run_id,
                    "stage": name,
                    "attempt": attempt,
                    "sleep_s": sleep_s,
                    "err": str(exc),
                },
            )
            time.sleep(sleep_s)
    duration = time.perf_counter() - t_start
    metrics.setdefault("stages", []).append(
        {"name": name, "duration_s": duration, "attempts": attempt, "status": "failed"}
    )
    raise DataPrepError(f"Stage '{name}' failed after {attempt} attempts: {last_exc}")


def adaptive_map(
    fn: Callable,
    items: Iterator[Any],
    kind: str = "thread",
    max_workers: int = 4,
) -> Iterator[Any]:
    """Adaptive parallel map (thread pool by default)."""
    pool = ThreadPoolExecutor(max_workers=max_workers)
    with pool as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            yield fut.result()

# utils/cuda_runtime.py
from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import dataclass

from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)

try:
    import rmm
except ImportError:
    rmm = None
try:
    import cupy
except ImportError:
    cupy = None
try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None
try:
    import cudf
except ImportError:
    cudf = None
try:
    import polars
except ImportError:
    polars = None
try:
    import kvikio
except ImportError:
    kvikio = None


@dataclass(frozen=True)
class GpuCapabilities:
    has_cuda: bool
    has_rmm: bool
    has_cudf: bool
    has_polars_gpu: bool
    has_nvtabular: bool
    has_kvikio: bool
    device_count: int = 0
    compute_capability: str | None = None


_CAPS: GpuCapabilities | None = None


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def capabilities() -> GpuCapabilities:
    global _CAPS
    if _CAPS is not None:
        return _CAPS
    has_cuda = bool(cupy and cupy.cuda.is_available()) or bool(
        numba_cuda and numba_cuda.is_available()
    )
    dev_count = 0
    cc = None
    if has_cuda:
        try:
            if cupy:
                dev_count = cupy.cuda.runtime.getDeviceCount()
                with cupy.cuda.Device(0):
                    props = cupy.cuda.runtime.getDeviceProperties(0)
                    cc = f"{props['major']}.{props['minor']}"
            elif numba_cuda:
                dev = numba_cuda.get_current_device()
                dev_count = len(numba_cuda.gpus)
                cc = f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
        except Exception:
            pass
    has_polars_gpu = bool(polars and hasattr(polars, "GpuEngine"))
    has_nvtabular = bool(_try_import("nvtabular"))
    _CAPS = GpuCapabilities(
        has_cuda=has_cuda,
        has_rmm=bool(rmm),
        has_cudf=bool(cudf),
        has_polars_gpu=has_polars_gpu,
        has_nvtabular=has_nvtabular,
        has_kvikio=bool(kvikio),
        device_count=dev_count,
        compute_capability=cc,
    )
    return _CAPS


def init_rmm_pool(
    pool_size: int | None = None,
    managed_memory: bool = False,
    async_alloc: bool = True,
    logging: bool = False,
    release_threshold: int | None = None,
) -> None:
    """Version-safe RMM init + allocator bindings (CuPy, cuDF)."""
    if not rmm or rmm.is_initialized():
        return
    kwargs = {
        "pool_allocator": True,
        "managed_memory": managed_memory,
        "initial_pool_size": pool_size or (1 << 30),
        "enable_logging": logging,
    }
    if release_threshold is not None:
        kwargs["release_threshold"] = release_threshold
    # Prefer new API; fallback to legacy
    if async_alloc:
        kwargs_try = dict(kwargs)
        kwargs_try["allocation_mode"] = "async"
    else:
        kwargs_try = dict(kwargs)
    try:
        rmm.reinitialize(**kwargs_try)
    except TypeError:
        if async_alloc:
            kwargs_fallback = dict(kwargs)
            kwargs_fallback["use_async_allocator"] = True
        else:
            kwargs_fallback = dict(kwargs)
        rmm.reinitialize(**kwargs_fallback)
    except Exception as e:
        logger.error("RMM init failed: %s", e)
        raise
    # Bind CuPy to RMM
    if cupy:
        try:
            cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
        except Exception as e:
            logger.debug("CuPy allocator bind failed: %s", e)
    # Bind cuDF / libcudf memory resource if available
    try:
        import libcudf

        libcudf.memory_resource.set_current_device_resource(rmm.get_current_device_resource())
    except Exception as e:
        logger.debug("libcudf memory resource bind skipped/failed: %s", e)


class StreamFactory:
    @staticmethod
    def create(non_blocking: bool = True):
        caps = capabilities()
        if cupy and caps.has_cuda:
            return cupy.cuda.Stream(non_blocking=non_blocking)
        if numba_cuda and caps.has_cuda:
            return numba_cuda.stream()
        return None


class StreamPool:
    def __init__(self, size: int = 4, non_blocking: bool = True):
        self._streams = deque()
        self._factory = StreamFactory
        for _ in range(size):
            s = self._factory.create(non_blocking)
            if s:
                self._streams.append(s)

    @contextlib.contextmanager
    def lease(self):
        if not self._streams:
            yield None
            return
        s = self._streams.popleft()
        try:
            yield s
        finally:
            self._streams.append(s)


def pinned_array(shape, dtype="float32"):
    if cupy:
        return cupy.empty_pinned(shape, dtype=dtype, order="C")
    import numpy as np

    return np.empty(shape, dtype=dtype)


def device_synchronize():
    caps = capabilities()
    if cupy and caps.has_cuda:
        cupy.cuda.runtime.deviceSynchronize()
    elif numba_cuda and caps.has_cuda:
        numba_cuda.synchronize()


@contextlib.contextmanager
def maybe_stream(stream):
    if stream is None:
        yield
        return
    if cupy and hasattr(cupy.cuda, "Stream") and isinstance(stream, cupy.cuda.Stream):
        with stream.use():
            yield
    elif numba_cuda and hasattr(stream, "auto_synchronize"):
        with stream.auto_synchronize():
            yield
    else:
        yield

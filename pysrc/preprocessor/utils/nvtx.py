# utils/nvtx.py
from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable

from pysrc.ops.mm_logkit import get_logger

# ... rest of the file
logger = get_logger(__name__)

try:
    import nvtx as _nvtx
except Exception:
    _nvtx = None

# Single, consistent flag (pure no-op when disabled)
NVTX_ENABLED = bool(_nvtx) and os.getenv("NVTX_ENABLED", "0") in {"1", "true", "TRUE"}


@contextlib.contextmanager
def range_ctx(message: str, color: int | None = None):
    if NVTX_ENABLED and _nvtx:
        with _nvtx.annotate(message=message, color=color or 0x33AAFF):
            yield
    else:
        # strict no-op (zero overhead)
        yield


def nvtx_range(message: str, color: int | None = None):
    """Decorator: annotate a function call with an NVTX range when enabled."""

    def deco(fn: Callable) -> Callable:
        if not NVTX_ENABLED or not _nvtx:
            return fn  # pure pass-through

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _nvtx.annotate(message=message, color=color or 0x33AAFF):
                return fn(*args, **kwargs)

        return wrapper

    return deco


class nvtx_plan:
    """Context manager for ad-hoc plan-level annotation."""

    def __init__(self, name: str, color: int | None = None):
        self.name = name
        self.color = color or 0x88CC66
        self._ctx = None

    def __enter__(self):
        if NVTX_ENABLED and _nvtx:
            self._ctx = _nvtx.annotate(message=self.name, color=self.color)
            self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if NVTX_ENABLED and self._ctx:
            self._ctx.__exit__(exc_type, exc, tb)

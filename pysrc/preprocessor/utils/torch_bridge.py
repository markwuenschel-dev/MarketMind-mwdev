# utils/torch_bridge.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)

try:
    import torch
except Exception:
    # Torch is optional; some environments ship a broken install that raises
    # at import time (e.g. docstring registration issues). Treat this as
    # "no torch available" so non-ML paths still work.
    torch = None
try:
    import cupy
except ImportError:
    cupy = None
try:
    import cudf
except ImportError:
    cudf = None
try:
    import polars as pl
except ImportError:
    pl = None


@dataclass
class TorchBatch:
    tensors: dict[str, torch.Tensor]
    lengths: torch.Tensor | None = None
    meta: dict[str, Any] = None


class BackendBridge(ABC):
    @abstractmethod
    def to_torch(
        self, df, cols: Sequence[str], dtypes: dict[str, torch.dtype] | None, include_lengths: bool
    ) -> TorchBatch:
        pass


class CuDFBridge(BackendBridge):
    def to_torch(
        self, df, cols: Sequence[str], dtypes: dict[str, torch.dtype] | None, include_lengths: bool
    ) -> TorchBatch:
        if not cols:
            return TorchBatch(tensors={}, meta={"backend": "cudf"})
        dtypes = dtypes or {}
        tensors = {c: self._series_to_torch(df[c], dtypes.get(c)) for c in cols}
        lengths = None
        if include_lengths and torch:
            lengths = torch.full(
                (len(df),), 1, dtype=torch.int32, device=next(iter(tensors.values())).device
            )
        return TorchBatch(tensors=tensors, lengths=lengths, meta={"backend": "cudf"})

    def _series_to_torch(self, series, dtype=None):
        if torch is None:
            raise RuntimeError("PyTorch not available")
        if hasattr(series, "to_dlpack"):
            dlpack = series.to_dlpack()
            t = torch.utils.dlpack.from_dlpack(dlpack)
            return t.to(dtype) if dtype else t
        raise TypeError("Cannot convert to DLPack")


class PolarsBridge(BackendBridge):
    def to_torch(
        self, df, cols: Sequence[str], dtypes: dict[str, torch.dtype] | None, include_lengths: bool
    ) -> TorchBatch:
        if not cols:
            return TorchBatch(tensors={}, meta={"backend": "polars"})
        if torch is None:
            raise RuntimeError("PyTorch not available")
        dtypes = dtypes or {}
        tensors = {}
        for c in cols:
            # Best-effort: Polars -> NumPy; move to CUDA if available.
            try:
                arr = df[c].to_numpy(allow_copy=False)
            except RuntimeError:
                arr = df[c].to_numpy(allow_copy=True)
            # Writable copy: Polars can yield read-only buffers PyTorch rejects.
            arr = np.array(arr, copy=True, dtype=arr.dtype)
            t = torch.as_tensor(
                arr,
                device="cuda" if (torch and torch.cuda.is_available()) else "cpu",
            )
            if c in dtypes:
                t = t.to(dtypes[c])
            tensors[c] = t
        lengths = None
        if include_lengths and torch:
            lengths = torch.full(
                (len(df),), 1, dtype=torch.int32, device=next(iter(tensors.values())).device
            )
        return TorchBatch(tensors=tensors, lengths=lengths, meta={"backend": "polars"})


def bridge_factory(df) -> BackendBridge:
    if cudf and isinstance(df, cudf.DataFrame):
        return CuDFBridge()
    if pl and isinstance(df, pl.DataFrame):
        return PolarsBridge()
    raise ValueError("Unsupported backend")


def profile_evolve(func: Callable) -> Callable:
    metrics = {}  # Self-evolving: cache timings to choose faster paths

    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start
        key = str(type(args[0]))  # E.g., backend type
        if key not in metrics or duration < metrics[key]:
            metrics[key] = duration
            logger.info(f"Evolved: Best time for {key} is {duration}")
        return result

    return wrapper


@profile_evolve
def to_torch_batch(
    df,
    cols: Sequence[str],
    dtypes: dict[str, torch.dtype] | None = None,
    include_lengths: bool = False,
) -> TorchBatch:
    bridge = bridge_factory(df)
    return bridge.to_torch(df, cols, dtypes, include_lengths)


def set_amp_precision(precision: str = "bf16"):
    if torch and torch.cuda.is_available():
        return {"precision": precision}
    return {"precision": "fp32"}


def seed_everything(seed: int = 1337):
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

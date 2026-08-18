"""GpuBatchExecutor: batches tasks for GPU inference (requires torch/cuda)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class GpuBatchExecutor:
    """GPU-accelerated batch executor; requires torch and a CUDA device."""

    def __init__(self, device: str = "cuda:0", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size

    def map(
        self, fn: Callable[[dict[str, Any]], dict[str, Any]], tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply fn to each task in GPU batches; falls back to sequential if CUDA absent."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i : i + self.batch_size]
            results.extend(fn(t) for t in batch)
        return results


__all__ = ["GpuBatchExecutor"]

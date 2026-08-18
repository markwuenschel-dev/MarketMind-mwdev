"""MultiprocessingExecutor: runs tasks in a process pool."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class MultiprocessingExecutor:
    """Parallel task executor using a process pool."""

    def __init__(self, n_workers: int = 4) -> None:
        self.n_workers = n_workers

    def map(
        self, fn: Callable[[dict[str, Any]], dict[str, Any]], tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply fn to each task in parallel; return results in submission order."""
        import multiprocessing

        with multiprocessing.Pool(self.n_workers) as pool:
            return list(pool.map(fn, tasks))


__all__ = ["MultiprocessingExecutor"]

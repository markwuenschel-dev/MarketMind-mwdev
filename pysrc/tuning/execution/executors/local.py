"""LocalExecutor: runs tasks sequentially in the current process."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LocalExecutor:
    """Single-process, sequential task executor."""

    def map(
        self, fn: Callable[[dict[str, Any]], dict[str, Any]], tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply fn to each task dict sequentially; return results in order."""
        return [fn(t) for t in tasks]


__all__ = ["LocalExecutor"]

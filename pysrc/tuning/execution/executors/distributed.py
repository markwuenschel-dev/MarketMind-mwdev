"""DistributedExecutor: delegates to a registered distributed backend (Ray / Dask)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DistributedExecutor:
    """Distributed task executor; backend resolved from environment at runtime."""

    def map(
        self, fn: Callable[[dict[str, Any]], dict[str, Any]], tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Submit tasks to the distributed backend; return results in order."""
        raise NotImplementedError(
            "DistributedExecutor.map must be configured with a Ray or Dask cluster"
        )


__all__ = ["DistributedExecutor"]

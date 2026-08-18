"""OnlineFeatureBuffer: accumulates streaming observations for online feature computation."""

from __future__ import annotations

from collections import deque
from typing import Any


class OnlineFeatureBuffer:
    """Fixed-capacity ring buffer of streaming feature observations."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)

    def push(self, observation: dict[str, Any]) -> None:
        """Append one observation to the buffer."""
        self._buf.append(observation)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of the current buffer contents (oldest first)."""
        return list(self._buf)

    @property
    def size(self) -> int:
        """Return the number of observations currently buffered."""
        return len(self._buf)


__all__ = ["OnlineFeatureBuffer"]

"""StreamListener: consumes a real-time data stream and emits typed events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

EventCallback = Callable[[dict[str, Any]], None]


class StreamListener:
    """Subscribes to a data stream and dispatches raw events to registered callbacks."""

    def __init__(self) -> None:
        self._callbacks: list[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> None:
        """Register a callback to receive each incoming event."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Begin consuming the stream; implementation-specific."""
        raise NotImplementedError(
            "StreamListener.start must be wired to a Kafka, Redis Streams, or WebSocket source"
        )

    def stop(self) -> None:
        """Halt stream consumption."""
        raise NotImplementedError("StreamListener.stop requires a live connection")

    def _dispatch(self, event: dict[str, Any]) -> None:
        for cb in self._callbacks:
            cb(event)


__all__ = ["EventCallback", "StreamListener"]

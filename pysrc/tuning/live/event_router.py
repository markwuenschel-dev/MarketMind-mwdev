"""EventRouter: routes typed tuning events to registered handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TuningEvent:
    """A typed event emitted by the live stream or internal triggers."""

    kind: str  # e.g. "drift_detected", "retrain_requested", "model_switched"
    payload: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[TuningEvent], None]


class EventRouter:
    """Routes TuningEvents to kind-specific handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}

    def register(self, kind: str, handler: Handler) -> None:
        """Register a handler for events of the given kind."""
        self._handlers.setdefault(kind, []).append(handler)

    def route(self, event: TuningEvent) -> None:
        """Dispatch event to all registered handlers for its kind."""
        for handler in self._handlers.get(event.kind, []):
            handler(event)


__all__ = ["TuningEvent", "Handler", "EventRouter"]

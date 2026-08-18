"""CommandBus: typed command dispatch for orchestration events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Command:
    """A named orchestration command with an opaque payload."""

    name: str
    payload: dict[str, Any]


Handler = Callable[[Command], None]


class CommandBus:
    """Synchronous in-process command bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}

    def register(self, command_name: str, handler: Handler) -> None:
        """Register a handler for a named command."""
        self._handlers.setdefault(command_name, []).append(handler)

    def dispatch(self, command: Command) -> None:
        """Dispatch *command* to all registered handlers."""
        for handler in self._handlers.get(command.name, []):
            handler(command)


__all__ = ["Command", "Handler", "CommandBus"]

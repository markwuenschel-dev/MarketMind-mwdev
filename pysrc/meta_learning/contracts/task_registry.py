"""TaskRegistry contract stubs for Phase II meta-learning surfaces."""

from __future__ import annotations

from typing import ClassVar, Protocol


class MetaTask(Protocol):
    task_id: str
    regime_id: str
    t0: str


# F-5 DEFERRED: persistence D-tier for TaskRegistry artifact writes
# (task_manifest role in py/artifact_registry/contracts.py) declared in
# OI-23 / GATE-I-F-05.


class TaskRegistryError(Exception):
    """Base typed exception for TaskRegistry contract violations."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class TaskNotFoundError(TaskRegistryError):
    """Raised when a requested task_id is not present."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message, details=details)


class TaskRegistryDuplicateError(TaskRegistryError):
    """Raised on append when task_id or stable identity already exists."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message, details=details)


class TaskRegistryProtocol(Protocol):
    """Append-only MetaTask registry contract.

    Implementations must reject duplicate task_id values. Mutation of existing
    tasks is not permitted.

    Implementations must not store or return any task whose pit_boundary was
    derived from data beyond the governed DataView.as_of(T) boundary.
    """

    CONTRACT_VERSION: ClassVar[str] = "v1"

    def append(self, task: MetaTask) -> None:
        """Append-only. Raises TaskRegistryError if task_id already exists."""
        ...

    def get(self, task_id: str) -> MetaTask:
        """Raises TaskNotFoundError if not present."""
        ...

    def query(
        self,
        regime_id: str | None = None,
        since: str | None = None,
    ) -> list[MetaTask]:
        """Filter tasks; ``since`` is ISO 8601 UTC comparable to ``MetaTask.t0``."""
        ...

    def __len__(self) -> int: ...


__all__ = [
    "TaskNotFoundError",
    "TaskRegistryDuplicateError",
    "TaskRegistryError",
    "TaskRegistryProtocol",
]

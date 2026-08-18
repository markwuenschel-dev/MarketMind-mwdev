"""MLC-0 · Promotable append-only :class:`TaskRegistry`.

Canonical path resolution (MLC-0 Step 1)
----------------------------------------

``MetaLearningCore.md`` §5.5 specifies ``py/meta/task_registry.py``.  The
repo's Python root is ``pysrc/``; therefore the canonical in-repo path
is ``pysrc/meta/task_registry.py``.

Contract authority
------------------

Implements :class:`pysrc.meta_learning.contracts.task_registry.TaskRegistryProtocol`
(OI-22 stub).  The protocol is the authoritative interface surface; this
module is the promotable concrete implementation.

Semantics
---------

- **Append-only.**  Once a task is registered it cannot be mutated or
  removed.  Re-use of a ``(regime_id, t0)`` stable key raises
  :class:`TaskRegistryDuplicateError`.  Duplicate ``task_id`` values
  raise the same exception.
- **Durable backend.**  A :class:`DurableTaskStore` protocol lets the
  registry persist each appended task to disk.  The default in-repo
  backend is :class:`JsonLinesDurableStore` — a minimal JSON-Lines
  append-only writer sufficient to make the interface stable for
  downstream MLC-1..MLC-3.  In-memory registry behaviour is fully
  covered even when no durable backend is attached (``None`` ⇒
  in-memory-only, audited via log message on append).
- **Logging.**  Every successful append emits a structured log record
  via :mod:`pysrc.ops.mm_logkit` so that operators can tail registry
  activity without opaque silence.

Out of scope (per MLC-0 brief §7)
---------------------------------

- ``task_manifest.json`` emission.
- CAS-backed durable store with cross-run replay guarantees (MLC-6).
- Governed ``build_meta_task`` constructor (MLC-2).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.task import MetaTask
from pysrc.meta_learning.contracts.task_registry import (
    TaskNotFoundError,
    TaskRegistryDuplicateError,
    TaskRegistryError,
    TaskRegistryProtocol,
)
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

__all__ = [
    "DurableTaskStore",
    "JsonLinesDurableStore",
    "NullDurableStore",
    "TaskNotFoundError",
    "TaskRegistry",
    "TaskRegistryDuplicateError",
    "TaskRegistryError",
    "TaskRegistryProtocol",
]


class DurableTaskStore(Protocol):
    """Minimal durable-store interface for :class:`TaskRegistry`.

    Implementations must be append-only and must reject any attempt to
    mutate a previously persisted task record.
    """

    def persist(self, task_id: str, record: dict[str, Any]) -> None:
        """Append ``record`` for ``task_id`` to the durable store."""
        ...

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield previously persisted records in insertion order."""
        ...


class NullDurableStore:
    """No-op durable store; used when the registry is in-memory-only."""

    def persist(self, task_id: str, record: dict[str, Any]) -> None:  # noqa: ARG002
        return None

    def iter_records(self) -> Iterator[dict[str, Any]]:
        return iter(())


class JsonLinesDurableStore:
    """Append-only JSON Lines durable store for :class:`TaskRegistry`.

    The file is opened in append mode on every ``persist`` call and
    flushed + ``fsync`` -ed before returning.  This is a minimal
    durability guarantee suitable for MLC-0; richer CAS-backed stores
    are deferred to MLC-6 per ResolutionLedger OI-23.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    @property
    def path(self) -> Path:
        return self._path

    def persist(self, task_id: str, record: dict[str, Any]) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise DataPreconditionError(
                "JsonLinesDurableStore.persist requires non-empty task_id",
                details={"task_id": repr(task_id)},
            )
        if not isinstance(record, dict):
            raise DataPreconditionError(
                "JsonLinesDurableStore.persist requires dict record",
                details={"record_type": type(record).__name__},
            )
        payload = json.dumps(
            {"task_id": task_id, "record": record},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if not self._path.exists():
            return iter(())
        return self._iter_file(self._path)

    @staticmethod
    def _iter_file(path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


class TaskRegistry(TaskRegistryProtocol):
    """Append-only MetaTask registry keyed by ``(regime_id, t0)`` and ``task_id``.

    Implements :class:`TaskRegistryProtocol` (OI-22 stub).  Mutation of
    previously registered tasks is impossible: ``MetaTask`` is frozen
    and the registry never hands out writable references.
    """

    CONTRACT_VERSION: ClassVar[str] = "mlc0.v1"

    def __init__(self, durable_store: DurableTaskStore | None = None) -> None:
        self._by_id: dict[str, MetaTask] = {}
        self._by_stable: dict[tuple[str, str], str] = {}
        self._insertion_order: list[str] = []
        self._durable_store: DurableTaskStore = durable_store or NullDurableStore()

    # ------------------------------------------------------------------
    # Append-only write path
    # ------------------------------------------------------------------

    def append(self, task: MetaTask) -> None:  # type: ignore[override]
        """Register ``task``; raise on duplicate stable key or ``task_id``.

        Duplicate detection is fail-closed: the first duplicate path
        encountered raises immediately.  No silent overwrite.  Log
        record ``task_registry_append`` emitted on success.
        """
        if not isinstance(task, MetaTask):
            raise DataPreconditionError(
                "TaskRegistry.append requires a MetaTask instance",
                details={"type": type(task).__name__},
            )

        if task.task_id in self._by_id:
            raise TaskRegistryDuplicateError(
                "task_id already registered (append-only)",
                details={"task_id": task.task_id},
            )
        stable = (task.regime_id, task.t0)
        if stable in self._by_stable:
            raise TaskRegistryDuplicateError(
                "stable identity (regime_id, t0) already registered (append-only)",
                details={
                    "regime_id": task.regime_id,
                    "t0": task.t0,
                    "existing_task_id": self._by_stable[stable],
                },
            )

        self._by_id[task.task_id] = task
        self._by_stable[stable] = task.task_id
        self._insertion_order.append(task.task_id)

        record = task.as_record()
        self._durable_store.persist(task.task_id, record)

        LOG.info(
            "task_registry_append",
            task_id=task.task_id,
            regime_id=task.regime_id,
            regime_class=task.regime_class,
            t0=task.t0,
            t1=task.t1,
            active_k=task.active_k,
            signal_set_version=task.signal_set_version,
        )

    # ------------------------------------------------------------------
    # Read / query path
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> MetaTask:  # type: ignore[override]
        if task_id not in self._by_id:
            raise TaskNotFoundError(
                "task not found",
                details={"task_id": task_id},
            )
        return self._by_id[task_id]

    def get_by_stable(self, *, regime_id: str, t0: str) -> MetaTask:
        key = (regime_id, t0)
        if key not in self._by_stable:
            raise TaskNotFoundError(
                "task not found for stable key",
                details={"regime_id": regime_id, "t0": t0},
            )
        return self._by_id[self._by_stable[key]]

    def contains_stable(self, *, regime_id: str, t0: str) -> bool:
        return (regime_id, t0) in self._by_stable

    def query(  # type: ignore[override]
        self,
        regime_id: str | None = None,
        since: str | None = None,
    ) -> list[MetaTask]:
        """Filter by ``regime_id`` and/or ``since`` (ISO-8601 UTC string).

        ``since`` is compared lexicographically against ``MetaTask.t0``;
        because ``t0`` is always emitted as a canonical ISO-8601 UTC
        string by ``build_meta_task`` (MLC-2), lexicographic compare
        equals temporal compare.
        """
        records = list(self._by_id.values())
        if regime_id is not None:
            records = [t for t in records if t.regime_id == regime_id]
        if since is not None:
            records = [t for t in records if t.t0 >= since]
        return sorted(records, key=lambda t: (t.regime_id, t.t0))

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[MetaTask]:
        for tid in self._insertion_order:
            yield self._by_id[tid]

    def iter_stable_keys(self) -> Iterable[tuple[str, str]]:
        return iter(self._by_stable.keys())

    # ------------------------------------------------------------------
    # Durable-store introspection (no mutation)
    # ------------------------------------------------------------------

    @property
    def durable_store(self) -> DurableTaskStore:
        return self._durable_store

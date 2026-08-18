"""ModelStore: atomic persistence for serialised model checkpoints."""

from __future__ import annotations

from typing import Any


class ModelStore:
    """Stores model checkpoints keyed by (job_id, candidate_id)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def _key(self, job_id: str, candidate_id: str) -> str:
        return f"{job_id}|{candidate_id}"

    def save(self, job_id: str, candidate_id: str, model: Any) -> None:
        """Persist a model checkpoint atomically (in-memory; subclass for disk)."""
        self._store[self._key(job_id, candidate_id)] = model

    def load(self, job_id: str, candidate_id: str) -> Any | None:
        """Return the checkpoint for (job_id, candidate_id), or None."""
        return self._store.get(self._key(job_id, candidate_id))


__all__ = ["ModelStore"]

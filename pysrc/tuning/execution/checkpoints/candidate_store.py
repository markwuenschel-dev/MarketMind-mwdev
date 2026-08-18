"""CandidateStore: atomic persistence for in-flight candidate trial results."""

from __future__ import annotations

from typing import Any


class CandidateStore:
    """Stores candidate trial results atomically during a search run."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def save(self, candidate_id: str, result: dict[str, Any]) -> None:
        """Persist a trial result atomically (in-memory; subclass for disk/DB)."""
        self._records[candidate_id] = dict(result)

    def load(self, candidate_id: str) -> dict[str, Any] | None:
        """Return the stored result for candidate_id, or None if absent."""
        return self._records.get(candidate_id)

    def all_ids(self) -> list[str]:
        """Return all stored candidate IDs."""
        return list(self._records.keys())


__all__ = ["CandidateStore"]

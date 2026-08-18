"""Deterministic task ID generation."""

from __future__ import annotations

import hashlib


def make_task_id(job_id: str, candidate_id: str, fold_index: int) -> str:
    """Return a stable task ID string for (job, candidate, fold)."""
    payload = f"{job_id}|{candidate_id}|{fold_index}".encode()
    digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return f"task:{digest}"


__all__ = ["make_task_id"]

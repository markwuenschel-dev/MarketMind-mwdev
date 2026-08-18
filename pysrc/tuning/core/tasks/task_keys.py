"""Deterministic task key generation from job/candidate/fold identifiers."""

from __future__ import annotations

import hashlib


def make_task_key(job_id: str, candidate_id: str, fold_index: int) -> str:
    """Return a short, stable hex key for (job, candidate, fold)."""
    payload = f"{job_id}|{candidate_id}|{fold_index}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def make_candidate_key(job_id: str, param_hash: str) -> str:
    """Return a stable key for a (job, param_hash) candidate."""
    payload = f"{job_id}|{param_hash}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


__all__ = ["make_task_key", "make_candidate_key"]

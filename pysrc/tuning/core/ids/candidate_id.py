"""Deterministic candidate ID generation from a spec hash and trial index."""

from __future__ import annotations

import hashlib


def make_candidate_id(spec_hash: str, trial_index: int) -> str:
    """Return a stable candidate ID for (spec_hash, trial_index)."""
    payload = f"{spec_hash}|{trial_index}".encode()
    digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return f"cand:{digest}"


__all__ = ["make_candidate_id"]

"""Execution integrity gate: verify artifact hashes and schema correctness."""

from __future__ import annotations

import re

_CAS_RE = re.compile(r"^cas\.v1:b3-256:[0-9a-f]{64}$")


def valid_cas_hash(h: str) -> bool:
    """Return True if h matches the canonical CAS hash format."""
    return bool(_CAS_RE.match(h))


def passes_integrity_gate(
    artifact_hash: str,
    expected_schema: str,
    actual_schema: str,
) -> tuple[bool, str]:
    """Return (passed, reason) for hash format and schema match checks."""
    if not valid_cas_hash(artifact_hash):
        return False, f"Invalid CAS hash: {artifact_hash!r}"
    if actual_schema != expected_schema:
        return False, f"Schema mismatch: expected {expected_schema!r}, got {actual_schema!r}"
    return True, "integrity_ok"


__all__ = ["valid_cas_hash", "passes_integrity_gate"]

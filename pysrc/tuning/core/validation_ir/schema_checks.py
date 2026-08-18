"""Schema-level IR checks: verify that spec hashes and version strings are well-formed."""

from __future__ import annotations

import re

from pysrc.tuning.core.ir.nodes import IRMetadata

__all__ = ["SchemaCheckError", "validate_spec_hash", "validate_metadata"]

_SPEC_HASH_RE = re.compile(r"^cas\.v1:b3-256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class SchemaCheckError(ValueError):
    """Raised when an IR field fails a schema format check."""


def validate_spec_hash(spec_hash: str) -> str:
    """Verify that *spec_hash* matches the canonical CAS format; return it if valid."""
    if not _SPEC_HASH_RE.match(spec_hash):
        raise SchemaCheckError(
            f"Invalid spec_hash format: {spec_hash!r}. Expected 'cas.v1:b3-256:<64 hex chars>'"
        )
    return spec_hash


def validate_metadata(meta: IRMetadata) -> IRMetadata:
    """Validate all fields of an IRMetadata object; return it if valid."""
    validate_spec_hash(meta.spec_hash)
    if meta.created_at_ns <= 0:
        raise SchemaCheckError("IRMetadata.created_at_ns must be positive")
    if meta.determinism_tier not in ("d0", "d1", "d2", "d3"):
        raise SchemaCheckError(f"Unknown determinism_tier: {meta.determinism_tier!r}")
    return meta

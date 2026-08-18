"""Stable BLAKE3-style spec hashing using blake2b as portable substitute."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_PREFIX = "cas.v1:b3-256:"


def hash_spec(data: dict[str, Any]) -> str:
    """Return a canonical CAS hash for a config/spec dict (JCS-ordered JSON)."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.blake2b(canonical.encode(), digest_size=32).hexdigest()
    return f"{_PREFIX}{digest}"


__all__ = ["hash_spec"]

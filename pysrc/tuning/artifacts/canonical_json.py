"""canonical_json: deterministic JSON serialisation for artifact hashing."""

from __future__ import annotations

import json
from typing import Any


def to_canonical_json(obj: Any) -> str:
    """Return a deterministic, compact JSON string (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def from_canonical_json(s: str) -> Any:
    """Parse a canonical JSON string back to a Python object."""
    return json.loads(s)


__all__ = ["to_canonical_json", "from_canonical_json"]

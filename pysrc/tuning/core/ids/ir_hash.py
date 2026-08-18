"""Stable hash for IR objects; serialises to JSON via dataclasses.asdict."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

_PREFIX = "cas.v1:b3-256:"


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def hash_ir(ir: Any) -> str:
    """Return a canonical CAS hash for any IR dataclass."""
    payload = json.dumps(_to_jsonable(ir), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode(), digest_size=32).hexdigest()
    return f"{_PREFIX}{digest}"


__all__ = ["hash_ir"]

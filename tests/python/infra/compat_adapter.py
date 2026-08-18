# tests/python/infra/compat_adapter.py
import contextlib
from typing import Any


def attach_caps(learning_engine: Any, caps: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        learning_engine.caps = caps or {}


def get_cap(learning_engine: Any, key: str, default=None):
    caps = getattr(learning_engine, "caps", {}) or {}
    return caps.get(key, default)

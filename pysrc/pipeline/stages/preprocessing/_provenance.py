from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

# Keys we shouldn't hash (volatile/execution-time hints may go here)
_IGNORED_HASH_KEYS: tuple[str, ...] = ("meta",)


def _freeze(obj: Any) -> Any:
    """
    Convert obj into a JSON-stable structure:
    - dict -> sorted tuples of (key, frozen(value))
    - list/tuple/set -> list of frozen elements (sets sorted)
    - bytes -> base64 str; other scalars left as-is
    - objects with __dict__ -> freeze __dict__
    """
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, bytes):
        import base64

        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Mapping):
        items = []
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            if k in _IGNORED_HASH_KEYS:
                continue
            items.append((str(k), _freeze(obj[k])))
        return tuple(items)
    if isinstance(obj, set):
        return [_freeze(x) for x in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, (list, tuple)):
        return [_freeze(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return _freeze(vars(obj))
    # Fallback to string
    return str(obj)


def _stable_hash(obj: Any) -> str:
    frozen = _freeze(obj)
    payload = json.dumps(frozen, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_meta(step_name: str, step_version: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Merge user-provided meta/export with provenance:
      meta = {
        ...<user meta/export>...,
        "provenance": {"step": step_name, "version": step_version, "cfg_hash": <sha256>}
      }
    """
    user_meta = dict(cfg.get("meta") or {})
    # Legacy 'export' passthrough
    if "export" in cfg and "export" not in user_meta:
        user_meta["export"] = cfg["export"]

    meta = {
        **user_meta,
        "provenance": {
            "step": step_name,
            "version": step_version,
            "cfg_hash": _stable_hash(cfg),
        },
    }
    return meta

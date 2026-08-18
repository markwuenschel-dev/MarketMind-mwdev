"""Stable hash for plan objects."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.core.ids.ir_hash import hash_ir


def hash_plan(plan: Any) -> str:
    """Return a canonical CAS hash for any plan dataclass."""
    return hash_ir(plan)


__all__ = ["hash_plan"]

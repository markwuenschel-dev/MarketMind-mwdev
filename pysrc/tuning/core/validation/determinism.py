"""Determinism enforcement: fingerprinting split boundaries for D0 replay checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint_splits(splits: list[tuple[Any, Any]]) -> str:
    """Return a stable hex digest of split boundaries for D0 replay checks."""
    payload = json.dumps(
        [
            {
                "train_start": str(tr[0]) if hasattr(tr, "__iter__") else str(tr),
                "train_end": str(tr[-1]) if hasattr(tr, "__iter__") else str(tr),
                "test_start": str(te[0]) if hasattr(te, "__iter__") else str(te),
                "test_end": str(te[-1]) if hasattr(te, "__iter__") else str(te),
            }
            for tr, te in splits
        ],
        sort_keys=True,
    ).encode()
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


__all__ = ["fingerprint_splits"]

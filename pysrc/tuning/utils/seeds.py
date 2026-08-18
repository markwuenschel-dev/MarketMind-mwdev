"""Deterministic seed derivation for the tuning sub-system.

Produces reproducible integer seeds from a base seed and a namespace string via HMAC-SHA256.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from typing import Final

__all__ = ["derive_seed", "SEED_DIGEST_BYTES"]

SEED_DIGEST_BYTES: Final[int] = 8


def derive_seed(base: int, namespace: str) -> int:
    """Return a deterministic int seed from base + namespace via HMAC-SHA256."""
    key = struct.pack(">q", base)
    msg = namespace.encode()
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return int(struct.unpack(">Q", digest[:SEED_DIGEST_BYTES])[0])

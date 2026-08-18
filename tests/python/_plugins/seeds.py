"""
Deterministic seed policy and ADR-007 v1.1 D-tier markers.

Per-test seeds are derived from master_seed + test node ID so tests stay
reproducible and independent. Use @pytest.mark.determinism("d0"|"d1"|"d2"|"d3")
to request a tier.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import random
import struct

import pytest


def _derive_seed(master_seed: int, test_id: str) -> int:
    """Derive a 31-bit seed from master seed and test node ID (HMAC-SHA256)."""
    key = struct.pack(">I", master_seed & 0xFFFFFFFF)
    digest = hmac.new(key, test_id.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


@pytest.fixture(scope="session")
def master_seed() -> int:
    """Session-wide master seed (env PYTEST_MASTER_SEED or 42)."""
    raw = os.environ.get("PYTEST_MASTER_SEED", "42")
    try:
        return int(raw) & 0x7FFFFFFF
    except ValueError:
        return 42


@pytest.fixture(autouse=True)
def deterministic_seed(request: pytest.FixtureRequest, master_seed: int) -> int:
    """
    Per-test deterministic seed. Sets random, numpy, and (if present) torch.
    Reads @pytest.mark.determinism("d0"|"d1"|"d2"|"d3") for tier; default is d2.
    """
    seed = _derive_seed(master_seed, request.node.nodeid)
    tier = "d2"
    for marker in getattr(request.node, "iter_markers", lambda name: ())("determinism"):
        if marker.args:
            tier = marker.args[0]
        break

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if tier in ("d0", "d2"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if tier == "d0" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass

    return seed

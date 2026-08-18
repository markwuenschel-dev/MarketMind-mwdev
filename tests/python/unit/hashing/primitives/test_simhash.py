from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/simhash/manifest.json")


def test_simhash_manifest_declares_dim_aware_seed_anchor() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "simhash"
    assert manifest["cases"][0]["id"] == "SH-001"


def test_simhash_zero_vector_contract() -> None:
    import numpy as np

    from pysrc.ops.hashing.primitives.simhash_impl import SimHash128

    hasher = SimHash128(master_seed=b"\x00" * 32, dim=4)
    raw_int, raw_bytes = hasher.hash_vector(np.zeros(4, dtype=float))
    assert raw_int == 0
    assert raw_bytes == b"\x00" * 16

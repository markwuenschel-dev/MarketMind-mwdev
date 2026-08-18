from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/minhash/manifest.json")


def test_minhash_manifest_declares_seed_schedule_anchor() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "minhash"
    assert manifest["cases"][0]["id"] == "MH-001"


def test_minhash_signature_shape_contract() -> None:
    from pysrc.ops.hashing.primitives.minhash_impl import MinHash128

    hasher = MinHash128(
        master_seed=b"\x00" * 32, master_key_id="00000000-0000-0000-0000-000000000001"
    )
    signature = hasher.hash_set({"AAPL", "MSFT"})
    assert signature.shape == (128,)

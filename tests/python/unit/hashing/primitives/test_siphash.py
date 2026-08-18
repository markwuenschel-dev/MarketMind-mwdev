from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/sip24/manifest.json")


def test_siphash_manifest_declares_reference_vector_anchor() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "sip24"
    assert manifest["cases"][0]["id"] == "SIP-001"


def test_siphash_keyed_hash_ref_includes_key_id() -> None:
    from pysrc.ops.hashing.primitives.siphash_impl import SipHash24Hasher, SipHashKey

    hasher = SipHash24Hasher(SipHashKey.generate())
    ref = hasher.hash_map_key(b"symbol", namespace="order_book.v1")
    assert ref.algo == "sip24"
    assert ref.key_id is not None

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/xxh3/manifest.json")


def test_xxh3_manifest_records_ci_gap() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "xxh3"
    assert "ci_harness_gap" in manifest


def test_xxh3_local_persistent_hash_ref_shape() -> None:
    from pysrc.ops.hashing.primitives.xxh3_impl import XXH3

    ref = XXH3.hash_local_persistent_key(b"cache-key", namespace="feature.v1")
    assert ref.domain == "cache.v1"
    assert ref.algo == "xxh3-128"

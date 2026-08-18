from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/blake3/manifest.json")


def test_blake3_manifest_declares_cross_language_requirement() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "blake3"
    assert manifest["languages_required"] == ["python-3.12", "cpp-20", "java-21"]
    assert manifest["cases"]


def test_blake3_artifact_hash_uses_hashref_envelope() -> None:
    from pysrc.ops.hashing.primitives.blake3_impl import BLAKE3

    ref = BLAKE3.hash_artifact_id(b"artifact")
    assert ref.domain == "cas.v1"
    assert ref.algo == "b3-256"

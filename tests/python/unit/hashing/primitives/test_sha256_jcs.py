from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/jcs_sha256/manifest.json")


def test_jcs_sha256_manifest_declares_dual_domain_anchor_cases() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "jcs_sha256"
    assert {case["id"] for case in manifest["cases"]} == {"AT-001", "AT-002"}


def test_sha256_jcs_hashes_canonical_json_bytes() -> None:
    from pysrc.ops.hashing.primitives.sha256_jcs import SHA256_JCS

    ref = SHA256_JCS.hash_gate_attestation({"b": 2, "a": 1})
    assert ref.domain == "attest.v1"
    assert ref.algo == "jcs-sha256"

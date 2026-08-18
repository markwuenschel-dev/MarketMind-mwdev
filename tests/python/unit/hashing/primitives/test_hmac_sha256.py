from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/hmac_sha256/manifest.json")


def test_hmac_sha256_manifest_tracks_seed_derivation_anchor() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "hmac_sha256"
    assert manifest["determinism_tier"] == "d2"


def test_hmac_sha256_run_seed_is_byte_stable() -> None:
    from pysrc.ops.hashing.primitives.hmac_sha256_impl import HmacSha256Deriver

    deriver = HmacSha256Deriver(
        master_key=b"\x00" * 32, master_key_id="00000000-0000-0000-0000-000000000001"
    )
    assert len(deriver.derive_run_seed("run-001")) == 32

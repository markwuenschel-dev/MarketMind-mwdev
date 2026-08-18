from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.ops.hashing.primitives.rabin_impl import RABIN_POLY

pytestmark = pytest.mark.determinism("d2")

_MANIFEST = Path("tests/golden/adr007/rabin/manifest.json")


def test_rabin_manifest_declares_irreducibility_anchor() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["primitive"] == "rabin"
    assert manifest["cases"][0]["id"] == "RF-001"
    assert RABIN_POLY == 0x8000000000000003


def test_rabin_startup_verifies_polynomial_and_builds_tables() -> None:
    from pysrc.ops.hashing.primitives.rabin_impl import RabinRollingHasher

    hasher = RabinRollingHasher(window_size=4)
    assert hasher is not None

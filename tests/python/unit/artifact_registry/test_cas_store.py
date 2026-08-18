from __future__ import annotations

from pathlib import Path

import pytest

from marketmind_gate.hashing.canonical import canonicalize
from pysrc.artifact_registry import LocalCAS

pytestmark = pytest.mark.determinism("d3")


def test_put_json_keeps_dual_domain_hashes_over_same_jcs_bytes(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = {"b": 2, "a": 1}

    hashes = cas.put_json(payload)
    canonical_bytes = canonicalize(payload)

    assert hashes.cas.domain == "cas.v1"
    assert hashes.cas.algo == "b3-256"
    assert hashes.attest is not None
    assert hashes.attest.domain == "attest.v1"
    assert hashes.attest.algo == "jcs-sha256"
    assert hashes.cas.hex_digest != hashes.attest.hex_digest
    assert hashes.canonical_bytes == canonical_bytes
    assert cas.get_bytes(hashes.cas) == canonical_bytes


def test_put_json_is_idempotent_for_same_payload(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")

    h1 = cas.put_json({"x": 1, "y": [1, 2, 3]})
    h2 = cas.put_json({"x": 1, "y": [1, 2, 3]})

    assert str(h1.cas) == str(h2.cas)
    assert h1.attest is not None
    assert h2.attest is not None
    assert str(h1.attest) == str(h2.attest)


def test_put_json_changes_both_domains_when_canonical_bytes_change(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")

    h1 = cas.put_json({"a": 1})
    h2 = cas.put_json({"a": 2})

    assert str(h1.cas) != str(h2.cas)
    assert h1.attest is not None
    assert h2.attest is not None
    assert str(h1.attest) != str(h2.attest)


def test_put_bytes_stores_only_cas_identity(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")

    b1 = cas.put_bytes(b"hello", media_type="text/plain")
    b2 = cas.put_bytes(b"hello", media_type="text/plain")

    assert str(b1.cas) == str(b2.cas)
    assert b1.attest is None
    assert b2.attest is None
    assert b1.canonical_bytes == b""


def test_attest_alias_index_resolves_back_to_cas(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")

    hashes = cas.put_json({"k": "v"})
    assert hashes.attest is not None

    resolved = cas.resolve_attest(hashes.attest)
    assert resolved is not None
    assert str(resolved) == str(hashes.cas)


def test_verify_detects_blob_corruption(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    hashes = cas.put_bytes(b"hello", media_type="text/plain")

    assert cas.verify(hashes.cas) is True

    blob_path = cas._path_for_cas(hashes.cas)  # type: ignore[attr-defined]
    blob_path.write_bytes(b"hacked")

    assert cas.verify(hashes.cas) is False

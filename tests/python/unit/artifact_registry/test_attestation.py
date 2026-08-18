from __future__ import annotations

from dataclasses import dataclass

import pytest

from pysrc.artifact_registry.attestation import (
    ArtifactAttestor,
    BundleManifestWriter,
    build_artifact_entry,
    bundle_entries_from_sequence,
)

pytestmark = pytest.mark.determinism("d3")


@dataclass(frozen=True, slots=True)
class _FakeHashRef:
    domain: str
    algo: str
    digest: str
    purpose: str = "TEST"

    def to_id_string(self) -> str:
        return f"{self.domain}:{self.algo}:{self.digest}"


@dataclass(frozen=True, slots=True)
class _FakeResult:
    cas: _FakeHashRef
    attest: _FakeHashRef
    canonical_bytes: bytes


class _FakeCAS:
    def put_json(self, payload: dict[str, object]) -> _FakeResult:
        return _FakeResult(
            cas=_FakeHashRef("cas.v1", "b3-256", "a" * 64, "CAS_ARTIFACT_ID"),
            attest=_FakeHashRef("attest.v1", "jcs-sha256", "b" * 64, "GATE_ATTESTATION"),
            canonical_bytes=b'{"a":1}',
        )

    def put_bytes(self, payload: bytes, *, media_type: str) -> _FakeResult:
        return _FakeResult(
            cas=_FakeHashRef("cas.v1", "b3-256", "c" * 64, "CAS_ARTIFACT_ID"),
            attest=_FakeHashRef("attest.v1", "jcs-sha256", "d" * 64, "GATE_ATTESTATION"),
            canonical_bytes=payload,
        )


def test_attestor_validates_dual_domain_pair_and_formats_gate_hash() -> None:
    attestor = ArtifactAttestor(cas=_FakeCAS())

    entry = attestor.attest_json({"a": 1}, bundle_path="bundle/plan.json")

    assert entry.cas.domain == "cas.v1"
    assert entry.attest.domain == "attest.v1"
    assert attestor.to_gate_content_hash(entry.attest) == "sha256:" + ("b" * 64)


def test_bundle_manifest_writer_emits_dual_domain_policy(tmp_path) -> None:
    writer = BundleManifestWriter(output_root=tmp_path)
    entry = build_artifact_entry(
        path="bundle/plan.json",
        cas=_FakeHashRef("cas.v1", "b3-256", "a" * 64),
        attest=_FakeHashRef("attest.v1", "jcs-sha256", "b" * 64),
        media_type="application/json",
        size=16,
    )

    manifest_path = writer.write(artifacts={"plan": entry})

    text = manifest_path.read_text(encoding="utf-8")
    assert '"cas": "cas.v1:b3-256:' in text
    assert '"attest": "attest.v1:jcs-sha256:' in text


def test_bundle_entries_reject_duplicate_roles() -> None:
    entry = build_artifact_entry(
        path="bundle/plan.json",
        cas=_FakeHashRef("cas.v1", "b3-256", "a" * 64),
        attest=_FakeHashRef("attest.v1", "jcs-sha256", "b" * 64),
        media_type="application/json",
        size=16,
    )

    with pytest.raises(ValueError):
        bundle_entries_from_sequence([("plan", entry), ("plan", entry)])

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from pysrc.ops.hashing.contract import HashContractViolation, HashPurpose, PersistenceTier
from pysrc.ops.hashing.envelope import HashRef
from pysrc.ops.hashing.equality import (
    EqualityEvidence,
    assert_ref_compatibility,
    requires_equality_fallback,
    validate_evidence_policy,
    verify_aux_check,
    verify_cache_hit,
    verify_payload_equality,
)

pytestmark = pytest.mark.determinism("d3")


def test_equality_evidence_is_frozen_slots_dataclass() -> None:
    assert is_dataclass(EqualityEvidence)
    evidence = EqualityEvidence(payload_bytes=b"payload", aux_check=b"aux", payload_length=7)
    assert [f.name for f in fields(EqualityEvidence)] == [
        "payload_bytes",
        "aux_check",
        "payload_length",
    ]
    assert not hasattr(evidence, "__dict__")
    with pytest.raises(FrozenInstanceError):
        evidence.payload_length = 99  # type: ignore[misc]


def test_equality_evidence_defaults_to_none() -> None:
    evidence = EqualityEvidence()
    assert evidence.payload_bytes is None
    assert evidence.aux_check is None
    assert evidence.payload_length is None


def test_verify_cache_hit_is_keyword_only() -> None:
    sig = inspect.signature(verify_cache_hit)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())


def _cache_ref(
    *,
    digest: str = "a" * 32,
    key_id: str | None = None,
    purpose: str = HashPurpose.LOCAL_PERSISTENT_CACHE_KEY.name,
) -> HashRef:
    return HashRef(
        domain="cache.v1",
        algo="xxh3-128",
        digest=digest,
        purpose=purpose,
        key_id=key_id,
    )


def _sip_ref(*, digest: str = "a" * 16, key_id: str) -> HashRef:
    return HashRef(
        domain="cache.v1",
        algo="sip24",
        digest=digest,
        purpose=HashPurpose.HASHDOS_TABLE_KEY.name,
        key_id=key_id,
    )


def test_equality_fallback_law_for_non_cryptographic_cache_surfaces() -> None:
    assert requires_equality_fallback(HashPurpose.LOCAL_PERSISTENT_CACHE_KEY) is True
    assert requires_equality_fallback(HashPurpose.CAS_ARTIFACT_ID) is False


def test_keyed_mismatch_is_reject_not_false() -> None:
    with pytest.raises(HashContractViolation):
        assert_ref_compatibility(
            _sip_ref(key_id="00000000-0000-0000-0000-000000000001"),
            _sip_ref(key_id="00000000-0000-0000-0000-000000000002"),
        )


def test_aux_check_or_payload_policy_is_enforced() -> None:
    with pytest.raises(HashContractViolation):
        validate_evidence_policy(
            HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
            PersistenceTier.LOCAL_PERSISTENT,
            EqualityEvidence(),
        )


def test_verify_payload_equality_compares_exact_bytes() -> None:
    assert verify_payload_equality(b"alpha", b"alpha") is True
    assert verify_payload_equality(b"alpha", b"beta") is False


def test_verify_aux_check_compares_secondary_witnesses() -> None:
    assert verify_aux_check(b"same", b"same") is True
    assert verify_aux_check(b"same", b"diff") is False


def test_verify_cache_hit_prefers_payload_or_aux_evidence() -> None:
    assert verify_cache_hit(
        purpose=HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
        stored_ref=_cache_ref(),
        observed_ref=_cache_ref(),
        stored_evidence=EqualityEvidence(payload_bytes=b"alpha", payload_length=5),
        observed_evidence=EqualityEvidence(payload_bytes=b"alpha", payload_length=5),
    )

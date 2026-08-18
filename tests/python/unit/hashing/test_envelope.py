from __future__ import annotations

import pytest

from pysrc.ops.hashing.contract import HashContractViolation, HashPurpose
from pysrc.ops.hashing.envelope import (
    HashRef,
    make_attest_ref,
    make_cas_ref,
    make_hmac_ref,
    make_siphash_ref,
)

pytestmark = pytest.mark.determinism("d3")


def test_hashref_validates_known_domain_algo_pairs_on_construction() -> None:
    ref = HashRef(
        domain="cas.v1",
        algo="b3-256",
        digest="a" * 64,
        purpose=HashPurpose.CAS_ARTIFACT_ID.name,
    )
    assert ref.to_id_string() == "cas.v1:b3-256:" + ("a" * 64)


def test_hashref_construction_via_cas_factory() -> None:
    ref = make_cas_ref("a" * 64)
    assert ref.domain == "cas.v1"
    assert ref.algo == "b3-256"


def test_hashref_construction_via_attest_factory() -> None:
    ref = make_attest_ref("b" * 64)
    assert ref.domain == "attest.v1"
    assert ref.algo == "jcs-sha256"


def test_hashref_rejects_missing_key_id_for_keyed_algorithms() -> None:
    with pytest.raises(HashContractViolation):
        HashRef(
            domain="cache.v1",
            algo="sip24",
            digest="a" * 16,
            purpose=HashPurpose.HASHDOS_TABLE_KEY.name,
            key_id=None,
        )


def test_version_and_key_id_requirements_for_keyed_factories() -> None:
    sip = make_siphash_ref(
        "a" * 16,
        purpose=HashPurpose.HASHDOS_TABLE_KEY,
        key_id="00000000-0000-0000-0000-000000000001",
    )
    hmac = make_hmac_ref(
        "b" * 64,
        key_id="00000000-0000-0000-0000-000000000002",
    )
    assert sip.key_id is not None
    assert hmac.key_id is not None


def test_key_id_mismatch_is_rejected_on_equality_check() -> None:
    a = HashRef(
        domain="cache.v1",
        algo="sip24",
        digest="a" * 16,
        purpose=HashPurpose.HASHDOS_TABLE_KEY.name,
        key_id="00000000-0000-0000-0000-000000000001",
    )
    b = HashRef(
        domain="cache.v1",
        algo="sip24",
        digest="a" * 16,
        purpose=HashPurpose.HASHDOS_TABLE_KEY.name,
        key_id="00000000-0000-0000-0000-000000000002",
    )
    with pytest.raises(HashContractViolation):
        a.equality_check(b)

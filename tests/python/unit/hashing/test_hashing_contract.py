"""
tests/unit/test_hashing_contract.py
════════════════════════════════════
Unit tests for the MarketMind Hashing Contract (ADR-007 v1.1).

COVERAGE TARGETS
  This file must achieve 100% branch coverage on:
    - py/ops/hashing/contract.py
    - py/ops/hashing/envelope.py
    - py/ops/hashing/canonicalizer.py
    - py/ops/hashing/ahm.py

  Each primitive test file covers its respective primitive.

CI WIRING STATUS (ADR-007 v1.1 §7.2):
  Incomplete suite → WARN not SKIP.
  A primitive without a full three-language golden vector harness
  is capped at D2 determinism tier until certified.

HYPOTHESIS PROPERTY TESTS
  Property tests are in tests/property/test_hashing_properties.pysrc.
  These use Hypothesis strategies to verify:
    - Domain separation: ∀ (ns_a, ns_b, data) where ns_a != ns_b:
        hash(ns_a, data) != hash(ns_b, data)
    - Avalanche: Hamming-1 input changes → expected digest bit-flip rate ≈ 50%
    - Equality fallback law: Non-cryptographic collisions detected before returning
"""

from __future__ import annotations

import struct

import pytest

from pysrc.ops.hashing.canonicalizer import CANON

# These imports will fail until the primitives are implemented.
# Mark all tests as xfail until NotImplementedError is resolved.
from pysrc.ops.hashing.contract import (
    AlgoId,
    CanonicalValueRejected,
    DTier,
    HashContractViolation,
    HashPurpose,
    PersistenceTier,
)
from pysrc.ops.hashing.envelope import HashRef, make_cas_ref

# ── Contract tests ────────────────────────────────────────────────────────────


class TestHashPurposeContract:
    """Verify that every HashPurpose has correct algorithm and tier bindings."""

    def test_cas_purposes_require_blake3_d3(self) -> None:
        """CAS_ARTIFACT_ID, MERKLE_NODE_HASH, AUDIT_LOG_DIGEST → BLAKE3 D3."""
        for purpose in [
            HashPurpose.CAS_ARTIFACT_ID,
            HashPurpose.MERKLE_NODE_HASH,
            HashPurpose.AUDIT_LOG_DIGEST,
        ]:
            assert purpose.meta.algo_id == AlgoId.BLAKE3_256
            assert purpose.meta.d_tier == DTier.BITWISE
            assert purpose.meta.persistence_tier == PersistenceTier.IMMUTABLE_CAS

    def test_gate_attestation_requires_sha256_jcs(self) -> None:
        purpose = HashPurpose.GATE_ATTESTATION
        assert purpose.meta.algo_id == AlgoId.SHA256_JCS
        assert purpose.meta.d_tier == DTier.BITWISE
        assert purpose.meta.persistence_tier == PersistenceTier.LOCAL_PERSISTENT

    def test_distributed_cache_key_requires_blake3_not_xxh3(self) -> None:
        """Regression: DISTRIBUTED_CACHE_KEY must use BLAKE3, never XXH3-128."""
        purpose = HashPurpose.DISTRIBUTED_CACHE_KEY
        assert purpose.meta.algo_id == AlgoId.BLAKE3_256, (
            "DISTRIBUTED_CACHE_KEY must use BLAKE3-256. "
            "XXH3-128 is non-cryptographic and cannot protect cross-process surfaces."
        )

    def test_local_persistent_uses_xxh3_128(self) -> None:
        for purpose in [
            HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
            HashPurpose.DATAFRAME_FINGERPRINT_FAST,
        ]:
            assert purpose.meta.algo_id == AlgoId.XXH3_128
            assert purpose.meta.persistence_tier == PersistenceTier.LOCAL_PERSISTENT

    def test_ephemeral_map_key_uses_xxh3_64(self) -> None:
        purpose = HashPurpose.EPHEMERAL_MAP_KEY
        assert purpose.meta.algo_id == AlgoId.XXH3_64
        assert purpose.meta.persistence_tier == PersistenceTier.EPHEMERAL
        assert purpose.meta.d_tier == DTier.SEMANTIC

    def test_no_purpose_is_d0(self) -> None:
        """No HashPurpose may use DTier.NONE (D0). This is a global ban."""
        for purpose in HashPurpose:
            assert purpose.meta.d_tier != DTier.NONE, (
                f"HashPurpose.{purpose.name} has D-Tier NONE — this is globally banned."
            )

    def test_ahm_forbidden_exactly_non_ephemeral(self) -> None:
        """AHM is forbidden for all non-EPHEMERAL purposes."""
        for purpose in HashPurpose:
            if purpose.meta.persistence_tier != PersistenceTier.EPHEMERAL:
                assert purpose.is_ahm_forbidden(), (
                    f"{purpose.name} is non-ephemeral but is_ahm_forbidden() returned False."
                )
            else:
                assert not purpose.is_ahm_forbidden(), (
                    f"{purpose.name} is ephemeral but is_ahm_forbidden() returned True."
                )


# ── Canonicalizer tests ───────────────────────────────────────────────────────


class TestCanonicalizer:
    """Unit tests for Canonicalizer — all operations are pre-hash."""

    # ── String encoding

    def test_encode_string_produces_utf8_no_bom(self) -> None:
        result = CANON.encode_string("hello")
        assert result == b"hello"
        assert not result.startswith(b"\xef\xbb\xbf")

    def test_encode_string_rejects_bom_source(self) -> None:
        bom_string = "\ufeffhello"
        with pytest.raises(HashContractViolation):
            CANON.encode_string(bom_string)

    def test_encode_string_handles_unicode(self) -> None:
        result = CANON.encode_string("€ → ∞")
        assert result == "€ → ∞".encode()

    # ── Integer encoding

    def test_encode_u64be_zero(self) -> None:
        assert CANON.encode_u64be(0) == b"\x00" * 8

    def test_encode_u64be_max(self) -> None:
        assert CANON.encode_u64be(2**64 - 1) == b"\xff" * 8

    def test_encode_u64be_overflow(self) -> None:
        with pytest.raises(OverflowError):
            CANON.encode_u64be(2**64)

    def test_encode_u64be_negative_raises(self) -> None:
        with pytest.raises((OverflowError, struct.error)):
            CANON.encode_u64be(-1)

    # ── Float normalization

    def test_float_nan_becomes_quiet_nan(self) -> None:
        result = CANON.normalize_float64(float("nan"))
        assert result == b"\x7f\xf8\x00\x00\x00\x00\x00\x00"

    def test_float_neg_zero_becomes_pos_zero(self) -> None:
        result = CANON.normalize_float64(-0.0)
        assert result == b"\x00" * 8

    def test_float_inf_rejected_at_persistent_tier(self) -> None:
        for v in [float("inf"), float("-inf")]:
            with pytest.raises(CanonicalValueRejected):
                CANON.normalize_float64(v, tier=PersistenceTier.IMMUTABLE_CAS)
            with pytest.raises(CanonicalValueRejected):
                CANON.normalize_float64(v, tier=PersistenceTier.DISTRIBUTED)
            with pytest.raises(CanonicalValueRejected):
                CANON.normalize_float64(v, tier=PersistenceTier.LOCAL_PERSISTENT)

    def test_float_inf_passes_at_ephemeral_tier(self) -> None:
        """±Inf is allowed at ephemeral tier — caller's responsibility not to use at D3."""
        result = CANON.normalize_float64(float("inf"), tier=PersistenceTier.EPHEMERAL)
        assert len(result) == 8

    def test_float_subnormal_preserved(self) -> None:
        subnormal = 5e-324  # smallest positive subnormal
        result = CANON.normalize_float64(subnormal)
        assert result == struct.pack(">d", subnormal)

    # ── Composite preimage

    def test_composite_preimage_domain_separation(self) -> None:
        """H('mm/ns', 'AB', 'C') must differ from H('mm/ns', 'A', 'BC')."""
        a = CANON.build_composite_preimage("mm/ns", b"AB", b"C")
        b = CANON.build_composite_preimage("mm/ns", b"A", b"BC")
        assert a != b, "Domain separation failure: length-extension attack possible."

    def test_composite_preimage_empty_namespace_raises(self) -> None:
        with pytest.raises(HashContractViolation):
            CANON.build_composite_preimage("", b"data")

    def test_composite_preimage_structure(self) -> None:
        """Verify exact byte structure: ns || u64be(len(a)) || a || u64be(len(b)) || b."""
        ns = b"mm/test"
        a = b"hello"
        b_ = b"world"
        result = CANON.build_composite_preimage("mm/test", a, b_)
        expected = ns + struct.pack(">Q", len(a)) + a + struct.pack(">Q", len(b_)) + b_
        assert result == expected


# ── HashRef envelope tests ────────────────────────────────────────────────────


class TestHashRefEnvelope:
    """Unit tests for HashRef construction and validation."""

    def test_valid_cas_ref_construction(self) -> None:
        digest = "a" * 64
        ref = make_cas_ref(digest)
        assert ref.domain == "cas.v1"
        assert ref.algo == "b3-256"
        assert ref.digest == digest

    def test_invalid_digest_length_raises(self) -> None:
        with pytest.raises(HashContractViolation):
            make_cas_ref("a" * 63)  # 63 not 64

    def test_uppercase_hex_rejected(self) -> None:
        with pytest.raises(HashContractViolation):
            make_cas_ref("A" * 64)

    def test_unknown_domain_raises(self) -> None:
        with pytest.raises(HashContractViolation):
            HashRef(domain="unknown.v1", algo="b3-256", digest="a" * 64, purpose="CAS_ARTIFACT_ID")

    def test_keyed_algo_without_key_id_raises(self) -> None:
        """SipHash and HMAC envelopes must include key_id."""
        with pytest.raises(HashContractViolation):
            HashRef(
                domain="cache.v1",
                algo="sip24",
                digest="a" * 16,
                purpose="HASHDOS_TABLE_KEY",
                key_id=None,
            )  # Must fail — key_id is mandatory for keyed algos

    def test_to_id_string_format(self) -> None:
        ref = make_cas_ref("a" * 64)
        assert ref.to_id_string() == "cas.v1:b3-256:" + "a" * 64

    def test_key_id_mismatch_raises_on_equality_check(self) -> None:
        """key_id mismatch must raise HashContractViolation, not return False."""
        ref_a = HashRef(
            domain="cache.v1",
            algo="sip24",
            digest="a" * 16,
            purpose="HASHDOS_TABLE_KEY",
            key_id="00000000-0000-0000-0000-000000000001",
        )
        ref_b = HashRef(
            domain="cache.v1",
            algo="sip24",
            digest="a" * 16,
            purpose="HASHDOS_TABLE_KEY",
            key_id="00000000-0000-0000-0000-000000000002",
        )
        with pytest.raises(HashContractViolation):
            ref_a.equality_check(ref_b)


# ── AHM boundary tests ────────────────────────────────────────────────────────


class TestAHMBoundary:
    """Verify the AHM permanently refuses all non-ephemeral purposes."""

    @pytest.fixture
    def ahm(self):
        from pysrc.ops.hashing.ahm import AHM
        from pysrc.ops.hashing.primitives.siphash_impl import SipHashKey

        return AHM(siphash_key=SipHashKey.generate())

    @pytest.mark.parametrize(
        "forbidden_purpose",
        [
            HashPurpose.CAS_ARTIFACT_ID,
            HashPurpose.MERKLE_NODE_HASH,
            HashPurpose.AUDIT_LOG_DIGEST,
            HashPurpose.GATE_ATTESTATION,
            HashPurpose.DISTRIBUTED_CACHE_KEY,
            HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
            HashPurpose.DATAFRAME_FINGERPRINT_FAST,
            HashPurpose.SEED_DERIVATION,
            HashPurpose.LSH_VECTOR_SIMHASH,
            HashPurpose.LSH_SET_MINHASH,
            HashPurpose.ROLLING_WINDOW_FINGERPRINT,
            HashPurpose.CHUNK_BOUNDARY_DETECTION,
        ],
    )
    def test_ahm_rejects_forbidden_purposes(self, ahm, forbidden_purpose: HashPurpose) -> None:
        with pytest.raises(HashContractViolation, match="PERMANENTLY FORBIDDEN"):
            ahm.validate_purpose_scope(forbidden_purpose)

    def test_ahm_permits_ephemeral_map_key(self, ahm) -> None:
        ahm.validate_purpose_scope(HashPurpose.EPHEMERAL_MAP_KEY)

    def test_ahm_permits_hashdos_table_key(self, ahm) -> None:
        ahm.validate_purpose_scope(HashPurpose.HASHDOS_TABLE_KEY)


# ── Dual-domain attestation invariant ─────────────────────────────────────────


class TestDualDomainInvariant:
    """AT-003: cas and attest hash the SAME canonical bytes with different algorithms."""

    def test_same_jcs_bytes_different_algos(self) -> None:
        import hashlib

        from pysrc.ops.hashing.primitives.blake3_impl import BLAKE3
        from pysrc.ops.hashing.primitives.sha256_jcs import SHA256_JCS

        artifact = {"strategy": "momentum", "version": "1.0", "threshold": 0.05}
        jcs_bytes = CANON.canonicalize_json(artifact)

        cas_ref = BLAKE3.hash_artifact_id(jcs_bytes)
        attest_ref = SHA256_JCS.hash_gate_attestation_from_jcs_bytes(jcs_bytes)

        assert cas_ref.algo == "b3-256"
        assert attest_ref.algo == "jcs-sha256"
        assert cas_ref.digest != attest_ref.digest  # Different algorithms → different digests
        assert cas_ref.domain == "cas.v1"
        assert attest_ref.domain == "attest.v1"

        # Verify gate wire format
        gate_str = SHA256_JCS.to_gate_content_hash(attest_ref)
        assert gate_str == f"sha256:{attest_ref.digest}"
        assert gate_str == f"sha256:{hashlib.sha256(jcs_bytes).hexdigest()}"

"""
tests/property/test_hashing_properties.py
══════════════════════════════════════════
Hypothesis property tests for hashing invariants (ADR-007 v1.1).

These tests verify mathematical properties that MUST hold for ALL inputs,
not just a fixed set of golden vectors.

DETERMINISM TIER COVERAGE REQUIREMENT
  Any property that crosses the D3 boundary (cross-machine reproducibility)
  MUST ALSO have a golden vector counterpart in tests/golden/adr007/.
  Hypothesis finds edge cases; golden vectors provide cross-language anchors.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pysrc.ops.hashing.canonicalizer import CANON
from pysrc.ops.hashing.contract import (
    CanonicalValueRejected,
    PersistenceTier,
)

VALID_NAMESPACE_TEXT = st.text(
    min_size=1,
    max_size=64,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="\ufeff",
    ),
)


# ── Canonicalizer properties ──────────────────────────────────────────────────


class TestCanonicalDomainSeparation:
    """For all namespaces and all field byte combinations, length-prefix separation holds."""

    @given(
        ns=VALID_NAMESPACE_TEXT,
        field_a=st.binary(min_size=0, max_size=128),
        field_b=st.binary(min_size=0, max_size=128),
    )
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_domain_separation_prevents_concatenation_collision(
        self,
        ns: str,
        field_a: bytes,
        field_b: bytes,
    ) -> None:
        """H(ns, a+b) != H(ns, a, b) for all non-trivial cases.

        Specifically: build_composite_preimage(ns, a+b) must differ from
        build_composite_preimage(ns, a, b) unless a=b'' or b=b''.
        """
        assume(len(field_a) > 0 and len(field_b) > 0)
        combined = CANON.build_composite_preimage(ns, field_a + field_b)
        separated = CANON.build_composite_preimage(ns, field_a, field_b)
        assert combined != separated, (
            f"Domain separation failure: ns={ns!r}, a={field_a!r}, b={field_b!r}\n"
            "The preimage of (a+b) as one field must differ from (a, b) as two fields."
        )

    @given(
        ns_a=VALID_NAMESPACE_TEXT.filter(lambda s: len(s) <= 32),
        ns_b=VALID_NAMESPACE_TEXT.filter(lambda s: len(s) <= 32),
        data=st.binary(min_size=1, max_size=256),
    )
    @settings(
        max_examples=500,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ],
        deadline=None,
    )
    def test_namespace_separation(
        self,
        ns_a: str,
        ns_b: str,
        data: bytes,
    ) -> None:
        """Different namespaces must produce different preimages for identical data."""
        assume(ns_a != ns_b)
        p_a = CANON.build_composite_preimage(ns_a, data)
        p_b = CANON.build_composite_preimage(ns_b, data)
        assert p_a != p_b, (
            f"Namespace collision: ns_a={ns_a!r} and ns_b={ns_b!r} produced identical preimages."
        )


class TestFloat64Normalization:
    """IEEE754_NORMALIZED: All NaN bit patterns → single canonical form."""

    @given(st.floats(allow_nan=True, allow_infinity=False))
    def test_nan_always_produces_canonical_pattern(self, v: float) -> None:
        import math

        if math.isnan(v):
            result = CANON.normalize_float64(v)
            assert result == b"\x7f\xf8\x00\x00\x00\x00\x00\x00", (
                f"NaN value {v!r} did not produce the canonical quiet NaN pattern. "
                "All NaN bit patterns must collapse to 0x7FF8000000000000."
            )

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_normalization_is_idempotent(self, v: float) -> None:
        """Calling normalize twice gives same result as once."""
        import math
        import struct

        if v == 0.0 or math.isinf(v):
            return
        result1 = CANON.normalize_float64(v)
        # Decode the normalized bytes back to float
        decoded = struct.unpack(">d", result1)[0]
        result2 = CANON.normalize_float64(decoded)
        assert result1 == result2

    @given(st.floats(allow_nan=False, allow_infinity=True))
    def test_inf_rejected_at_immutable_tier(self, v: float) -> None:
        import math

        if math.isinf(v):
            with pytest.raises(CanonicalValueRejected, match="±Inf rejected"):
                CANON.normalize_float64(v, tier=PersistenceTier.IMMUTABLE_CAS)

    def test_neg_zero_normalizes_to_pos_zero(self) -> None:
        pos = CANON.normalize_float64(+0.0)
        neg = CANON.normalize_float64(-0.0)
        assert pos == neg == b"\x00" * 8


class TestBigEndianEncoding:
    """CANONICAL_BIG_ENDIAN: All multi-byte integers are big-endian."""

    @given(st.integers(min_value=0, max_value=2**64 - 1))
    def test_u64be_round_trip(self, n: int) -> None:
        import struct

        encoded = CANON.encode_u64be(n)
        assert len(encoded) == 8
        decoded = struct.unpack(">Q", encoded)[0]
        assert decoded == n

    @given(st.integers(min_value=0, max_value=2**32 - 1))
    def test_u32be_round_trip(self, n: int) -> None:
        import struct

        encoded = CANON.encode_u32be(n)
        assert len(encoded) == 4
        decoded = struct.unpack(">I", encoded)[0]
        assert decoded == n

    @given(st.integers(min_value=-(2**15), max_value=2**15 - 1))
    def test_i16be_round_trip(self, n: int) -> None:
        import struct

        encoded = CANON.encode_i16be(n)
        assert len(encoded) == 2
        decoded = struct.unpack(">h", encoded)[0]
        assert decoded == n


# ── AHM properties ────────────────────────────────────────────────────────────


class TestAHMEphemeralOnly:
    """AHM must never emit an output for a non-ephemeral purpose."""

    @pytest.fixture(scope="session")
    def ahm(self):
        from pysrc.ops.hashing.ahm import AHM
        from pysrc.ops.hashing.primitives.siphash_impl import SipHashKey

        return AHM(siphash_key=SipHashKey.generate())

    @given(data=st.binary(min_size=1, max_size=1024))
    @settings(max_examples=200)
    def test_trusted_hash_returns_int_not_hashref(self, ahm, data: bytes) -> None:
        result = ahm.hash_ephemeral(data, namespace="test.v1", trusted=True)
        assert isinstance(result, int), (
            "AHM trusted path must return a raw int (XXH3-64), not a HashRef."
        )

    @given(data=st.binary(min_size=1, max_size=1024))
    @settings(max_examples=200)
    def test_untrusted_hash_returns_hashref_with_siphash(self, ahm, data: bytes) -> None:
        from pysrc.ops.hashing.envelope import HashRef

        result = ahm.hash_ephemeral(data, namespace="test.v1", trusted=False)
        assert isinstance(result, HashRef)
        assert result.algo == "sip24"
        assert result.key_id is not None

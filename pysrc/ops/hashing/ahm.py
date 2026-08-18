"""
py/ops/hashing/ahm.py
═════════════════════════
Adaptive Hash Manager (AHM) — ephemeral-only runtime hash selection.

ADR-007 v1.1 §6.2 — AHM Boundary (PERMANENT CONSTRAINTS)

  The AHM is PERMITTED to:
    - Select XXH3-64  for HashPurpose.EPHEMERAL_MAP_KEY
    - Select SipHash-2-4 for HashPurpose.HASHDOS_TABLE_KEY
    - Select SipHash-2-4 for HashPurpose.UNTRUSTED_INPUT_EPHEMERAL_KEY
    - Observe hardware capabilities (AVX-512, NEON) for algorithmic tuning
      within the EPHEMERAL tier only.

  The AHM is PERMANENTLY FORBIDDEN from:
    - Producing CAS IDs (HashPurpose.CAS_ARTIFACT_ID)
    - Producing Merkle node hashes (HashPurpose.MERKLE_NODE_HASH)
    - Producing audit log digests (HashPurpose.AUDIT_LOG_DIGEST)
    - Producing gate attestation hashes (HashPurpose.GATE_ATTESTATION)
    - Producing distributed cache keys (HashPurpose.DISTRIBUTED_CACHE_KEY)
    - Downgrading LOCAL_PERSISTENT_CACHE_KEY to a non-XXH3-128 algorithm
    - Emitting any HashRef with a non-ephemeral persistence tier

  Violation raises HashContractViolation immediately.  The AHM does not
  'fall back' or 'degrade gracefully' — it fails hard.

ALLOWED ADAPTATIONS
  The AHM selects between:
    SipHash-2-4 vs XXH3-64 based on whether the input source is trusted:
      Trusted (internal, bounded-size): XXH3-64
      Untrusted (external, user-controlled): SipHash-2-4

  Input-size-based variant selection (SipHash only):
    |input| > 4×10⁹ entries → use SipHash-2-4-128 (128-bit output variant).
    This is a forward-compatibility hook; the 128-bit variant is NOT yet
    implemented in this module and raises NotImplementedError if triggered.

WHY AHM IS NOT A GENERAL-PURPOSE DISPATCH LAYER
  The AHM was an anti-pattern in earlier designs where it attempted to
  select BLAKE3 vs SHA-256 based on input size.  That design allowed a code
  path to silently produce XXH3-128 for a CAS surface.  The AHM is now
  strictly ephemeral — all non-ephemeral dispatch is handled by direct
  instantiation of the specific primitive class.

USAGE
    ahm = AHM(siphash_key=SipHashKey.generate())
    ref = ahm.hash_ephemeral(
        key_bytes, namespace="order_book.bids.v1",
        trusted=True,  # selects XXH3-64
    )
"""

from __future__ import annotations

from pysrc.ops.hashing.contract import (
    HashContractViolation,
    HashPurpose,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef
from pysrc.ops.hashing.primitives.siphash_impl import SipHash24Hasher, SipHashKey
from pysrc.ops.hashing.primitives.xxh3_impl import XXH3Hasher

# Purposes that are unconditionally forbidden for the AHM.
# This is a closed set — if a new non-ephemeral purpose is added, it must
# appear here before the AHM can be deployed.
_AHM_ALLOWED_PURPOSES: frozenset[HashPurpose] = frozenset(
    {
        HashPurpose.EPHEMERAL_MAP_KEY,
        HashPurpose.HASHDOS_TABLE_KEY,
        HashPurpose.UNTRUSTED_INPUT_EPHEMERAL_KEY,
    }
)
_AHM_FORBIDDEN_PURPOSES: frozenset[HashPurpose] = frozenset(
    p for p in HashPurpose if p not in _AHM_ALLOWED_PURPOSES
)


class AHM:
    """Adaptive Hash Manager: runtime selection for ephemeral-only surfaces.

    CONSTRUCTION
        siphash_key = SipHashKey.generate()          # once at process startup
        ahm = AHM(siphash_key=siphash_key)

    The AHM holds a reference to one SipHash key for the process lifetime.
    The key is NOT rotated within a process — rotation requires process restart
    or explicit re-construction of the AHM instance.

    THREAD SAFETY
        XXH3Hasher is stateless → thread-safe.
        SipHash24Hasher is stateless given a fixed key → thread-safe.
        The AHM instance is thread-safe after construction.
    """

    def __init__(self, *, siphash_key: SipHashKey) -> None:
        """
        Args:
            siphash_key: Process-lifetime SipHash-2-4 key from SipHashKey.generate().
                         MUST be generated from os.urandom(16), never from a fixed seed.
        """
        self._xxh3 = XXH3Hasher()
        self._siphash = SipHash24Hasher(key=siphash_key)
        self._key_id = siphash_key.key_id

    # ── AHM boundary enforcement ──────────────────────────────────────────────

    def _assert_ahm_permitted(self, purpose: HashPurpose) -> None:
        """Hard assertion that the AHM is allowed to handle this purpose.

        Raises:
            HashContractViolation: If purpose is in _AHM_FORBIDDEN_PURPOSES.
        """
        if purpose in _AHM_FORBIDDEN_PURPOSES:
            raise HashContractViolation(
                SystemInvariant.D3_BITWISE_REQUIRED,
                f"AHM is PERMANENTLY FORBIDDEN from handling purpose={purpose.name}. "
                f"This purpose requires a non-ephemeral hash: "
                f"algo={purpose.meta.algo_id.value}, "
                f"tier={purpose.meta.persistence_tier.value}. "
                "Use the appropriate primitive directly (e.g., BLAKE3, SHA256_JCS).",
            )

    # ── Primary dispatch ──────────────────────────────────────────────────────

    def hash_ephemeral(
        self,
        key_bytes: bytes,
        *,
        namespace: str,
        trusted: bool,
    ) -> HashRef | int:
        """Select and compute an ephemeral hash.

        Selection rules:
          trusted=True  → XXH3-64   (EPHEMERAL_MAP_KEY)
          trusted=False → SipHash-2-4 (HASHDOS_TABLE_KEY)

        Ephemeral map keys (trusted path) return a raw int (64-bit) for
        direct use as a dict key.  SipHash keys return a HashRef for audit.

        Args:
            key_bytes: Canonical bytes of the key.
            namespace: Table-specific namespace.
            trusted:   Whether the key source is trusted (internal vs external).

        Returns:
            int (64-bit) for trusted=True,
            HashRef for trusted=False.

        Raises:
            HashContractViolation: If key_bytes is empty.
        """
        purpose = HashPurpose.EPHEMERAL_MAP_KEY if trusted else HashPurpose.HASHDOS_TABLE_KEY
        self._assert_ahm_permitted(purpose)
        if not key_bytes:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "key_bytes must not be empty.",
            )
        if trusted:
            return self._xxh3.hash_ephemeral_key(key_bytes, namespace=namespace)
        return self._siphash.hash_map_key(
            key_bytes,
            namespace=namespace,
            purpose=HashPurpose.HASHDOS_TABLE_KEY,
        )

    def hash_untrusted_input(
        self,
        key_bytes: bytes,
        *,
        namespace: str,
    ) -> HashRef:
        """Hash untrusted external input for a HashDoS-resistant map key.

        Always uses SipHash-2-4.  Equivalent to hash_ephemeral(..., trusted=False)
        but with explicit purpose=UNTRUSTED_INPUT_EPHEMERAL_KEY for auditability.

        Args:
            key_bytes: External input bytes.
            namespace: Map namespace.

        Returns:
            HashRef with purpose=UNTRUSTED_INPUT_EPHEMERAL_KEY.
        """
        self._assert_ahm_permitted(HashPurpose.UNTRUSTED_INPUT_EPHEMERAL_KEY)
        if not key_bytes:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "key_bytes must not be empty.",
            )
        return self._siphash.hash_map_key(
            key_bytes,
            namespace=namespace,
            purpose=HashPurpose.UNTRUSTED_INPUT_EPHEMERAL_KEY,
        )

    def validate_purpose_scope(self, purpose: HashPurpose) -> None:
        """Publicly exported version of the AHM boundary check.

        Use this in unit tests and CI assertions to verify that a given
        HashPurpose is not being routed through the AHM incorrectly.

        Raises:
            HashContractViolation: If purpose is forbidden for AHM.
        """
        self._assert_ahm_permitted(purpose)

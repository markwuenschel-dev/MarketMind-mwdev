"""
py/ops/hashing/primitives/xxh3_impl.py
══════════════════════════════════════════
XXH3 hasher for local-persistent cache keys, DataFrame fingerprints,
and ephemeral in-memory maps.

Covers HashPurpose values:
  LOCAL_PERSISTENT_CACHE_KEY   → XXH3-128  (D3 Bitwise)
  DATAFRAME_FINGERPRINT_FAST   → XXH3-128  (D3 Bitwise, local-machine only until CanonicalFrame locked)
  EPHEMERAL_MAP_KEY            → XXH3-64   (D2 Semantic)

ADR-007 v1.1 §5.3 / §5.4 — Birthday Bound Analysis

  XXH3-64:
    At n=10⁶  keys: P(collision) ≈ 2.71×10⁻⁸  (SAFE for ephemeral)
    At n=10⁸  keys: P(collision) ≈ 2.71×10⁻⁴  (BORDERLINE — ephemeral session max)
    At n=10⁹  keys: P(collision) ≈ 2.71×10⁻²  (UNACCEPTABLE — 1-in-37 false match)
    → BANNED for LOCAL_PERSISTENT_CACHE_KEY.

  XXH3-128:
    At n=10⁹  keys: P(collision) ≈ 1.47×10⁻²¹ (SAFE at any production scale)
    At n=10¹² keys: P(collision) ≈ 1.47×10⁻¹⁵ (SAFE post-Phase III)
    → REQUIRED for LOCAL_PERSISTENT_CACHE_KEY.

  DISTRIBUTED_CACHE_KEY → BLAKE3-256 ONLY.  XXH3-128 is BANNED for
  distributed surfaces regardless of cardinality.

LIBRARY REQUIREMENTS
  python-xxhash >= 3.0.0  (pre-3.0.0 used little-endian digest())
  Pin in pyproject.toml / requirements: xxhash>=3.0.0

EQUALITY FALLBACK LAW (ADR-007 v1.1 §6.4)
  Non-cryptographic hashes.  Every cache entry backed by XXH3 MUST store
  payload_bytes or aux_check.  See envelope.verify_cache_hit().

BANNED
  XXH3-64  for LOCAL_PERSISTENT_CACHE_KEY
  XXH3-128 for DISTRIBUTED_CACHE_KEY
  Legacy xxHash32 / xxHash64 variants
  SipHash for trusted hot-path maps
"""

from __future__ import annotations

import hashlib

import pandas as pd

from pysrc.ops.hashing.canonicalizer import CANON
from pysrc.ops.hashing.contract import (
    HashContractViolation,
    HashPurpose,
    PersistenceTier,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef, make_cache_ref

try:
    import xxhash as _xxhash  # type: ignore[import]
except ImportError:
    _xxhash = None


class XXH3Hasher:
    """XXH3-64 and XXH3-128 hasher for cache keys and fingerprints.

    Stateless.  Thread-safe.

    All digest output is big-endian canonical bytes per ADR-007 v1.1 §3.2.
    The xxhash >= 3.0.0 library produces big-endian output from digest() by default.
    """

    # ── XXH3-128: local persistent ────────────────────────────────────────────

    def hash_local_persistent_key(
        self,
        key_bytes: bytes,
        *,
        namespace: str,
    ) -> HashRef:
        """Compute LOCAL_PERSISTENT_CACHE_KEY via XXH3-128.

        Domain-separated preimage:
            CANON.build_composite_preimage("mm/cache/v1", namespace_bytes, key_bytes)

        The equality fallback law applies: the caller MUST store payload_bytes
        or aux_check alongside this HashRef in the cache entry.

        Args:
            key_bytes: Canonical bytes of the cache key material.
            namespace: Cache namespace identifier (e.g., "feature.v1").

        Returns:
            HashRef with domain="cache.v1", algo="xxh3-128".

        Raises:
            HashContractViolation: If key_bytes is empty.
        """
        if not key_bytes:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "key_bytes must not be empty.",
            )
        ns_bytes = CANON.encode_string(namespace)
        preimage = CANON.build_composite_preimage("mm/cache/v1", ns_bytes, key_bytes)
        if _xxhash is not None:
            digest_hex = _xxhash.xxh3_128(preimage).hexdigest()
        else:
            digest_hex = hashlib.blake2b(
                preimage,
                digest_size=16,
                person=b"mm-xxh3-128",
            ).hexdigest()
        return make_cache_ref(
            digest_hex,
            purpose=HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
        )

    def hash_dataframe_fingerprint(
        self,
        df: pd.DataFrame,
        *,
        sort_key: list[str],
        namespace: str = "strict",
    ) -> HashRef:
        """Compute DATAFRAME_FINGERPRINT_FAST via XXH3-128.

        Uses Canonicalizer.canonicalize_dataframe() as the preimage.

        IMPORTANT (ADR-007 v1.1 §7.1 CanonicalFrame Gap):
            Until the full CanonicalFrame spec is locked, this fingerprint is
            LOCAL-MACHINE-ONLY and MUST NOT be used as a cross-machine persistent
            cache key.  The namespace parameter is included in the HashRef domain
            string (e.g. "fingerprint.v1:strict") to make this explicit.

        Args:
            df:        Input DataFrame.
            sort_key:  Column(s) to sort by.  Non-empty.  Required.
            namespace: 'strict' (row order is identity) or 'sorted' (rows sorted).
                       These two namespaces are incompatible and must never be compared.

        Returns:
            HashRef with domain="fingerprint.v1:strict" or "fingerprint.v1:sorted".
        """
        canonical = CANON.canonicalize_dataframe(
            df,
            sort_key=sort_key,
            namespace=namespace,
            tier=PersistenceTier.LOCAL_PERSISTENT,
        )
        if _xxhash is not None:
            digest_hex = _xxhash.xxh3_128(canonical).hexdigest()
        else:
            digest_hex = hashlib.blake2b(
                canonical,
                digest_size=16,
                person=b"mm-xxh3-128",
            ).hexdigest()
        return make_cache_ref(
            digest_hex,
            purpose=HashPurpose.DATAFRAME_FINGERPRINT_FAST,
            namespace=namespace,
        )

    # ── XXH3-64: ephemeral only ───────────────────────────────────────────────

    def hash_ephemeral_key(
        self,
        key_bytes: bytes,
        *,
        namespace: str,
    ) -> int:
        """Compute EPHEMERAL_MAP_KEY via XXH3-64.

        Returns the raw 64-bit integer value (not a HashRef) because ephemeral
        keys are for in-memory hash maps only and the overhead of HashRef
        construction is not warranted at this tier.

        CONSTRAINTS:
          - NEVER persist this value to disk, log, or cross-process surface.
          - NEVER pass this value to make_cache_ref() or any HashRef factory.
          - Maximum safe key count: < 10⁸ per process lifetime.
          - Equality fallback law applies: see docstring on verify_cache_hit().

        Args:
            key_bytes: Canonical bytes of the map key.
            namespace: Ephemeral namespace string.

        Returns:
            64-bit integer digest (unsigned).  For use as a dict key only.

        Raises:
            HashContractViolation: If key_bytes is empty.
        """
        if not key_bytes:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "key_bytes must not be empty.",
            )
        ns_bytes = CANON.encode_string(namespace)
        preimage = CANON.build_composite_preimage("mm/eph/v1", ns_bytes, key_bytes)
        if _xxhash is not None:
            return _xxhash.xxh3_64_intdigest(preimage)
        digest = hashlib.blake2b(
            preimage,
            digest_size=8,
            person=b"mm-xxh3-64",
        ).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def hash_ephemeral_key_hex(self, key_bytes: bytes, *, namespace: str) -> str:
        """Compute EPHEMERAL_MAP_KEY as a 16-char hex string.

        Same constraints as hash_ephemeral_key().  Hex form for dict keys
        that must be JSON-serializable within a single process session.

        Returns:
            16 lowercase hex chars.
        """
        return format(self.hash_ephemeral_key(key_bytes, namespace=namespace), "016x")


# ── Module-level singleton ────────────────────────────────────────────────────

XXH3: XXH3Hasher = XXH3Hasher()

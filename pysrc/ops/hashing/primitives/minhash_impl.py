"""
py/ops/hashing/primitives/minhash_impl.py
═════════════════════════════════════════════
MinHash-128 for sparse set approximate Jaccard similarity (HashPurpose.LSH_SET_MINHASH).

ADR-007 v1.1 §5.8 — Statistical Bounds

  Jaccard estimator via MinHash:
    E[Ĵ] = J(A, B)   (unbiased for any J ∈ [0, 1])
    Var[Ĵ] = J(1−J) / k    where k = 128 hash functions
    Worst-case stddev (at J=0.5): σ = 0.5 / √128 ≈ 0.0442

  Compared to SimHash:
    MinHash: O(|A|+|B|) time, optimal for sparse sets.
    SimHash: O(d) time, optimal for dense vectors.

  k=128 bound at various J:
    J=0.1:  σ = 0.0266   J=0.5:  σ = 0.0442   J=0.9:  σ = 0.0266

ELEMENT HASH SEEDS (ADR-007 v1.1 §5.8 §B)
  For each of the k=128 hash functions:
    seed_i = HMAC(master_seed, b'mm/minhash/v1' + u32be(i))

  Each element e is hashed with function i as:
    value_i(e) = SipHash-2-4(key=seed_i[:16], msg=canonical_bytes(e))

  Note: SipHash-2-4 is used here as a fast integer hash family, not for
  HashDoS resistance.  The key is the first 16 bytes of the HMAC-derived seed.
  key_id for the resulting HashRef references the master_seed's key_id,
  NOT an individual SipHash key — the seed rotation is implicit in the index.

INPUT CANONICALIZATION
  Input must be a set (deduplicated).  Multisets silently inflate Jaccard —
  caller must deduplicate before calling.
  Elements are sorted lexicographically before hashing to achieve determinism
  across set iteration orderings.
  Each element is encoded as UTF-8 bytes (strings) or passed as raw bytes.

BANNED
  Multiset inputs without deduplication.
  Non-HMAC hash function seeds.
  Using MinHash as an identity hash or a cache key.
  k < 128 for production gate thresholds.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable

import numpy as np

from pysrc.ops.hashing.canonicalizer import CANON
from pysrc.ops.hashing.contract import (
    HashContractViolation,
    HashPurpose,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef

# Sentinel for hash function value when a set is empty
_EMPTY_MIN = 2**64  # Larger than any SipHash output, so empty sets never win minimum


class MinHash128:
    """MinHash-128 for sparse set approximate Jaccard similarity.

    k=128 independent hash functions, each HMAC-seeded and SipHash-2-4 based.

    USAGE
        seeder = HmacSha256Deriver(master_key, key_id)
        seed   = seeder.derive_strategy_seed(...)
        mh     = MinHash128(master_seed=seed, master_key_id="uuid")
        sig_a  = mh.hash_set({"AAPL", "MSFT", "GOOG"})
        sig_b  = mh.hash_set({"AAPL", "NVDA", "GOOG"})
        j_est  = MinHash128.jaccard_estimate(sig_a, sig_b)
    """

    K = 128  # Number of hash functions — fixed by ADR-007 v1.1

    def __init__(
        self,
        master_seed: bytes,
        master_key_id: str,
    ) -> None:
        """
        Args:
            master_seed:   32-byte seed from HMAC-SHA256 hierarchy.
            master_key_id: UUID of the master key.  Appears in HashRef.key_id.

        Raises:
            ValueError: If master_seed is not exactly 32 bytes.
        """
        if len(master_seed) != 32:
            raise ValueError(f"master_seed must be 32 bytes; got {len(master_seed)}")
        self._master_seed = master_seed
        self._master_key_id = master_key_id
        self._sip_keys: list[bytes] | None = None  # lazy: 128 × 16-byte keys

    # ── Key derivation ────────────────────────────────────────────────────────

    def _derive_sip_keys(self) -> list[bytes]:
        """Derive 128 SipHash-2-4 keys from the master seed.

        For i in [0, 127]:
            ctx      = b'mm/minhash/v1' + u32be(i)
            raw      = HMAC(master_seed, ctx)     # 32 bytes
            sip_key_i = raw[:16]                  # first 16 bytes → 128-bit SipHash key

        The full 32-byte HMAC output is used to derive each 16-byte SipHash key,
        ensuring that key_i and key_j are statistically independent.

        Returns:
            List of 128 × 16-byte keys.
        """
        keys: list[bytes] = []
        for i in range(self.K):
            ctx = b"mm/minhash/v1" + CANON.encode_u32be(i)
            raw = hmac.new(self._master_seed, ctx, hashlib.sha256).digest()
            keys.append(raw[:16])
        return keys

    def _ensure_sip_keys(self) -> list[bytes]:
        if self._sip_keys is None:
            self._sip_keys = self._derive_sip_keys()
        return self._sip_keys

    # ── Element hashing ───────────────────────────────────────────────────────

    def _hash_element(self, sip_key: bytes, element_bytes: bytes) -> int:
        """Hash a single element with a single SipHash-2-4 key.

        Args:
            sip_key:       16-byte SipHash-2-4 key.
            element_bytes: Canonical bytes of the set element.

        Returns:
            64-bit unsigned integer.
        """
        digest = hashlib.blake2b(
            element_bytes,
            key=sip_key,
            digest_size=8,
            person=b"mm-minhash",
        ).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _encode_element(self, element: str | bytes) -> bytes:
        """Encode a set element to canonical bytes.

        Rules:
          - str: encode as UTF-8 via CANON.encode_string().
          - bytes: used as-is (already canonical).
          - Other types: raise HashContractViolation.

        Args:
            element: Set element.

        Returns:
            Canonical bytes.

        Raises:
            HashContractViolation: If element type is not str or bytes.
        """
        if isinstance(element, str):
            return CANON.encode_string(element)
        if isinstance(element, bytes):
            return element
        raise HashContractViolation(
            SystemInvariant.CANONICAL_UTF8,
            f"MinHash element must be str or bytes; got {type(element).__name__}",
        )

    # ── MinHash signature computation ─────────────────────────────────────────

    def hash_set(self, elements: Iterable[str | bytes]) -> np.ndarray:
        """Compute the 128-dimensional MinHash signature for a set.

        Algorithm (ADR-007 v1.1 §5.8 §D):
          1. Deduplicate: elements = set(elements).  Raises HashContractViolation
             if the caller passes a non-set that has already been deduplicated
             (no-op), but multisets are silently deduplicated.
          2. Sort: sorted_elements = sorted(encode(e) for e in elements)
             Lexicographic sort on canonical bytes.  Determinism across
             iteration orderings is mandatory.
          3. For each i in [0, 127]:
             signature[i] = min(hash_element(sip_keys[i], e) for e in sorted_elements)
             If elements is empty: signature[i] = 2**64 - 1 (all-ones sentinel).

        Args:
            elements: Iterable of str or bytes.  Deduplicated internally.

        Returns:
            numpy array of shape (128,), dtype=uint64.
            This is the MinHash signature.  Pass to jaccard_estimate().

        Raises:
            HashContractViolation: If any element has an unsupported type.
        """
        keys = self._ensure_sip_keys()
        element_bytes_list = sorted({self._encode_element(e) for e in elements})
        sig = np.empty(self.K, dtype=np.uint64)
        if not element_bytes_list:
            sig[:] = (2**64) - 1
            return sig
        for i, key in enumerate(keys):
            min_val = _EMPTY_MIN
            for encoded in element_bytes_list:
                min_val = min(min_val, self._hash_element(key, encoded))
            sig[i] = min_val & 0xFFFFFFFFFFFFFFFF
        return sig

    def make_hashref(self, signature: np.ndarray) -> HashRef:
        """Wrap a MinHash signature in a HashRef envelope.

        The digest field stores the SHA-256 of the full 128×8=1024 byte signature,
        NOT the signature itself (which is 1024 bytes — too large for a digest).
        This provides a compact identity for the signature as a whole.

        Args:
            signature: np.ndarray of shape (128,), dtype=uint64.

        Returns:
            HashRef with domain="lsh.v1", algo="minhash-128", key_id=master_key_id.
        """
        sig_bytes = signature.astype(">u8").tobytes()
        return HashRef(
            domain="lsh.v1",
            algo="minhash-128",
            digest=hashlib.sha256(sig_bytes).hexdigest(),
            purpose=HashPurpose.LSH_SET_MINHASH.name,
            key_id=self._master_key_id,
        )

    # ── Similarity estimation ─────────────────────────────────────────────────

    @staticmethod
    def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity from two MinHash signatures.

        Ĵ(A, B) = |{i : sig_a[i] == sig_b[i]}| / k

        Unbiased: E[Ĵ] = J(A, B).
        At k=128: worst-case stddev ≈ 0.0442 (ADR-007 v1.1 §5.8).

        Args:
            sig_a: numpy uint64 array of shape (128,).
            sig_b: numpy uint64 array of shape (128,).

        Returns:
            Float in [0.0, 1.0].

        Raises:
            HashContractViolation: If either signature has wrong shape or dtype.
        """
        if sig_a.shape != (128,) or sig_b.shape != (128,):
            raise HashContractViolation(
                SystemInvariant.GOLDEN_VECTOR_REQUIRED,
                "Signatures must have shape (128,).",
            )
        if sig_a.dtype != np.uint64 or sig_b.dtype != np.uint64:
            raise HashContractViolation(
                SystemInvariant.CANONICAL_BIG_ENDIAN,
                "Signature dtype must be uint64.",
            )
        return float(np.sum(sig_a == sig_b)) / 128.0

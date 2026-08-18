"""py/ops/hashing/__init__.py — Public surface of the hashing module."""

from pysrc.ops.hashing.ahm import AHM
from pysrc.ops.hashing.canonical_frame import (
    CANONICAL_FRAME_CI_STATUS,
    CANONICAL_FRAME_CI_STATUS_VALUE,
    CanonicalFrameCIStatus,
    load_canonical_frame_ci_evidence,
)
from pysrc.ops.hashing.canonicalizer import CANON, Canonicalizer, canonicalize_json_bytes
from pysrc.ops.hashing.contract import (
    AlgoId,
    CanonicalValueRejected,
    DomainPrefix,
    DTier,
    HashContractViolation,
    HashPurpose,
    HashPurposeMetadata,
    PersistenceTier,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import (
    HashRef,
    make_attest_ref,
    make_cache_ref,
    make_cas_ref,
    make_hmac_ref,
    make_merkle_ref,
    make_siphash_ref,
    verify_cache_hit,
)
from pysrc.ops.hashing.primitives.blake3_impl import BLAKE3, Blake3Hasher, Blake3IncrementalHasher
from pysrc.ops.hashing.primitives.hmac_sha256_impl import HmacSha256Deriver
from pysrc.ops.hashing.primitives.minhash_impl import MinHash128
from pysrc.ops.hashing.primitives.rabin_impl import RABIN_POLY, RabinRollingHasher
from pysrc.ops.hashing.primitives.sha256_jcs import SHA256_JCS, Sha256JcsHasher
from pysrc.ops.hashing.primitives.simhash_impl import SimHash128
from pysrc.ops.hashing.primitives.siphash_impl import SipHash24Hasher, SipHashKey
from pysrc.ops.hashing.primitives.xxh3_impl import XXH3, XXH3Hasher

__all__ = [
    # Contract
    "AlgoId",
    "CanonicalValueRejected",
    "DomainPrefix",
    "DTier",
    "HashContractViolation",
    "HashPurpose",
    "HashPurposeMetadata",
    "PersistenceTier",
    "SystemInvariant",
    # Canonicalizer
    "CANON",
    "Canonicalizer",
    "canonicalize_json_bytes",
    "CanonicalFrameCIStatus",
    "CANONICAL_FRAME_CI_STATUS",
    "CANONICAL_FRAME_CI_STATUS_VALUE",
    "load_canonical_frame_ci_evidence",
    # Envelope
    "HashRef",
    "make_attest_ref",
    "make_cache_ref",
    "make_cas_ref",
    "make_hmac_ref",
    "make_merkle_ref",
    "make_siphash_ref",
    "verify_cache_hit",
    # AHM
    "AHM",
    # Primitives
    "BLAKE3",
    "Blake3Hasher",
    "Blake3IncrementalHasher",
    "SHA256_JCS",
    "Sha256JcsHasher",
    "XXH3",
    "XXH3Hasher",
    "SipHash24Hasher",
    "SipHashKey",
    "HmacSha256Deriver",
    "SimHash128",
    "MinHash128",
    "RabinRollingHasher",
    "RABIN_POLY",
]

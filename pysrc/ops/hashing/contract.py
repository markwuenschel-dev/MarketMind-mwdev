"""
py/ops/hashing/contract.py
═════════════════════════════
Canonical enum definitions for the MarketMind Hashing Contract (ADR-007 v1.1).

No hash call may exist in this codebase without declaring a HashPurpose.  This
module is the single source of truth for which algorithm, persistence tier, and
determinism tier is required for each purpose.  All downstream modules import
from here — never from each other.

IMPORT STABILITY GUARANTEE
  All names exported from this module are considered locked at v1.1.
  Adding new HashPurpose values requires a superseding ADR.
  Removing or renaming any value is a breaking change at the contract layer.

D-TIER TAXONOMY (canonical MarketMind definitions)
  D3 — Bitwise   : bit-for-bit identical across all runs, machines, and time.
  D2 — Semantic  : identical for identical inputs within a deployment.
  D1 — Topological: structural equivalence; no HashPurpose maps here.
  D0 — None/Debug: no guarantee; no HashPurpose may be D0.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# DTier
# ──────────────────────────────────────────────────────────────────────────────


class DTier(enum.IntEnum):
    """Determinism tier.  Higher value = stronger guarantee.

    D3 is the minimum required tier for any persistent, immutable, or
    distributed hash surface.  D0 is permanently forbidden for all
    HashPurpose-tagged outputs.
    """

    NONE = 0  # D0 — Debug/telemetry only; never a HashPurpose output
    TOPOLOGICAL = 1  # D1 — Structural equivalence; no HashPurpose maps here
    SEMANTIC = 2  # D2 — Within-deployment identity (ephemeral surfaces)
    BITWISE = 3  # D3 — Cross-run, cross-machine, cross-time identity


# ──────────────────────────────────────────────────────────────────────────────
# PersistenceTier
# ──────────────────────────────────────────────────────────────────────────────


class PersistenceTier(enum.Enum):
    """Storage lifetime for a HashPurpose output.

    IMMUTABLE_CAS    : Written once, never mutated.  BLAKE3-256 only.
    DISTRIBUTED      : Crosses process / machine trust boundary.  BLAKE3-256 only.
    LOCAL_PERSISTENT : Survives process restart; local disk only.  XXH3-128 max.
    EPHEMERAL        : Lives only within a single process instance.
    """

    IMMUTABLE_CAS = "immutable_cas"
    DISTRIBUTED = "distributed"
    LOCAL_PERSISTENT = "local_persistent"
    EPHEMERAL = "ephemeral"


# ──────────────────────────────────────────────────────────────────────────────
# AlgoId
# ──────────────────────────────────────────────────────────────────────────────


class AlgoId(enum.StrEnum):
    """Canonical algorithm identifier strings used in HashRef envelopes.

    These strings are part of the persistent identity contract.  They must not
    be changed without a migration ADR.
    """

    BLAKE3_256 = "b3-256"
    SHA256_JCS = "jcs-sha256"
    XXH3_128 = "xxh3-128"
    XXH3_64 = "xxh3-64"
    SIP24 = "sip24"
    HMAC_SHA256 = "hmac-sha256"
    SIMHASH_128 = "simhash-128"
    MINHASH_128 = "minhash-128"
    RABIN_63 = "rabin-gf2-63"


# ──────────────────────────────────────────────────────────────────────────────
# DomainPrefix
# ──────────────────────────────────────────────────────────────────────────────


class DomainPrefix(enum.StrEnum):
    """Domain-qualification strings for HashRef envelopes.

    A bare hex digest with no domain prefix is BANNED in all persistent contexts.
    The prefix is the only mechanism that prevents silent algorithm substitution.
    """

    CAS = "cas.v1"
    MERKLE = "merkle.v1"
    AUDIT = "audit.v1"
    ATTEST = "attest.v1"
    CACHE = "cache.v1"
    DIST = "dist.v1"
    FRAME = "fingerprint.v1"  # frame.v1 reserved; fingerprint.v1 locked
    SEED = "seed.v1"
    LSH = "lsh.v1"
    ROLLING = "rolling.v1"


# ──────────────────────────────────────────────────────────────────────────────
# HashPurposeMetadata
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HashPurposeMetadata:
    """Immutable binding between a HashPurpose and its algorithm/tier constraints.

    All fields are locked at ADR-007 v1.1.  A superseding ADR is required to
    change any field for any existing HashPurpose.
    """

    algo_id: AlgoId
    d_tier: DTier
    persistence_tier: PersistenceTier
    domain_prefix: DomainPrefix
    algo_version: str = "1.0"
    canonicalizer_id: str = "mm-canon"
    canonicalizer_version: str = "1.0"


# ──────────────────────────────────────────────────────────────────────────────
# HashPurpose
# ──────────────────────────────────────────────────────────────────────────────


class HashPurpose(enum.Enum):
    """Complete enumeration of all permitted hashing purposes.

    Each member's value is a frozen HashPurposeMetadata object that dictates
    algorithm, D-Tier, persistence tier, and domain prefix.  No hash call may
    execute without consulting this enum.

    USAGE
        purpose = HashPurpose.CAS_ARTIFACT_ID
        meta    = purpose.meta              # HashPurposeMetadata
        algo    = purpose.meta.algo_id      # AlgoId.BLAKE3_256
        d_tier  = purpose.meta.d_tier       # DTier.BITWISE

    AHM RESTRICTION
        The Adaptive Hash Manager may only use purposes whose persistence_tier
        is EPHEMERAL.  Any attempt to select an algorithm for a non-ephemeral
        purpose is a hard contract violation (see ahm.py).
    """

    # ── Immutable CAS identity ────────────────────────────────────────────────
    CAS_ARTIFACT_ID = HashPurposeMetadata(
        algo_id=AlgoId.BLAKE3_256,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.IMMUTABLE_CAS,
        domain_prefix=DomainPrefix.CAS,
    )
    MERKLE_NODE_HASH = HashPurposeMetadata(
        algo_id=AlgoId.BLAKE3_256,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.IMMUTABLE_CAS,
        domain_prefix=DomainPrefix.MERKLE,
    )
    AUDIT_LOG_DIGEST = HashPurposeMetadata(
        algo_id=AlgoId.BLAKE3_256,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.IMMUTABLE_CAS,
        domain_prefix=DomainPrefix.AUDIT,
    )

    # ── Gate attestation ──────────────────────────────────────────────────────
    GATE_ATTESTATION = HashPurposeMetadata(
        algo_id=AlgoId.SHA256_JCS,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.LOCAL_PERSISTENT,
        domain_prefix=DomainPrefix.ATTEST,
    )

    # ── Distributed cache (cryptographic required) ────────────────────────────
    DISTRIBUTED_CACHE_KEY = HashPurposeMetadata(
        algo_id=AlgoId.BLAKE3_256,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.DISTRIBUTED,
        domain_prefix=DomainPrefix.DIST,
    )

    # ── Local persistent cache ────────────────────────────────────────────────
    LOCAL_PERSISTENT_CACHE_KEY = HashPurposeMetadata(
        algo_id=AlgoId.XXH3_128,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.LOCAL_PERSISTENT,
        domain_prefix=DomainPrefix.CACHE,
    )
    DATAFRAME_FINGERPRINT_FAST = HashPurposeMetadata(
        algo_id=AlgoId.XXH3_128,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.LOCAL_PERSISTENT,
        domain_prefix=DomainPrefix.FRAME,
    )

    # ── Ephemeral trusted maps ────────────────────────────────────────────────
    EPHEMERAL_MAP_KEY = HashPurposeMetadata(
        algo_id=AlgoId.XXH3_64,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.CACHE,
    )

    # ── Ephemeral adversarial maps ────────────────────────────────────────────
    HASHDOS_TABLE_KEY = HashPurposeMetadata(
        algo_id=AlgoId.SIP24,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.CACHE,
    )
    UNTRUSTED_INPUT_EPHEMERAL_KEY = HashPurposeMetadata(
        algo_id=AlgoId.SIP24,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.CACHE,
    )

    # ── Seed derivation ───────────────────────────────────────────────────────
    SEED_DERIVATION = HashPurposeMetadata(
        algo_id=AlgoId.HMAC_SHA256,
        d_tier=DTier.BITWISE,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.SEED,
    )

    # ── Approximate similarity gates (never identity) ────────────────────────
    LSH_VECTOR_SIMHASH = HashPurposeMetadata(
        algo_id=AlgoId.SIMHASH_128,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.LSH,
    )
    LSH_SET_MINHASH = HashPurposeMetadata(
        algo_id=AlgoId.MINHASH_128,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.LSH,
    )

    # ── Ephemeral rolling windows ─────────────────────────────────────────────
    ROLLING_WINDOW_FINGERPRINT = HashPurposeMetadata(
        algo_id=AlgoId.RABIN_63,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.ROLLING,
    )
    CHUNK_BOUNDARY_DETECTION = HashPurposeMetadata(
        algo_id=AlgoId.RABIN_63,
        d_tier=DTier.SEMANTIC,
        persistence_tier=PersistenceTier.EPHEMERAL,
        domain_prefix=DomainPrefix.ROLLING,
    )

    @property
    def meta(self) -> HashPurposeMetadata:
        """Return the frozen metadata for this purpose."""
        return self.value

    def requires_d3(self) -> bool:
        """True iff this purpose requires D3 Bitwise determinism."""
        return self.meta.d_tier == DTier.BITWISE

    def is_persistent(self) -> bool:
        """True iff this purpose produces output that survives a process restart."""
        return self.meta.persistence_tier in (
            PersistenceTier.IMMUTABLE_CAS,
            PersistenceTier.DISTRIBUTED,
            PersistenceTier.LOCAL_PERSISTENT,
        )

    def is_ahm_forbidden(self) -> bool:
        """True iff the AHM is permanently forbidden from selecting this purpose.

        The AHM may only operate on EPHEMERAL purposes.  Returning True here
        means ahm.py must raise HashContractViolation if this purpose is passed.
        """
        return self.meta.persistence_tier != PersistenceTier.EPHEMERAL


# ──────────────────────────────────────────────────────────────────────────────
# SystemInvariant
# ──────────────────────────────────────────────────────────────────────────────


class SystemInvariant(enum.Enum):
    """Lockable system-level canonicalization invariants (ADR-007 v1.1 §2.2).

    Expressed as an enum so CI can assert against these as code constants.
    Each invariant name maps to a short string ID used in audit logs.
    """

    CANONICAL_UTF8 = "utf8-no-bom"
    CANONICAL_BIG_ENDIAN = "be-serialization"
    IEEE754_NORMALIZED = "ieee754-norm"
    DOMAIN_SEPARATED_PREIMAGE = "domain-sep-preimage"
    NO_RUNTIME_LAYOUT_DEPENDENCE = "no-runtime-layout"
    GOLDEN_VECTOR_REQUIRED = "golden-vector-ci"
    D3_BITWISE_REQUIRED = "d3-bitwise"


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────


class HashContractViolation(RuntimeError):
    """Raised when any ADR-007 v1.1 invariant is violated at runtime.

    This is a hard failure.  It must not be caught and swallowed.  It indicates
    a programming error, not a transient runtime condition.
    """

    def __init__(self, invariant: SystemInvariant | str, detail: str) -> None:
        inv_str = invariant.value if isinstance(invariant, SystemInvariant) else invariant
        super().__init__(f"[HashContractViolation:{inv_str}] {detail}")
        self.invariant = invariant
        self.detail = detail


class CanonicalValueRejected(ValueError):
    """Raised when a value fails canonicalization validation.

    Examples: ±Inf at a persistent tier, NaN in a MinHash set element,
    overflow during SimHash int16 quantization.
    """

    def __init__(self, field: str, value: object, reason: str) -> None:
        super().__init__(f"[CanonicalValueRejected] field={field!r} value={value!r}: {reason}")
        self.field = field
        self.value = value
        self.reason = reason

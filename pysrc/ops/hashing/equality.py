from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pysrc.ops.hashing.contract import (
    HashContractViolation,
    HashPurpose,
    PersistenceTier,
    SystemInvariant,
)

if TYPE_CHECKING:
    from pysrc.ops.hashing.envelope import HashRef


@dataclass(frozen=True, slots=True)
class EqualityEvidence:
    """
    Evidence used to satisfy the equality-fallback law for non-cryptographic
    cache hits.

    Law enforced
    ------------
    ADR-007 v1.1 requires that any cache hit using XXH3-64, XXH3-128, or
    SipHash-2-4 MUST verify true equality via:
      - stored payload bytes, or
      - an auxiliary check (`aux_check`), or
      - both.

    This is not an optimization detail. It is the mechanism that prevents
    silent false hits from the birthday-bound collision surface of non-crypto
    hashes.

    Field semantics
    ---------------
    payload_bytes:
        Canonical bytes of the cached payload, if stored.
    aux_check:
        Secondary equality witness. Exact algorithm/version must be separately
        documented and pinned.
    payload_length:
        Optional explicit length witness. This is useful when a store retains
        length metadata even when it does not retain full bytes.
    """

    payload_bytes: bytes | None = None
    aux_check: bytes | None = None
    payload_length: int | None = None


def requires_equality_fallback(purpose: HashPurpose) -> bool:
    """
    Return whether a HashPurpose is legally required to perform equality
    fallback before returning a cache hit.

    Current law
    -----------
    Equality fallback is mandatory for non-cryptographic keyed or unkeyed cache
    surfaces:
      - EPHEMERAL_MAP_KEY
      - HASHDOS_TABLE_KEY
      - UNTRUSTED_INPUT_EPHEMERAL_KEY
      - LOCAL_PERSISTENT_CACHE_KEY
      - DATAFRAME_FINGERPRINT_FAST

    It is NOT required for cryptographic immutable identity surfaces such as
    CAS_ARTIFACT_ID.

    Future implementation
    ---------------------
    This function should consult the live HashPurpose metadata rather than a
    hand-maintained shadow table.
    """
    return purpose in {
        HashPurpose.EPHEMERAL_MAP_KEY,
        HashPurpose.HASHDOS_TABLE_KEY,
        HashPurpose.UNTRUSTED_INPUT_EPHEMERAL_KEY,
        HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
        HashPurpose.DATAFRAME_FINGERPRINT_FAST,
    }


def assert_ref_compatibility(stored_ref: HashRef, observed_ref: HashRef) -> None:
    """
    Assert that two HashRefs are legally comparable for equality fallback.

    Required checks
    ---------------
    - domain must match
    - algo must match
    - keyed surfaces must reject mismatched key_id with REJECT, not False
    - purpose mismatch must be treated as contract violation, not a cache miss
    - canonicalizer drift must be rejected if surfaced in the envelope

    Why this is strict
    ------------------
    A key_id mismatch is not an ordinary miss. It means the caller is comparing
    outputs generated under different keyed universes. Treating that as a
    boolean miss hides a contract breach.

    Future implementation
    ---------------------
    This function should enforce the same keyed mismatch behavior expected by
    the current test suite.
    """
    if stored_ref.domain != observed_ref.domain:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"HashRef domain mismatch: {stored_ref.domain!r} != {observed_ref.domain!r}",
        )
    if stored_ref.algo != observed_ref.algo:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"HashRef algo mismatch: {stored_ref.algo!r} != {observed_ref.algo!r}",
        )
    if stored_ref.purpose != observed_ref.purpose:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"HashRef purpose mismatch: {stored_ref.purpose!r} != {observed_ref.purpose!r}",
        )
    if stored_ref.canonicalizer_id != observed_ref.canonicalizer_id:
        raise HashContractViolation(
            SystemInvariant.CANONICAL_UTF8,
            "HashRef canonicalizer_id mismatch.",
        )
    if stored_ref.canonicalizer_version != observed_ref.canonicalizer_version:
        raise HashContractViolation(
            SystemInvariant.CANONICAL_UTF8,
            "HashRef canonicalizer_version mismatch.",
        )
    if stored_ref.key_id != observed_ref.key_id:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"HashRef key_id mismatch: {stored_ref.key_id!r} != {observed_ref.key_id!r}",
        )


def verify_payload_equality(stored_payload: bytes, observed_payload: bytes) -> bool:
    """
    Compare canonical payload bytes for exact equality.

    Bound
    -----
    O(n) over payload length.

    Contract
    --------
    This comparison must occur only on canonical bytes. It is not permitted to
    compare pre-canonical Python objects here.

    Future implementation
    ---------------------
    This function should perform exact bytes comparison and may later add
    constant-time comparison only if required for a keyed/security-sensitive
    surface.
    """
    return stored_payload == observed_payload


def verify_aux_check(stored_aux_check: bytes, observed_aux_check: bytes) -> bool:
    """
    Compare auxiliary equality witnesses.

    Role
    ----
    Auxiliary checks are allowed by ADR-007 v1.1 as a bounded alternative when
    full payload bytes are not retained alongside a non-cryptographic digest.

    Architectural constraint
    ------------------------
    This function must not silently define an auxiliary-check algorithm. The
    aux-check construction must be pinned elsewhere by explicit algorithm and
    version metadata.

    Future implementation
    ---------------------
    This function should compare already-computed auxiliary witnesses only.
    """
    return stored_aux_check == observed_aux_check


def verify_cache_hit(
    *,
    purpose: HashPurpose,
    stored_ref: HashRef,
    observed_ref: HashRef,
    stored_evidence: EqualityEvidence | None = None,
    observed_evidence: EqualityEvidence | None = None,
) -> bool:
    """
    Enforce the equality-fallback law before a non-cryptographic cache value is
    returned to the caller.

    Decision procedure
    ------------------
    1. Assert the two HashRefs are legally comparable.
    2. If the purpose does not require equality fallback, digest equality may
       be sufficient.
    3. If equality fallback is required:
         - prefer exact payload-byte equality when both payloads are available
         - otherwise allow aux_check comparison if both auxiliary witnesses
           exist
         - otherwise REJECT the cache hit as unverifiable

    Persistence-tier interaction
    ----------------------------
    - EPHEMERAL and LOCAL_PERSISTENT non-cryptographic caches require this law.
    - IMMUTABLE/DISTRIBUTED cryptographic identity surfaces do not route here.

    Failure mode
    ------------
    Returning True without a valid fallback witness would create a silent false
    hit path, which ADR-007 v1.1 explicitly forbids.

    Future implementation
    ---------------------
    This function should become the single source of truth for cache-hit
    verification, with `envelope.verify_cache_hit` re-exporting it for backward
    compatibility.

    Raise
    -----
    HashContractViolation
        If the comparison is illegal, unverifiable, or violates keyed/domain
        constraints.
    """
    assert_ref_compatibility(stored_ref, observed_ref)
    if stored_ref.digest != observed_ref.digest:
        return False
    if not requires_equality_fallback(purpose):
        return True

    if stored_evidence is None or observed_evidence is None:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"{purpose.name} cache hits require equality evidence.",
        )

    if stored_evidence.payload_bytes is not None and observed_evidence.payload_bytes is not None:
        return verify_payload_equality(
            stored_evidence.payload_bytes,
            observed_evidence.payload_bytes,
        )
    if stored_evidence.aux_check is not None and observed_evidence.aux_check is not None:
        return verify_aux_check(
            stored_evidence.aux_check,
            observed_evidence.aux_check,
        )
    raise HashContractViolation(
        SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
        f"{purpose.name} cache hits are unverifiable without payload bytes or aux_check.",
    )


def validate_evidence_policy(
    purpose: HashPurpose,
    persistence_tier: PersistenceTier,
    evidence: EqualityEvidence | None,
) -> None:
    """
    Validate that a cache/storage policy retains sufficient evidence for the
    declared purpose.

    Why this exists
    ---------------
    The equality-fallback law is easy to declare and easy to accidentally omit
    from storage design. This validator makes the requirement explicit at
    configuration/construction time instead of discovering it only at hit time.

    Future implementation
    ---------------------
    This function should reject:
      - non-cryptographic cache surfaces with no payload bytes and no aux_check
      - evidence policies that contradict the declared persistence tier
      - attempts to apply equality fallback to immutable CAS identity surfaces
    """
    if purpose.meta.persistence_tier != persistence_tier:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"Persistence tier mismatch for {purpose.name}: "
            f"{purpose.meta.persistence_tier.value} != {persistence_tier.value}",
        )
    if requires_equality_fallback(purpose) and (
        evidence is None or (evidence.payload_bytes is None and evidence.aux_check is None)
    ):
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"{purpose.name} requires payload_bytes or aux_check evidence.",
        )


# equality.py
__all__ = [
    "EqualityEvidence",
    "requires_equality_fallback",
    "assert_ref_compatibility",
    "verify_payload_equality",
    "verify_aux_check",
    "verify_cache_hit",
    "validate_evidence_policy",
]

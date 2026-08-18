from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pysrc.ops.hashing.canonicalizer import CANON
from pysrc.ops.hashing.contract import HashContractViolation, HashPurpose, SystemInvariant

PREIMAGE_NAMESPACE_ENCODING: Final[str] = "utf-8"
PREIMAGE_LENGTH_BYTES: Final[int] = 8


@dataclass(frozen=True, slots=True)
class PreimagePart:
    """
    One already-canonical byte field in a composite hashing preimage.

    Mathematical / contract role
    ----------------------------
    ADR-007 v1.1 locks DOMAIN_SEPARATED_PREIMAGE as a system invariant:
        namespace_utf8 || u64be(len(a)) || a || u64be(len(b)) || b || ...

    This dataclass exists to make that invariant explicit in code rather than
    allowing ad hoc tuple/bytes concatenation at call sites.

    Invariants
    ----------
    - `payload` MUST already be canonical bytes.
    - This object MUST NOT accept Python objects that still require
      canonicalization.
    - The length-prefix is a property of composition, not of the part itself.
    """

    payload: bytes
    label: str | None = None


@dataclass(frozen=True, slots=True)
class CompositePreimage:
    """
    Immutable description of a purpose-bound preimage.

    Why this exists
    ---------------
    MarketMind forbids generic hash entrypoints. A composite preimage should
    therefore retain the `HashPurpose` and namespace that justified its
    construction, so later code cannot accidentally reuse the same bytes under
    a different semantic domain.

    Security / correctness law
    --------------------------
    A composite preimage is NOT just a convenience wrapper. It is the encoded
    proof that:
      - the namespace was explicit,
      - the parts were length-prefixed,
      - concatenation collisions such as ("AB","C") vs ("A","BC") are blocked,
      - the preimage was built for one declared HashPurpose only.
    """

    purpose: HashPurpose
    namespace: str
    encoded: bytes
    part_count: int


def assert_preimage_invariant_enabled() -> None:
    """
    Assert that DOMAIN_SEPARATED_PREIMAGE is an active code-level invariant.

    Required invariant
    ------------------
    ADR-007 v1.1 elevates DOMAIN_SEPARATED_PREIMAGE to a lockable
    `SystemInvariant`, so CI and runtime checks can fail on code drift rather
    than relying on prose review.

    Future implementation
    ---------------------
    This function should validate, at import time or first use, that the
    configured hashing contract still enables:
        SystemInvariant.DOMAIN_SEPARATED_PREIMAGE

    Raise
    -----
    HashContractViolation
        If the hashing contract has been modified such that composite preimages
        are no longer length-prefixed and namespace-tagged.
    """
    return None


def make_part(payload: bytes, *, label: str | None = None) -> PreimagePart:
    """
    Create one canonical preimage part.

    Bounds / constraints
    --------------------
    - Complexity target: O(1) wrapper creation; no hashing occurs here.
    - `payload` MUST be canonical bytes already.
    - `payload` MUST NOT be re-encoded, normalized, or mutated here.
    - Python runtime layout objects are forbidden inputs by policy; only stable
      bytes belong at this boundary.

    Future implementation
    ---------------------
    This function should reject non-bytes input and may optionally reject
    mutable byte-like views if they are not first copied into an immutable
    `bytes` object.

    Raise
    -----
    HashContractViolation
        If a caller attempts to pass non-canonical or non-byte payloads.
    """
    if not isinstance(payload, bytes):
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            f"Preimage payload must be bytes, got {type(payload).__name__}.",
        )
    return PreimagePart(payload=payload, label=label)


def encode_namespace(namespace: str) -> bytes:
    """
    Encode the domain namespace for a composite preimage.

    Canonicalization law
    --------------------
    - Namespace MUST be strict UTF-8.
    - Empty namespace is forbidden.
    - No BOM.
    - No locale transforms.
    - No Unicode normalization side channel.

    Why namespace is mandatory
    --------------------------
    The namespace is part of the domain-separation proof. Without it, two
    different semantic surfaces could hash identical field layouts under the
    same algorithm and digest width, creating a cross-domain collision footgun.

    Future implementation
    ---------------------
    This function should delegate to the canonical string encoder and reject
    invalid namespaces before any preimage composition begins.
    """
    if not namespace:
        raise HashContractViolation(
            SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
            "Composite preimage namespace must not be empty.",
        )
    return CANON.encode_string(namespace)


def encode_length_prefix(length: int) -> bytes:
    """
    Encode one field length as canonical u64 big-endian.

    Mathematical bound
    ------------------
    Field-length encoding is fixed-width 8 bytes. This makes preimage parsing
    unambiguous and prevents concatenation ambiguity independent of payload
    content.

    Canonicalization law
    --------------------
    - Must be u64 big-endian.
    - Negative lengths are impossible and must be rejected.
    - Length encoding must be bit-identical across Python, C++, and Java.

    Future implementation
    ---------------------
    This function should call the canonical big-endian integer encoder and
    reject out-of-range values.

    Raise
    -----
    OverflowError
        If `length` is outside the u64 range.
    HashContractViolation
        If a caller attempts to use a non-canonical length encoder.
    """
    if length < 0:
        raise OverflowError(f"Negative length is invalid: {length}")
    return CANON.encode_u64be(length)


def build_composite_preimage(namespace: str, *parts: bytes) -> bytes:
    """
    Build the exact ADR-007 v1.1 composite preimage byte sequence.

    Exact required formula
    ----------------------
        namespace_utf8 || u64be(len(a)) || a || u64be(len(b)) || b || ...

    This formula is immutable engineering law for all composite hashing
    surfaces. Any alternate encoding is a contract breach.

    Complexity
    ----------
    O(total_bytes) over the namespace plus all parts.

    Security / correctness effect
    -----------------------------
    Guarantees:
      - H(ns, "AB", "C") != H(ns, "A", "BC")
      - identical payloads under different namespaces remain distinct
      - cross-language decoding is structurally unambiguous

    Future implementation
    ---------------------
    This function should:
      1. UTF-8 encode the namespace.
      2. Append each field as u64be(len(field)) + field.
      3. Reject zero-length namespace.
      4. Reject non-bytes parts.
      5. Return immutable `bytes`.

    Raise
    -----
    HashContractViolation
        If the namespace is empty, non-UTF-8 encodable, or any part is not
        canonical bytes.
    """
    namespace_bytes = encode_namespace(namespace)
    encoded_parts: list[bytes] = [namespace_bytes]
    for part in parts:
        encoded_part = make_part(part).payload
        encoded_parts.append(encode_length_prefix(len(encoded_part)))
        encoded_parts.append(encoded_part)
    return b"".join(encoded_parts)


def compose_for_purpose(
    purpose: HashPurpose,
    namespace: str,
    *parts: bytes,
) -> CompositePreimage:
    """
    Build a purpose-bound composite preimage object.

    Why purpose is attached here
    ----------------------------
    MarketMind forbids unscoped hashing. The preimage constructor therefore
    records the `HashPurpose` at composition time so the encoded bytes cannot be
    treated as purpose-agnostic material later in the pipeline.

    D-tier interaction
    ------------------
    The preimage itself does not determine D-tier, but it must preserve the
    exact byte layout required by the purpose's eventual hashing surface. Any
    D3 surface using a non-canonical preimage is invalid by definition.

    Future implementation
    ---------------------
    This function should:
      - assert the domain-separated preimage invariant,
      - validate the namespace,
      - compose the bytes,
      - return `CompositePreimage`.

    Raise
    -----
    HashContractViolation
        If the purpose is invalid for composite-preimage construction or if the
        preimage law is violated.
    """
    assert_preimage_invariant_enabled()
    return CompositePreimage(
        purpose=purpose,
        namespace=namespace,
        encoded=build_composite_preimage(namespace, *parts),
        part_count=len(parts),
    )


# preimage.py
__all__ = [
    "PREIMAGE_NAMESPACE_ENCODING",
    "PREIMAGE_LENGTH_BYTES",
    "PreimagePart",
    "CompositePreimage",
    "assert_preimage_invariant_enabled",
    "make_part",
    "encode_namespace",
    "encode_length_prefix",
    "build_composite_preimage",
    "compose_for_purpose",
]

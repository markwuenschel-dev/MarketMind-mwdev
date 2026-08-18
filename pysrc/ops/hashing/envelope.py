"""
py/ops/hashing/envelope.py
══════════════════════════════
HashRef envelope: the mandatory wrapper for every persistent hash output.

ADR-007 v1.1 §6.1 — HashRef Envelope Schema
  Bare hex digests are BANNED in all persistent contexts.  Every hash that
  crosses a process boundary, enters a log, is written to disk, or is used
  for cross-session comparison MUST be represented as a HashRef.

FIELD CONTRACT
  domain               : e.g. "cas.v1", "attest.v1", "cache.v1"
  algo                 : AlgoId string, e.g. "b3-256", "jcs-sha256"
  algo_version         : pinned implementation version string
  purpose              : HashPurpose enum name as str (for logging / audit)
  canonicalizer_id     : "mm-canon" (locked)
  canonicalizer_version: "1.0" (locked at this revision)
  key_id               : UUID of the SipHash / HMAC key used; None for keyless
  digest               : hex-encoded, big-endian canonical bytes

KEY_ID SEMANTICS
  For keyed hash surfaces (SipHash-2-4, HMAC-SHA256), key_id is a
  non-nullable opaque UUID string.  Two HashRefs with the same digest but
  different key_ids MUST be treated as a hard REJECT, never a match.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from pysrc.ops.hashing.contract import (
    AlgoId,
    DomainPrefix,
    HashContractViolation,
    HashPurpose,
    PersistenceTier,
    SystemInvariant,
)
from pysrc.ops.hashing.equality import verify_cache_hit as verify_cache_hit

# Allowed hex digest lengths (nibbles) per algorithm
_DIGEST_NIBBLES: dict[AlgoId, int] = {
    AlgoId.BLAKE3_256: 64,
    AlgoId.SHA256_JCS: 64,
    AlgoId.XXH3_128: 32,
    AlgoId.XXH3_64: 16,
    AlgoId.SIP24: 16,
    AlgoId.HMAC_SHA256: 64,
    AlgoId.SIMHASH_128: 32,
    AlgoId.MINHASH_128: 32,  # 128 × 16-bit values → but stored as compact 32 hex chars of control hash
    AlgoId.RABIN_63: 16,
}

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_PARSE_DOMAIN_TO_PURPOSE: dict[str, HashPurpose] = {
    DomainPrefix.CAS.value: HashPurpose.CAS_ARTIFACT_ID,
    DomainPrefix.MERKLE.value: HashPurpose.MERKLE_NODE_HASH,
    DomainPrefix.AUDIT.value: HashPurpose.AUDIT_LOG_DIGEST,
    DomainPrefix.ATTEST.value: HashPurpose.GATE_ATTESTATION,
    DomainPrefix.DIST.value: HashPurpose.DISTRIBUTED_CACHE_KEY,
    DomainPrefix.CACHE.value: HashPurpose.LOCAL_PERSISTENT_CACHE_KEY,
    DomainPrefix.SEED.value: HashPurpose.SEED_DERIVATION,
    DomainPrefix.LSH.value: HashPurpose.LSH_VECTOR_SIMHASH,
    DomainPrefix.ROLLING.value: HashPurpose.ROLLING_WINDOW_FINGERPRINT,
}


@dataclass(frozen=True)
class HashRef:
    """Immutable envelope for a single hash output.

    All field values are strings (JSON-serializable by design).  The envelope
    is the only valid representation of a hash in any persistent context.

    CONSTRUCTION
        Do NOT construct HashRef directly in application code.  Use the
        factory functions in this module:
            make_cas_ref()
            make_attest_ref()
            make_cache_ref()
            make_ephemeral_ref()

    EQUALITY SEMANTICS
        Two HashRef instances are equal iff ALL fields match.  A HashRef with
        key_id="abc" and one with key_id="def" are NOT equal even if digests
        match — see equality_check().

    SERIALIZATION
        Use to_dict() / from_dict() for JSON round-trips.  The wire format
        is the full dict; partial representations are forbidden.
    """

    domain: str
    algo: str
    digest: str
    purpose: str
    algo_version: str = "1.0"
    canonicalizer_id: str = "mm-canon"
    canonicalizer_version: str = "1.0"
    key_id: str | None = None

    # ── Post-init validation ──────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Validate all fields at construction time.

        Raises HashContractViolation if any field violates the envelope schema.
        This runs on every construction — there is no 'lazy' validation.
        """
        self._validate_domain()
        self._validate_algo()
        self._validate_digest()
        self._validate_key_id()

    def _validate_domain(self) -> None:
        allowed = {d.value for d in DomainPrefix}
        if self.domain.startswith(f"{DomainPrefix.FRAME.value}:"):
            return
        if self.domain not in allowed:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"Unknown domain={self.domain!r}. Must be one of: {sorted(allowed)}",
            )

    def _validate_algo(self) -> None:
        allowed = {a.value for a in AlgoId}
        if self.algo not in allowed:
            raise HashContractViolation(
                SystemInvariant.NO_RUNTIME_LAYOUT_DEPENDENCE,
                f"Unknown algo={self.algo!r}. Must be one of: {sorted(allowed)}",
            )

    def _validate_digest(self) -> None:
        algo_id = AlgoId(self.algo)
        expected_len = _DIGEST_NIBBLES.get(algo_id)
        if expected_len is None:
            raise HashContractViolation(
                SystemInvariant.GOLDEN_VECTOR_REQUIRED,
                f"No digest-length mapping for algo={self.algo!r}.",
            )
        if len(self.digest) != expected_len:
            raise HashContractViolation(
                SystemInvariant.CANONICAL_BIG_ENDIAN,
                f"algo={self.algo!r} requires a {expected_len}-nibble digest; "
                f"got {len(self.digest)} nibbles.",
            )
        if not _HEX_RE.match(self.digest):
            raise HashContractViolation(
                SystemInvariant.CANONICAL_BIG_ENDIAN,
                f"Digest contains non-lowercase-hex characters: {self.digest!r}",
            )

    def _validate_key_id(self) -> None:
        algo_id = AlgoId(self.algo)
        keyed_algos = {AlgoId.SIP24, AlgoId.HMAC_SHA256}
        if algo_id in keyed_algos:
            if self.key_id is None:
                raise HashContractViolation(
                    SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                    f"algo={self.algo!r} is keyed but key_id is None. "
                    "Keyed hashes MUST include a key_id UUID.",
                )
            if not _UUID_RE.match(self.key_id.lower()):
                raise HashContractViolation(
                    SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                    f"key_id={self.key_id!r} is not a valid lowercase UUID.",
                )

    # ── Formatted string representation ──────────────────────────────────────

    def to_id_string(self) -> str:
        """Return the canonical ID string: '{domain}:{algo}:{digest}'.

        This is the only format permitted for logging, storage keys, and
        cross-process identity comparison.  The key_id is NOT included in
        the ID string; it lives in the full envelope only.
        """
        return f"{self.domain}:{self.algo}:{self.digest}"

    @property
    def hex_digest(self) -> str:
        """Compatibility alias for legacy callers still using hex_digest."""
        return self.digest

    def __str__(self) -> str:
        return self.to_id_string()

    # ── Equality check with key_id enforcement ────────────────────────────────

    def equality_check(self, other: HashRef) -> bool:
        """Verify that two HashRef instances represent the same hash.

        IMPORTANT: Two HashRefs with the same digest but different key_ids
        MUST NOT be considered equal.  This method returns False in that case
        and logs a hard-rejection event.

        ADR-007 v1.1 §6.1: 'key_id mismatch = REJECT not match.'

        Raises:
            HashContractViolation: If key_ids are mismatched (not just different
                                   hash content — this is a contract violation).
        """
        if self.domain != other.domain or self.algo != other.algo:
            return False
        if self.key_id != other.key_id:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"HashRef key_id mismatch: {self.key_id!r} != {other.key_id!r}",
            )
        return self.digest == other.digest

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, str | None]:
        """Serialize to a JSON-safe dict.  All fields included, none omitted."""
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> HashRef:
        """Deserialize from a dict produced by to_dict().

        Raises:
            KeyError: If a required field is missing.
            HashContractViolation: If any field fails validation.
        """
        required_fields = {"domain", "algo", "digest", "purpose"}
        optional_fields = {
            "algo_version",
            "canonicalizer_id",
            "canonicalizer_version",
            "key_id",
        }
        allowed_fields = required_fields | optional_fields

        unknown = set(data.keys()) - allowed_fields
        if unknown:
            raise HashContractViolation(
                SystemInvariant.GOLDEN_VECTOR_REQUIRED,
                f"Unknown field(s) in HashRef dict: {sorted(unknown)}",
            )

        missing = required_fields - set(data.keys())
        if missing:
            raise KeyError(f"Missing required HashRef field(s): {sorted(missing)}")

        payload: dict[str, object] = {k: data[k] for k in allowed_fields if k in data}
        return cls(**payload)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, s: str) -> HashRef:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    @classmethod
    def parse(cls, value: str) -> HashRef:
        """Parse legacy '<domain>:<algo>:<digest>' wire strings."""
        parts = value.split(":")
        if len(parts) < 3:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"Invalid HashRef identifier {value!r}. Expected '<domain>:<algo>:<digest>'.",
            )
        domain = ":".join(parts[:-2]) if len(parts) > 3 else parts[0]
        algo = parts[-2]
        digest = parts[-1]
        if domain.startswith(f"{DomainPrefix.FRAME.value}:"):
            purpose = HashPurpose.DATAFRAME_FINGERPRINT_FAST
        else:
            purpose = _PARSE_DOMAIN_TO_PURPOSE.get(domain)
            if purpose is None:
                raise HashContractViolation(
                    SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                    f"Unsupported HashRef domain {domain!r}.",
                )
        return cls(domain=domain, algo=algo, digest=digest, purpose=purpose.name)


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────────────


def make_cas_ref(digest_hex: str) -> HashRef:
    """Create a HashRef for HashPurpose.CAS_ARTIFACT_ID.

    Args:
        digest_hex: 64 lowercase hex chars (BLAKE3-256 big-endian output).

    Returns:
        HashRef with domain="cas.v1", algo="b3-256".

    Raises:
        HashContractViolation: If digest is not 64 lowercase hex chars.
    """
    return HashRef(
        domain=DomainPrefix.CAS.value,
        algo=AlgoId.BLAKE3_256.value,
        digest=digest_hex,
        purpose=HashPurpose.CAS_ARTIFACT_ID.name,
    )


def make_attest_ref(digest_hex: str) -> HashRef:
    """Create a HashRef for HashPurpose.GATE_ATTESTATION.

    This is the attest.v1:jcs-sha256 envelope used by the gate pipeline.
    The to_gate_content_hash() function in artifact_registry/attestation.py
    converts this to the gate-facing 'sha256:<hex>' wire format.

    Args:
        digest_hex: 64 lowercase hex chars (SHA-256 of JCS bytes, big-endian).

    Returns:
        HashRef with domain="attest.v1", algo="jcs-sha256".
    """
    return HashRef(
        domain=DomainPrefix.ATTEST.value,
        algo=AlgoId.SHA256_JCS.value,
        digest=digest_hex,
        purpose=HashPurpose.GATE_ATTESTATION.name,
    )


def make_merkle_ref(digest_hex: str) -> HashRef:
    """Create a HashRef for HashPurpose.MERKLE_NODE_HASH."""
    return HashRef(
        domain=DomainPrefix.MERKLE.value,
        algo=AlgoId.BLAKE3_256.value,
        digest=digest_hex,
        purpose=HashPurpose.MERKLE_NODE_HASH.name,
    )


def make_cache_ref(
    digest_hex: str,
    *,
    purpose: HashPurpose,
    namespace: str = "strict",
) -> HashRef:
    """Create a HashRef for LOCAL_PERSISTENT_CACHE_KEY or DATAFRAME_FINGERPRINT_FAST.

    Args:
        digest_hex: 32 lowercase hex chars (XXH3-128 big-endian output).
        purpose:    Must be LOCAL_PERSISTENT_CACHE_KEY or DATAFRAME_FINGERPRINT_FAST.
        namespace:  For DATAFRAME_FINGERPRINT_FAST: "strict" (order-sensitive) or
                    "sorted" (sorted-row variant).  Must be included in the
                    domain string: e.g. "fingerprint.v1:strict".

    Raises:
        HashContractViolation: If purpose is not a local-persistent XXH3-128 purpose.
    """
    if purpose.meta.algo_id != AlgoId.XXH3_128:
        raise HashContractViolation(
            SystemInvariant.D3_BITWISE_REQUIRED,
            f"make_cache_ref requires XXH3-128 purpose; got {purpose.name}.",
        )
    if purpose.meta.persistence_tier != PersistenceTier.LOCAL_PERSISTENT:
        raise HashContractViolation(
            SystemInvariant.D3_BITWISE_REQUIRED,
            f"make_cache_ref requires local-persistent purpose; got {purpose.name}.",
        )
    if purpose == HashPurpose.DATAFRAME_FINGERPRINT_FAST:
        domain = f"{DomainPrefix.FRAME.value}:{namespace}"
    else:
        domain = DomainPrefix.CACHE.value
    return HashRef(
        domain=domain,
        algo=AlgoId.XXH3_128.value,
        digest=digest_hex,
        purpose=purpose.name,
    )


def make_siphash_ref(
    digest_hex: str,
    *,
    purpose: HashPurpose,
    key_id: str,
) -> HashRef:
    """Create a HashRef for a SipHash-2-4 keyed output.

    Args:
        digest_hex: 16 lowercase hex chars (SipHash-2-4 64-bit big-endian output).
        purpose:    Must be HASHDOS_TABLE_KEY or UNTRUSTED_INPUT_EPHEMERAL_KEY.
        key_id:     UUID of the SipHash-2-4 key used.  MANDATORY.

    Raises:
        HashContractViolation: If purpose is not a SipHash purpose or key_id is None.
    """
    if purpose.meta.algo_id != AlgoId.SIP24:
        raise HashContractViolation(
            SystemInvariant.D3_BITWISE_REQUIRED,
            f"make_siphash_ref requires SIP24 purpose; got {purpose.name}.",
        )
    if purpose.meta.persistence_tier != PersistenceTier.EPHEMERAL:
        raise HashContractViolation(
            SystemInvariant.D3_BITWISE_REQUIRED,
            f"make_siphash_ref requires ephemeral purpose; got {purpose.name}.",
        )
    return HashRef(
        domain=DomainPrefix.CACHE.value,
        algo=AlgoId.SIP24.value,
        digest=digest_hex,
        purpose=purpose.name,
        key_id=key_id,
    )


def make_hmac_ref(digest_hex: str, *, key_id: str) -> HashRef:
    """Create a HashRef for HashPurpose.SEED_DERIVATION (HMAC-SHA256).

    Args:
        digest_hex: 64 lowercase hex chars.
        key_id:     UUID identifying the master key used in derivation.
    """
    return HashRef(
        domain=DomainPrefix.SEED.value,
        algo=AlgoId.HMAC_SHA256.value,
        digest=digest_hex,
        purpose=HashPurpose.SEED_DERIVATION.name,
        key_id=key_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Equality fallback law helper
# ──────────────────────────────────────────────────────────────────────────────

# Backward-compatible re-export for older imports.  The equality module is the
# canonical home of the ADR-007 cache-hit verification law.

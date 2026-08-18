"""
py/ops/hashing/primitives/sha256_jcs.py
═══════════════════════════════════════════
SHA-256 over RFC 8785 JCS bytes for gate-facing attestation.

Covers HashPurpose:
  GATE_ATTESTATION  → domain: attest.v1:jcs-sha256

ADR-007 v1.1 §5.2 — Dual-Domain Contract
  MarketMind runs a dual-domain identity model:
    cas.v1:b3-256      → BLAKE3-256   (immutable artifact identity)
    attest.v1:jcs-sha256 → SHA-256/JCS (gate-facing verifiable attestation)

  These are NOT interchangeable.  CAS identity is content-addressable truth.
  Attestation is verifiable interoperability proof for the gate pipeline.

  Every artifact with a CAS identity MUST ALSO have an attestation hash.
  Both are stored in bundle_manifest.json as {cas, attest} per artifact role.

GATE WIRE FORMAT
  The gate pipeline consumes attestation as "sha256:<hex>" — NOT the full
  HashRef envelope.  The to_gate_content_hash() function in this module is
  the ONLY permitted conversion.  Callers must never hand-slice the prefix.

BANNED
  BLAKE3 for GATE_ATTESTATION : Gate pipeline pinned to SHA-256.
  Bare SHA-256(bytes)          : JCS canonicalization is mandatory before SHA-256.
  SHA-256 for CAS_ARTIFACT_ID : SHA-256 has no Merkle parallelism.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pysrc.ops.hashing.canonicalizer import CANON
from pysrc.ops.hashing.contract import (
    HashContractViolation,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef, make_attest_ref


class Sha256JcsHasher:
    """SHA-256 over RFC 8785 JCS bytes for gate attestation.

    Stateless.  Thread-safe.

    INVARIANT (tested in AT-003):
        For the same JCS-representable artifact, BOTH of the following hold:
          attest_hash = sha256(jcs_bytes)
          cas_hash    = blake3(jcs_bytes)
        They hash the SAME canonical bytes with different algorithms.
        This invariant must be asserted in tests/python/unit/artifact_registry/test_cas_store.pysrc.
    """

    def hash_gate_attestation(self, artifact_obj: Any) -> HashRef:
        """Compute GATE_ATTESTATION hash for a JSON-representable artifact.

        Steps:
          1. Produce JCS bytes: CANON.canonicalize_json(artifact_obj)
          2. Compute SHA-256: hashlib.sha256(jcs_bytes).hexdigest()
          3. Wrap in HashRef: make_attest_ref(digest_hex)

        The JCS bytes are the canonical preimage.  SHA-256 is a pure
        byte-stream function applied to them — no further transformation.

        Args:
            artifact_obj: JSON-serializable Python object representing the artifact.
                          Must not contain ±Inf or NaN values.

        Returns:
            HashRef with domain="attest.v1", algo="jcs-sha256".

        Raises:
            CanonicalValueRejected: If artifact_obj contains ±Inf or NaN.
            HashContractViolation:  If artifact_obj is not JSON-serializable.
        """
        jcs_bytes = CANON.canonicalize_json(artifact_obj)
        return self.hash_gate_attestation_from_jcs_bytes(jcs_bytes)

    def hash_gate_attestation_from_jcs_bytes(self, jcs_bytes: bytes) -> HashRef:
        """Compute GATE_ATTESTATION from pre-computed JCS bytes.

        Use this when JCS bytes have already been computed (e.g., they were
        also used for the CAS identity hash).  Avoids double-serialization.

        Args:
            jcs_bytes: Output of CANON.canonicalize_json().

        Returns:
            HashRef with domain="attest.v1", algo="jcs-sha256".
        """
        return make_attest_ref(hashlib.sha256(jcs_bytes).hexdigest())

    def to_gate_content_hash(self, attest_ref: HashRef) -> str:
        """Convert an attestation HashRef to the gate-facing wire format.

        The gate pipeline consumes 'sha256:<64-hex-chars>' — NOT the full
        HashRef envelope.  This function is the ONLY permitted conversion.

        ADR-007 v1.1 §5.2: 'Callers must never hand-slice the prefix.'

        Args:
            attest_ref: HashRef with domain="attest.v1", algo="jcs-sha256".

        Returns:
            String in format "sha256:<64-hex-chars>".

        Raises:
            HashContractViolation: If attest_ref.domain != "attest.v1" or
                                   attest_ref.algo != "jcs-sha256".
        """
        if attest_ref.domain != "attest.v1":
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"to_gate_content_hash requires domain='attest.v1', got {attest_ref.domain!r}.",
            )
        if attest_ref.algo != "jcs-sha256":
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"to_gate_content_hash requires algo='jcs-sha256', got {attest_ref.algo!r}.",
            )
        return f"sha256:{attest_ref.digest}"


# ── Module-level singleton ────────────────────────────────────────────────────

SHA256_JCS: Sha256JcsHasher = Sha256JcsHasher()

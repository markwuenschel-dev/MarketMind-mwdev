from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pysrc.artifact_registry._atomic import atomic_write_bytes, atomic_write_json
from pysrc.ops.hashing import (
    BLAKE3,
    SHA256_JCS,
    HashContractViolation,
    HashRef,
    canonicalize_json_bytes,
)


@dataclass(frozen=True, slots=True)
class HashRefs:
    """
    Structured hash references for a stored artifact.

    Attributes:
        cas: Primary CAS identity (cas.v1:b3-256:<hex>).
        attest: Optional gate/attestation hash (attest.v1:jcs-sha256:<hex>).
        size: Size in bytes of the stored canonical blob.
        media_type: Optional media type descriptor (e.g., "application/json").
        canonical_bytes: Exact bytes used for hashing and persistence.
    """

    cas: HashRef
    attest: HashRef | None
    size: int
    media_type: str | None = None
    canonical_bytes: bytes = b""

    def to_lineage_dict(
        self,
        *,
        schema_version: str | None = None,
        determinism_tier: str | None = None,
    ) -> dict[str, str]:
        """§7.2 lineage fields derived from this reference (optional schema / tier context)."""
        from pysrc.artifact_registry.reproducibility import json_artifact_lineage_fields

        return json_artifact_lineage_fields(
            cas_id=str(self.cas),
            attest_id=str(self.attest) if self.attest is not None else None,
            schema_version=schema_version,
            determinism_tier=determinism_tier,
        )


class LocalCAS:
    """
    Local filesystem-backed CAS store.

    Blob and index writes use ``atomic_write_bytes`` / ``atomic_write_json`` (temp file + rename).

    Storage layout:
        <root>/
          cas.v1_b3-256/
            <prefix>/<rest>          # blob content, keyed by cas.v1:b3-256
          attest_index.json          # optional attest→cas alias index

    Notes:
        - All JSON artifacts are canonicalized via the repo-owned hashing
          canonicalizer surface before hashing and storage.
        - CAS identity is always BLAKE3-256 over the canonical bytes
          (cas.v1:b3-256:<hex>).
        - Attestation hash is SHA-256 over the same canonical bytes
          (attest.v1:jcs-sha256:<hex>).
    """

    CAS_DIR_NAME = "cas.v1_b3-256"
    ATTEST_INDEX_NAME = "attest_index.json"

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._cas_root = self._root / self.CAS_DIR_NAME
        self._cas_root.mkdir(parents=True, exist_ok=True)
        self._attest_index_path = self._root / self.ATTEST_INDEX_NAME
        self._attest_to_cas: dict[str, str] = self._load_attest_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put_json(
        self, obj: dict[str, Any] | list[Any], media_type: str = "application/json"
    ) -> HashRefs:
        """
        Store a JSON-serializable object in the CAS.

        - Canonicalizes via the repo-owned hashing canonicalizer surface.
        - Computes:
            cas = cas.v1:b3-256:<hex(blake3(canonical_bytes))>
            attest = attest.v1:jcs-sha256:<hex(sha256(canonical_bytes))>
        - Writes canonical bytes once per CAS key (idempotent).
        """
        canonical_bytes = canonicalize_json_bytes(obj)
        size = len(canonical_bytes)

        cas_ref = self._make_cas_ref(canonical_bytes)
        attest_ref = self._make_attest_ref(canonical_bytes)

        self._store_if_missing(cas_ref, canonical_bytes)
        self._record_attest_alias(attest_ref, cas_ref)

        return HashRefs(
            cas=cas_ref,
            attest=attest_ref,
            size=size,
            media_type=media_type,
            canonical_bytes=canonical_bytes,
        )

    def put_bytes(self, data: bytes, media_type: str, logical_name: str | None = None) -> HashRefs:
        """
        Store arbitrary bytes in the CAS.

        - CAS identity is BLAKE3-256 over raw bytes.
        - No attestation is computed by default for non-JSON blobs.
        """
        size = len(data)
        cas_ref = self._make_cas_ref(data)
        self._store_if_missing(cas_ref, data)
        # No attest by default for raw bytes
        return HashRefs(cas=cas_ref, attest=None, size=size, media_type=media_type)

    def get_bytes(self, cas: HashRef | str) -> bytes:
        """
        Retrieve bytes for a CAS identity.

        Args:
            cas: HashRef or string in form "cas.v1:b3-256:<hex>".
        """
        cas_ref = self._ensure_cas_ref(cas)
        path = self._path_for_cas(cas_ref)
        if not path.exists():
            raise FileNotFoundError(f"CAS object not found for {cas_ref}")
        return path.read_bytes()

    def materialize(self, cas: HashRef | str, path: Path) -> None:
        """
        Materialize a CAS object to the given filesystem path.
        Overwrites any existing file.
        """
        data = self.get_bytes(cas)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, data)

    def exists(self, cas: HashRef | str) -> bool:
        """Return True if the CAS blob exists."""
        cas_ref = self._ensure_cas_ref(cas)
        return self._path_for_cas(cas_ref).exists()

    def delete(self, cas: HashRef | str) -> bool:
        """Delete one unreferenced CAS object and return whether it existed.

        Reference analysis is deliberately performed by the registry before this
        primitive is called; CAS itself has no knowledge of run retention.
        """
        cas_ref = self._ensure_cas_ref(cas)
        path = self._path_for_cas(cas_ref)
        if not path.is_file():
            return False
        path.unlink()
        if not any(path.parent.iterdir()):
            path.parent.rmdir()
        return True

    def verify(self, cas: HashRef | str) -> bool:
        """
        Verify that the on-disk bytes for *cas* match its BLAKE3 digest.

        Returns:
            True if the blob exists and its bytes hash to the expected digest,
            False otherwise.
        """
        cas_ref = self._ensure_cas_ref(cas)
        path = self._path_for_cas(cas_ref)
        if not path.exists():
            return False
        try:
            data = path.read_bytes()
        except OSError:
            return False

        expected_hex = cas_ref.hex_digest
        actual_hex = BLAKE3.hash_artifact_id(data).digest
        return actual_hex == expected_hex

    def verify_or_raise(self, cas: HashRef | str) -> None:
        """
        Verify the integrity of a CAS blob or raise HashContractViolation.

        Raises:
            HashContractViolation if the blob is missing or corrupted.
        """
        ok = self.verify(cas)
        if not ok:
            cas_ref = self._ensure_cas_ref(cas)
            raise HashContractViolation(
                "immutable_cas_integrity",
                f"CAS blob verification failed for {cas_ref}",
            )

    def resolve_attest(self, attest: HashRef | str) -> HashRef | None:
        """
        Resolve an attestation hash to its CAS identity using the alias index.

        Returns:
            HashRef for CAS identity, or None if unknown.
        """
        ref = self._ensure_attest_ref(attest)
        cas_str = self._attest_to_cas.get(str(ref))
        if cas_str is None:
            return None
        return HashRef.parse(cas_str)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cas_ref(self, data: bytes) -> HashRef:
        return BLAKE3.hash_artifact_id(data)

    def _make_attest_ref(self, canonical_bytes: bytes) -> HashRef:
        return SHA256_JCS.hash_gate_attestation_from_jcs_bytes(canonical_bytes)

    def _ensure_cas_ref(self, value: HashRef | str) -> HashRef:
        if isinstance(value, HashRef):
            if value.domain != "cas.v1" or value.algo != "b3-256":
                raise HashContractViolation(
                    "cas_ref_shape",
                    f"Expected CAS hash with domain 'cas.v1' and algo 'b3-256', got {value}",
                )
            return value
        ref = HashRef.parse(value)
        if ref.domain != "cas.v1" or ref.algo != "b3-256":
            raise HashContractViolation(
                "cas_ref_shape",
                f"Expected CAS hash with domain 'cas.v1' and algo 'b3-256', got {ref}",
            )
        return ref

    def _ensure_attest_ref(self, value: HashRef | str) -> HashRef:
        if isinstance(value, HashRef):
            if value.domain != "attest.v1" or value.algo != "jcs-sha256":
                raise HashContractViolation(
                    "attest_ref_shape",
                    f"Expected attestation hash with domain 'attest.v1' and algo 'jcs-sha256', got {value}",
                )
            return value
        ref = HashRef.parse(value)
        if ref.domain != "attest.v1" or ref.algo != "jcs-sha256":
            raise HashContractViolation(
                "attest_ref_shape",
                f"Expected attestation hash with domain 'attest.v1' and algo 'jcs-sha256', got {ref}",
            )
        return ref

    def _path_for_cas(self, cas: HashRef) -> Path:
        # Use first 2 hex chars as directory prefix for fan-out.
        prefix = cas.hex_digest[:2]
        rest = cas.hex_digest[2:]
        return self._cas_root / prefix / rest

    def _store_if_missing(self, cas: HashRef, data: bytes) -> None:
        path = self._path_for_cas(cas)
        if path.exists():
            return
        # Race-safe, idempotent write: atomic_write_bytes will either create
        # the blob or overwrite it with identical bytes. Since CAS identity is
        # derived from the content, any concurrent writer for the same CAS key
        # must be writing identical bytes.
        atomic_write_bytes(path, data)

    def _load_attest_index(self) -> dict[str, str]:
        if not self._attest_index_path.exists():
            return {}
        try:
            text = self._attest_index_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                return {}
            # Only keep string→string mappings
            return {
                str(k): str(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str)
            }
        except (OSError, json.JSONDecodeError):
            # Treat malformed index as empty; callers can repopulate lazily.
            return {}

    def _record_attest_alias(self, attest: HashRef, cas: HashRef) -> None:
        key = str(attest)
        value = str(cas)
        existing = self._attest_to_cas.get(key)
        if existing == value:
            return
        if existing is not None and existing != value:
            # This should not happen under a correct CAS policy; different CAS
            # identities for the same attestation indicate a bug.
            raise HashContractViolation(
                "attest_cas_alias_unique",
                f"Attestation hash {key!r} already mapped to different CAS {existing!r}",
            )
        self._attest_to_cas[key] = value
        try:
            atomic_write_json(self._attest_index_path, self._attest_to_cas)
        except OSError:
            # Alias index is best-effort; failure to persist it should not
            # prevent CAS operations from succeeding.
            pass

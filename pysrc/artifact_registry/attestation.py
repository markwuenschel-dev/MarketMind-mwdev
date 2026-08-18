from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HashRefLike(Protocol):
    """Minimal structural contract for a persisted MarketMind hash envelope.

    The live dual-domain contract requires domain-qualified hash IDs rather than
    bare digests: immutable CAS identity uses ``cas.v1:b3-256:<hex>`` and
    gate-facing attestation uses ``attest.v1:jcs-sha256:<hex>``. This facade
    intentionally depends only on the stable envelope surface needed by the
    registry layer:

    - ``domain`` identifies the hash namespace (e.g. ``cas.v1`` or
      ``attest.v1``).
    - ``algo`` identifies the locked algorithm selection.
    - ``digest`` is the already-computed hex payload.
    - ``purpose`` is optional at runtime here because older call sites may not
      yet expose it, but the persisted envelope is expected to carry it.

    This module does not compute hashes. It only transports and validates the
    dual-domain pair already produced by the canonical hashing surfaces.
    """

    domain: str
    algo: str
    digest: str

    def to_id_string(self) -> str: ...


@runtime_checkable
class LocalCASProtocol(Protocol):
    """Structural protocol for the canonical CAS store.

    This facade is registry-facing only. It delegates all canonicalization and
    hashing to ``LocalCAS``. The expected implementation contract is:

    - JSON payloads are canonicalized to RFC 8785 JCS bytes before hashing.
    - The same canonical bytes yield both a CAS HashRef and an attestation
      HashRef under the locked dual-domain model.
    - ``put_json`` and ``put_bytes`` return a structured pair containing both
      refs plus the canonical bytes that were persisted.

    The exact concrete implementation lives in ``py/artifact_registry/cas.py``.
    """

    def put_json(self, payload: Mapping[str, Any]) -> Any: ...

    def put_bytes(self, payload: bytes, *, media_type: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ArtifactHashPair:
    """One dual-domain hash pair over a single canonical byte sequence.

    Invariant
    ---------
    ``cas`` and ``attest`` must describe the same canonical bytes with different
    locked algorithms. The registry layer must never synthesize one without the
    other for JSON artifacts that participate in gate validation or bundle
    reconstruction.
    """

    cas: HashRefLike
    attest: HashRefLike
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Bundle-manifest view of one artifact.

    ``path`` is the bundle-relative path. ``cas`` and ``attest`` are the only
    authoritative identities. ``media_type`` and ``size`` are convenience
    fields for downstream registry and bundle consumers.
    """

    path: str
    cas: HashRefLike
    attest: HashRefLike
    media_type: str
    size: int

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "cas": self.cas.to_id_string(),
            "attest": self.attest.to_id_string(),
            "media_type": self.media_type,
            "size": self.size,
        }


@dataclass(slots=True)
class ArtifactAttestor:
    """Registry-facing facade over the locked dual-domain contract.

    Responsibilities
    ----------------
    - Delegate canonicalization + hashing to ``LocalCAS``.
    - Validate that returned refs satisfy the expected domain/algo pair.
    - Convert attestation refs to the gate wire format via the one legal helper.
    - Produce manifest-friendly ``ArtifactEntry`` objects for run bundles.

    Non-responsibilities
    --------------------
    - No raw SHA-256 or BLAKE3 computation here.
    - No alternate canonicalization path.
    - No bare-hex persistence.

    This module exists specifically so the artifact registry can speak the
    already-locked hashing contract without duplicating hashing logic.
    """

    cas: LocalCASProtocol

    def attest_json(
        self,
        payload: Mapping[str, Any],
        *,
        bundle_path: str,
        media_type: str = "application/json",
    ) -> ArtifactEntry:
        result = self.cas.put_json(payload)
        pair = self._coerce_dual_hash_pair(result)
        self._validate_dual_domain_pair(pair)
        return ArtifactEntry(
            path=bundle_path,
            cas=pair.cas,
            attest=pair.attest,
            media_type=media_type,
            size=len(pair.canonical_bytes),
        )

    def attest_bytes(
        self,
        payload: bytes,
        *,
        bundle_path: str,
        media_type: str,
    ) -> ArtifactEntry:
        result = self.cas.put_bytes(payload, media_type=media_type)
        pair = self._coerce_dual_hash_pair(result)
        self._validate_dual_domain_pair(pair)
        return ArtifactEntry(
            path=bundle_path,
            cas=pair.cas,
            attest=pair.attest,
            media_type=media_type,
            size=len(pair.canonical_bytes),
        )

    def to_gate_content_hash(self, attest: HashRefLike) -> str:
        """Convert ``attest.v1:jcs-sha256:<hex>`` to ``sha256:<hex>``.

        This is the only permitted wire conversion. Callers must not hand-slice
        the prefix because doing so hides domain/algo validation and makes future
        migrations silent instead of explicit.
        """

        self._validate_attest_ref(attest)
        return f"sha256:{attest.digest}"

    def _coerce_dual_hash_pair(self, result: Any) -> ArtifactHashPair:
        """Normalize ``LocalCAS`` return shapes into ``ArtifactHashPair``.

        Supported shapes
        ----------------
        - object with ``cas``, ``attest``, ``canonical_bytes`` attributes
        - mapping with the same keys
        - object with ``hash_refs.cas`` / ``hash_refs.attest`` plus
          ``canonical_bytes``
        """

        if (
            hasattr(result, "cas")
            and hasattr(result, "attest")
            and hasattr(result, "canonical_bytes")
        ):
            if result.attest is None:
                raise TypeError("LocalCAS return value does not expose an attestation hash.")
            return ArtifactHashPair(
                cas=result.cas,
                attest=result.attest,
                canonical_bytes=bytes(result.canonical_bytes),
            )
        if isinstance(result, Mapping) and {"cas", "attest", "canonical_bytes"}.issubset(
            result.keys()
        ):
            if result["attest"] is None:
                raise TypeError("LocalCAS return value does not expose an attestation hash.")
            return ArtifactHashPair(
                cas=result["cas"],
                attest=result["attest"],
                canonical_bytes=bytes(result["canonical_bytes"]),
            )
        if hasattr(result, "hash_refs") and hasattr(result, "canonical_bytes"):
            refs = result.hash_refs
            if hasattr(refs, "cas") and hasattr(refs, "attest"):
                return ArtifactHashPair(
                    cas=refs.cas,
                    attest=refs.attest,
                    canonical_bytes=bytes(result.canonical_bytes),
                )
        raise TypeError(
            "LocalCAS return value does not expose a dual-domain hash pair. "
            "Expected {cas, attest, canonical_bytes} or an equivalent object."
        )

    def _validate_dual_domain_pair(self, pair: ArtifactHashPair) -> None:
        self._validate_cas_ref(pair.cas)
        self._validate_attest_ref(pair.attest)
        if not pair.canonical_bytes:
            raise ValueError("Canonical bytes may not be empty for persisted artifact entries.")

    @staticmethod
    def _validate_cas_ref(ref: HashRefLike) -> None:
        if ref.domain != "cas.v1":
            raise ValueError(f"Expected CAS domain 'cas.v1', got {ref.domain!r}.")
        if ref.algo != "b3-256":
            raise ValueError(f"Expected CAS algo 'b3-256', got {ref.algo!r}.")

    @staticmethod
    def _validate_attest_ref(ref: HashRefLike) -> None:
        if ref.domain != "attest.v1":
            raise ValueError(f"Expected attestation domain 'attest.v1', got {ref.domain!r}.")
        if ref.algo != "jcs-sha256":
            raise ValueError(f"Expected attestation algo 'jcs-sha256', got {ref.algo!r}.")


@dataclass(slots=True)
class BundleManifestWriter:
    """Write the self-describing ``bundle_manifest.json`` contract.

    The emitted manifest is intentionally registry-facing and reconstruction-
    friendly. Each artifact role resolves to exactly one ``ArtifactEntry`` with
    both ``cas`` and ``attest`` refs recorded. The manifest does not compute the
    bundle identifier itself; it records the policy and artifact map required by
    the CAS-backed bundle bridge.
    """

    output_root: Path
    hash_policy: Mapping[str, str] = field(
        default_factory=lambda: {"cas": "cas.v1:b3-256", "attest": "attest.v1:jcs-sha256"}
    )

    def write(
        self,
        *,
        artifacts: Mapping[str, ArtifactEntry],
        created_at: datetime | None = None,
        schema_version: str = "1.0.0",
        filename: str = "bundle_manifest.json",
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        ts = created_at or datetime.now(UTC)
        payload: dict[str, Any] = {
            "schema_version": schema_version,
            "created_at": ts.isoformat().replace("+00:00", "Z"),
            "hash_policy": dict(self.hash_policy),
            "artifacts": {
                role: entry.to_manifest_dict()
                for role, entry in sorted(artifacts.items(), key=lambda item: item[0])
            },
        }
        if extra:
            payload.update(dict(extra))
        out_path = self.output_root / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out_path


def build_artifact_entry(
    *,
    path: str,
    cas: HashRefLike,
    attest: HashRefLike,
    media_type: str,
    size: int,
) -> ArtifactEntry:
    """Small typed helper for tests and adapter code."""

    return ArtifactEntry(path=path, cas=cas, attest=attest, media_type=media_type, size=size)


def bundle_entries_from_sequence(
    entries: Sequence[tuple[str, ArtifactEntry]],
) -> dict[str, ArtifactEntry]:
    """Convert ordered ``(role, entry)`` pairs into the bundle map.

    Duplicate roles are rejected because role names are the stable manifest keys.
    """

    out: dict[str, ArtifactEntry] = {}
    for role, entry in entries:
        if role in out:
            raise ValueError(f"Duplicate artifact role {role!r} in bundle manifest input.")
        out[role] = entry
    return out

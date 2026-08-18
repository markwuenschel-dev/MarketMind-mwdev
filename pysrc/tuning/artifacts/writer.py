"""ArtifactWriter: atomically writes artifacts to the content-addressed store."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.artifacts.canonical_json import to_canonical_json
from pysrc.tuning.artifacts.metadata import ArtifactMetadata


class ArtifactWriter:
    """Writes artifact payloads to the CAS store via the registry bridge."""

    def write(
        self,
        payload: dict[str, Any],
        metadata: ArtifactMetadata,
    ) -> str:
        """Serialise payload, compute CAS hash, write atomically; return hash."""
        canonical = to_canonical_json(payload)
        import hashlib

        digest = hashlib.blake2b(canonical.encode(), digest_size=32).hexdigest()
        cas_hash = f"cas.v1:b3-256:{digest}"
        self._store(cas_hash, canonical, metadata)
        return cas_hash

    def _store(self, cas_hash: str, canonical: str, metadata: ArtifactMetadata) -> None:
        """Persist to the underlying store; subclass for disk/S3/registry."""
        raise NotImplementedError(
            "ArtifactWriter._store must be subclassed or wired to the artifact registry"
        )


__all__ = ["ArtifactWriter"]

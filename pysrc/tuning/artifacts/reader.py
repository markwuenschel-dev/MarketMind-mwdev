"""ArtifactReader: reads artifacts from the content-addressed store by CAS hash."""

from __future__ import annotations

from typing import Any


class ArtifactReader:
    """Reads artifact payloads from the CAS store by hash."""

    def read(self, cas_hash: str) -> dict[str, Any]:
        """Return the artifact payload for the given CAS hash."""
        raise NotImplementedError(
            "ArtifactReader.read must be wired to the underlying artifact registry"
        )

    def exists(self, cas_hash: str) -> bool:
        """Return True if an artifact with this hash exists in the store."""
        raise NotImplementedError(
            "ArtifactReader.exists must be wired to the underlying artifact registry"
        )


__all__ = ["ArtifactReader"]

"""ArtifactManifest: ordered record of all artifacts produced by a tuning job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ManifestEntry:
    """A single entry in the artifact manifest."""

    artifact_type: str  # e.g. "tuning.trial", "tuning.gate_result"
    cas_hash: str  # cas.v1:b3-256:<hex>
    schema: str
    created_at: datetime


@dataclass(frozen=True)
class ArtifactManifest:
    """Immutable manifest of all artifacts for one tuning job."""

    job_id: str
    spec_hash: str
    entries: tuple[ManifestEntry, ...]
    sealed_at: datetime | None = None

    def by_type(self, artifact_type: str) -> tuple[ManifestEntry, ...]:
        """Return all entries of the given artifact_type."""
        return tuple(e for e in self.entries if e.artifact_type == artifact_type)


__all__ = ["ManifestEntry", "ArtifactManifest"]

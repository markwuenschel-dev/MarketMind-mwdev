"""ArtifactMetadata: provenance and governance fields attached to every artifact."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ArtifactMetadata:
    """Provenance and governance metadata for a single artifact blob."""

    job_id: str
    spec_hash: str
    determinism_tier: str
    created_at: datetime
    schema: str
    tags: dict[str, str] = field(default_factory=dict)


__all__ = ["ArtifactMetadata"]

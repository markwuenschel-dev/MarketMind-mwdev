from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignalArtifactSchema:
    """Phase I-Db placeholder for future stat-arb artifact schema objects."""

    schema_version: str = "phase-i-db-stub"
    metadata: dict[str, Any] | None = None

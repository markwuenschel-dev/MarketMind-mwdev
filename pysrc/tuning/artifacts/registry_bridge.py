"""RegistryBridge: adapter between the tuning artifact layer and pysrc.artifact_registry."""

from __future__ import annotations


class RegistryBridge:
    """Thin adapter to the canonical pysrc.artifact_registry CAS store."""

    def write_blob(self, cas_hash: str, payload: str) -> None:
        """Write a canonical JSON blob to the artifact registry."""
        raise NotImplementedError(
            "RegistryBridge.write_blob must be wired to pysrc.artifact_registry"
        )

    def read_blob(self, cas_hash: str) -> str:
        """Read a canonical JSON blob from the artifact registry by CAS hash."""
        raise NotImplementedError(
            "RegistryBridge.read_blob must be wired to pysrc.artifact_registry"
        )

    def blob_exists(self, cas_hash: str) -> bool:
        """Return True if the CAS hash is present in the registry."""
        raise NotImplementedError(
            "RegistryBridge.blob_exists must be wired to pysrc.artifact_registry"
        )


__all__ = ["RegistryBridge"]

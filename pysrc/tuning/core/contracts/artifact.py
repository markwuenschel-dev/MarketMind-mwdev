"""ArtifactProtocol: interface for content-addressed artifact writers/readers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArtifactWriterProtocol(Protocol):
    """Writes an artifact and returns its CAS hash."""

    def write(self, payload: dict[str, Any], metadata: dict[str, str]) -> str: ...


@runtime_checkable
class ArtifactReaderProtocol(Protocol):
    """Reads an artifact by CAS hash."""

    def read(self, cas_hash: str) -> dict[str, Any]: ...

    def exists(self, cas_hash: str) -> bool: ...


__all__ = ["ArtifactWriterProtocol", "ArtifactReaderProtocol"]

from __future__ import annotations

from pathlib import Path
from typing import Any

from pysrc.artifact_registry.artifacts import read_json
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.artifact_registry.sanitization import sanitize_json_payload
from pysrc.backtesting.contracts.types import ArtifactRef


class BundleBacktestArtifactStore:
    """Backtesting artifact store backed by the canonical artifact-registry writer."""

    def __init__(self, writer: BundleWriter) -> None:
        self._writer = writer

    @property
    def bundle_dir(self) -> Path:
        return self._writer.output_dir

    def put_json(self, role: str, payload: dict[str, Any]) -> ArtifactRef:
        clean_payload = sanitize_json_payload(payload)
        if role == "stat_validity_report.json":
            self._writer.write_stat_validity_report(clean_payload)
        else:
            self._writer._write_and_register_json(role, role, clean_payload)
        return self._make_ref(role)

    def put_bytes(self, role: str, payload: bytes, media_type: str) -> ArtifactRef:
        target = self.bundle_dir / role
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._writer._cas is not None:
            hash_refs = self._writer._cas.put_bytes(payload, media_type=media_type)
            self._writer._cas.materialize(hash_refs.cas, target)
            self._writer._role_hashes[role] = hash_refs
            self._writer._role_paths[role] = role
            if self._writer._run_registry is not None and self._writer._run_id is not None:
                self._writer._run_registry.add_artifact(self._writer._run_id, role, hash_refs)
        else:
            target.write_bytes(payload)
        if role not in self._writer._written:
            self._writer._written.append(role)
        return self._make_ref(role)

    def get_json(self, ref: ArtifactRef | str) -> dict[str, Any]:
        role = ref.role if isinstance(ref, ArtifactRef) else ref
        return read_json(self.bundle_dir / role)

    def _make_ref(self, role: str) -> ArtifactRef:
        hash_refs = self._writer._role_hashes.get(role)
        if hash_refs is None:
            return ArtifactRef(role=role, path=role)
        return ArtifactRef(
            role=role,
            path=role,
            cas=str(hash_refs.cas),
            attest=str(hash_refs.attest) if hash_refs.attest is not None else None,
        )


__all__ = ["BundleBacktestArtifactStore"]

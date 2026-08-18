"""Registry-owned run allocation, typed materialization, and safe cleanup."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.cas import HashRefs, LocalCAS
from pysrc.artifact_registry.run_registry import RunRecord, RunRegistry, RunStatus
from pysrc.ops.hashing import HashContractViolation, HashRef

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ResolvedArtifact[T]:
    """A validated product coupled to registry-owned identity metadata."""

    run_id: str
    role: str
    cas: HashRef
    attest: HashRef | None
    payload: T


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Deterministic report for a dry-run or applied registry cleanup."""

    planned_run_ids: tuple[str, ...]
    deleted_run_ids: tuple[str, ...]
    retained_run_ids: tuple[str, ...]
    planned_cas: tuple[str, ...]
    deleted_cas: tuple[str, ...]
    retained_cas: tuple[str, ...]


class ArtifactRegistry:
    """The sole owner of artifact-run allocation and typed product lookup."""

    def __init__(self, root: Path | str = Path("artifacts")) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.pinned_dir = self.root / "pinned"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cas = LocalCAS(self.root / "cas")
        self.runs = RunRegistry(self.root / "registry")

    def begin_run(self, metadata: dict[str, object] | None = None) -> str:
        """Allocate the registry run and its materialization directory together."""
        run_id = self.runs.begin_run(dict(metadata or {}))
        run_dir = self.runs_dir / run_id
        for name in ("products", "reports", "diagnostics", "logs"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / "run_meta.json", {"run_id": run_id, **dict(metadata or {})})
        return run_id

    def register_json(self, run_id: str, role: str, payload: T) -> HashRefs:
        """Store a validated product and bind its identity to a registering run."""
        if not role:
            raise ValueError("artifact role must be non-empty")
        return self.register_payload(run_id, role, payload.model_dump(mode="json"))

    def register_payload(
        self, run_id: str, role: str, payload: dict[str, object] | list[object]
    ) -> HashRefs:
        """Store a JSON payload for an owner-specific output role."""
        if not role:
            raise ValueError("artifact role must be non-empty")
        hashes = self.cas.put_json(payload)
        self.runs.add_artifact(run_id, role, hashes)
        self.cas.materialize(hashes.cas, self.runs_dir / run_id / "products" / f"{role}.json")
        return hashes

    def complete_run(self, run_id: str) -> None:
        self.runs.finalize_run(run_id, RunStatus.COMPLETE)

    def fail_run(self, run_id: str) -> None:
        self.runs.finalize_run(run_id, RunStatus.FAILED)

    def resolve(self, run_id: str, role: str, model_type: type[T]) -> ResolvedArtifact[T]:
        """Resolve a role from a COMPLETE run and validate its schema and hashes."""
        record = self.runs.get_run(run_id)
        if record is None:
            visible = self.runs.get_run(run_id, include_incomplete=True, include_failed=True)
            if visible is not None:
                raise HashContractViolation(
                    "run_registry_state",
                    f"Source run {run_id!r} must be COMPLETE; found {visible.status.value}",
                )
            raise HashContractViolation("run_registry_missing_run", f"Unknown run_id {run_id!r}")
        artifact = next((item for item in record.artifacts if item.role == role), None)
        if artifact is None:
            raise HashContractViolation(
                "run_registry_missing_role", f"Run {run_id!r} has no artifact role {role!r}"
            )
        self.cas.verify_or_raise(artifact.cas)
        if artifact.attest is not None and self.cas.resolve_attest(artifact.attest) != artifact.cas:
            raise HashContractViolation(
                "attest_cas_mismatch",
                f"Attestation does not resolve to artifact CAS for role {role!r}",
            )
        try:
            raw = json.loads(self.cas.get_bytes(artifact.cas).decode("utf-8"))
            payload = model_type.model_validate(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HashContractViolation(
                "artifact_schema", f"Artifact role {role!r} failed {model_type.__name__} validation"
            ) from exc
        return ResolvedArtifact(run_id, role, artifact.cas, artifact.attest, payload)

    def list_runs(self) -> tuple[RunRecord, ...]:
        """List every lifecycle state, in deterministic creation/id order."""
        return tuple(
            sorted(
                self.runs.iter_runs(status_filter=tuple(RunStatus)),
                key=lambda item: (item.created_at, item.run_id),
            )
        )

    def cleanup(self, *, keep_latest: int = 5, apply: bool = False) -> CleanupReport:
        """Retain pinned/newest runs and delete only CAS objects no longer referenced."""
        if keep_latest < 0:
            raise ValueError("keep_latest must be non-negative")
        records = self.list_runs()
        newest = sorted(records, key=lambda item: (item.created_at, item.run_id), reverse=True)
        pinned = {record.run_id for record in records if self._is_pinned(record)}
        retained = pinned | {record.run_id for record in newest[:keep_latest]}
        planned = tuple(
            sorted(record.run_id for record in records if record.run_id not in retained)
        )
        retained_ids = tuple(sorted(retained))
        retained_cas = self._cas_for(record for record in records if record.run_id in retained)
        planned_cas = (
            self._cas_for(record for record in records if record.run_id in planned) - retained_cas
        )
        if not apply:
            return CleanupReport(
                planned,
                (),
                retained_ids,
                tuple(sorted(planned_cas)),
                (),
                tuple(sorted(retained_cas)),
            )
        for run_id in planned:
            run_dir = self.runs_dir / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
            self.runs.delete_run(run_id)
        deleted = tuple(sorted(cas for cas in planned_cas if self.cas.delete(cas)))
        return CleanupReport(
            planned,
            planned,
            retained_ids,
            tuple(sorted(planned_cas)),
            deleted,
            tuple(sorted(retained_cas)),
        )

    def _is_pinned(self, record: RunRecord) -> bool:
        return bool(record.metadata.get("pinned")) or (self.pinned_dir / record.run_id).exists()

    @staticmethod
    def _cas_for(records: Iterable[RunRecord]) -> set[str]:
        return {str(item.cas) for record in records for item in record.artifacts}


__all__ = ["ArtifactRegistry", "CleanupReport", "ResolvedArtifact"]

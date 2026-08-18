from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.cas import HashRefs
from pysrc.ops.hashing import HashContractViolation, HashRef
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)


class RunStatus(StrEnum):
    """Lifecycle state for a run."""

    REGISTERING = "REGISTERING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class RunArtifact:
    """
    Single artifact bound to a run.

    Attributes:
        role: Logical role (e.g. "plan", "env_fingerprint", "model_snapshot_manifest").
        cas: Primary CAS identity (cas.v1:b3-256:<hex>).
        attest: Optional gate attestation hash (attest.v1:jcs-sha256:<hex>).
        media_type: Optional media type string.
        size: Size in bytes of the stored blob.
    """

    role: str
    cas: HashRef
    attest: HashRef | None = None
    media_type: str | None = None
    size: int = 0


@dataclass
class RunRecord:
    """
    Persistent record for a single run.

    Fields:
        run_id: Stable identifier for the run (UUID string).
        status: Lifecycle state (REGISTERING, COMPLETE, FAILED).
        created_at: ISO-8601 UTC timestamp when the run was created.
        updated_at: ISO-8601 UTC timestamp of the last mutation.
        artifacts: List of bound artifacts.
        metadata: Opaque metadata bag (plan hash, strategy id, etc.).
    """

    run_id: str
    status: RunStatus
    created_at: str
    updated_at: str
    artifacts: list[RunArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RunRegistry:
    """
    On-disk registry for runs and their CAS-bound artifacts.

    Persistence:
        - Backed by a single JSON file under the provided root directory.
        - Safe for single-process usage; external locking is required for
          multi-process writers.

    Semantics:
        - begin_run() → creates a REGISTERING run.
        - add_artifact() → allowed only while REGISTERING.
        - finalize_run() → transitions REGISTERING → COMPLETE or FAILED.
        - Default query surface only exposes COMPLETE runs.
    """

    INDEX_FILENAME = "runs_index.json"
    TRIAL_COUNTERS_FILENAME = "trial_counters.json"

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / self.INDEX_FILENAME
        self._trial_counters_path = self._root / self.TRIAL_COUNTERS_FILENAME
        self._runs: dict[str, RunRecord] = {}
        self._trial_counters: dict[str, int] = {}
        self._load()
        self._load_trial_counters()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_run(self, metadata: dict[str, Any] | None = None) -> str:
        """Create a new run in REGISTERING state and return its run_id."""
        self._load()
        run_id = str(uuid4())
        now = _now_iso()
        record = RunRecord(
            run_id=run_id,
            status=RunStatus.REGISTERING,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._runs[run_id] = record
        self._flush()
        return run_id

    def add_artifact(self, run_id: str, role: str, hashes: HashRefs) -> None:
        """
        Bind an artifact (by CAS + attest) to a run.

        Allowed only while the run is in REGISTERING state.
        """
        self._load()
        record = self._get_run_internal(run_id)
        if record.status is not RunStatus.REGISTERING:
            raise HashContractViolation(
                "run_registry_state",
                f"Cannot add artifacts to run {run_id!r} in state {record.status.value!r}. "
                f"Only REGISTERING runs are mutable.",
            )

        # Ensure role uniqueness: update in place if role already exists.
        for art in record.artifacts:
            if art.role == role:
                art.cas = hashes.cas
                art.attest = hashes.attest
                art.media_type = hashes.media_type
                art.size = hashes.size
                record.updated_at = _now_iso()
                self._flush()
                return

        record.artifacts.append(
            RunArtifact(
                role=role,
                cas=hashes.cas,
                attest=hashes.attest,
                media_type=hashes.media_type,
                size=hashes.size,
            )
        )
        record.updated_at = _now_iso()
        self._flush()

    def finalize_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETE) -> None:
        """
        Finalize a run, transitioning REGISTERING → COMPLETE or FAILED.
        After finalization, the run becomes immutable.
        """
        self._load()
        if status not in {RunStatus.COMPLETE, RunStatus.FAILED}:
            raise HashContractViolation(
                "run_registry_state",
                f"Run can only be finalized to COMPLETE or FAILED, got {status.value!r}",
            )

        record = self._get_run_internal(run_id)
        if record.status is not RunStatus.REGISTERING:
            raise HashContractViolation(
                "run_registry_state",
                f"Cannot finalize run {run_id!r} from state {record.status.value!r}",
            )

        record.status = status
        record.updated_at = _now_iso()
        self._flush()

    def get_run(
        self,
        run_id: str,
        *,
        include_incomplete: bool = False,
        include_failed: bool = False,
    ) -> RunRecord | None:
        """
        Retrieve a run, respecting default visibility rules.

        By default:
            - Only COMPLETE runs are visible.
        With flags:
            - include_incomplete=True → allow REGISTERING.
            - include_failed=True → allow FAILED.
        """
        self._load()
        record = self._runs.get(run_id)
        if record is None:
            return None

        if record.status is RunStatus.COMPLETE:
            return record

        if record.status is RunStatus.REGISTERING and include_incomplete:
            return record

        if record.status is RunStatus.FAILED and include_failed:
            return record

        return None

    def iter_runs(self, *, status_filter: Iterable[RunStatus] | None = None) -> Iterator[RunRecord]:
        """
        Iterate over runs, with an optional status filter.

        Default:
            status_filter={RunStatus.COMPLETE}
        """
        self._load()
        allowed = {RunStatus.COMPLETE} if status_filter is None else set(status_filter)
        for record in self._runs.values():
            if record.status in allowed:
                yield record

    def record_trial(self, family: str = "global") -> int:
        """Increment and return the platform-managed trial counter for a family."""
        self._load_trial_counters()
        next_value = int(self._trial_counters.get(family, 0)) + 1
        self._trial_counters[family] = next_value
        try:
            atomic_write_json(self._trial_counters_path, self._trial_counters)
        except OSError as exc:
            LOG.warning(
                "run_registry_trial_counter_persist_failed",
                path=str(self._trial_counters_path),
                family=family,
                error=str(exc),
            )
        return next_value

    def get_trial_count(self, family: str = "global") -> int:
        """Return the current platform-managed trial count for a family."""
        self._load_trial_counters()
        return int(self._trial_counters.get(family, 0))

    def delete_run(self, run_id: str) -> None:
        """Remove a finalized run after the caller has retained its references."""
        self._load()
        record = self._get_run_internal(run_id)
        if record.status is RunStatus.REGISTERING:
            raise HashContractViolation(
                "run_registry_state",
                f"Cannot delete registering run {run_id!r}",
            )
        del self._runs[run_id]
        self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_run_internal(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise HashContractViolation("run_registry_missing_run", f"Unknown run_id {run_id!r}")
        return record

    def _load(self) -> None:
        if not self._index_path.exists():
            self._runs = {}
            return
        try:
            text = self._index_path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except (OSError, json.JSONDecodeError):
            # Treat malformed index as empty; callers can rebuild.
            self._runs = {}
            return

        runs: dict[str, RunRecord] = {}
        if not isinstance(raw, dict):
            self._runs = {}
            return

        for run_id, payload in raw.items():
            try:
                runs[run_id] = _run_from_payload(run_id, payload)
            except HashContractViolation:
                # Skip malformed entries; better to lose a run than corrupt all.
                continue
        self._runs = runs

    def _flush(self) -> None:
        serializable: dict[str, Any] = {
            run_id: _run_to_payload(record) for run_id, record in self._runs.items()
        }
        try:
            atomic_write_json(self._index_path, serializable)
        except OSError as exc:
            # Persistence errors should surface quickly in tests, but at runtime
            # callers may choose to tolerate transient failures. Do not raise
            # here to avoid partial writes corrupting the in-memory state.
            LOG.warning(
                "run_registry_flush_failed",
                path=str(self._index_path),
                error=str(exc),
            )

    def _load_trial_counters(self) -> None:
        if not self._trial_counters_path.exists():
            self._trial_counters = {}
            return
        try:
            payload = json.loads(self._trial_counters_path.read_text(encoding="utf-8"))
        except OSError as exc:
            LOG.warning(
                "run_registry_trial_counter_load_failed",
                path=str(self._trial_counters_path),
                error=str(exc),
            )
            self._trial_counters = {}
            return
        except json.JSONDecodeError as exc:
            LOG.warning(
                "run_registry_trial_counter_decode_failed",
                path=str(self._trial_counters_path),
                error=str(exc),
            )
            self._trial_counters = {}
            return
        if not isinstance(payload, dict):
            self._trial_counters = {}
            return
        self._trial_counters = {
            str(key): int(value)
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, int)
        }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_from_payload(run_id: str, payload: Any) -> RunRecord:
    if not isinstance(payload, dict):
        raise HashContractViolation("run_registry_payload", f"Malformed run payload for {run_id!r}")

    status_str = payload.get("status")
    if not isinstance(status_str, str):
        raise HashContractViolation(
            "run_registry_payload",
            f"Invalid status {status_str!r} for run {run_id!r}",
        )
    try:
        status = RunStatus(status_str)
    except Exception as e:  # noqa: BLE001
        raise HashContractViolation(
            "run_registry_payload",
            f"Invalid status {status_str!r} for run {run_id!r}",
        ) from e

    created_at = str(payload.get("created_at", ""))
    updated_at = str(payload.get("updated_at", ""))
    metadata = payload.get("metadata") or {}
    artifacts_payload = payload.get("artifacts") or []

    artifacts: list[RunArtifact] = []
    if isinstance(artifacts_payload, list):
        for art in artifacts_payload:
            if not isinstance(art, dict):
                continue
            role = str(art.get("role", ""))
            cas_str = art.get("cas")
            if not role or not isinstance(cas_str, str):
                continue
            try:
                cas_ref = HashRef.parse(cas_str)
            except HashContractViolation:
                continue

            attest_str = art.get("attest")
            attest_ref: HashRef | None
            if isinstance(attest_str, str):
                try:
                    attest_ref = HashRef.parse(attest_str)
                except HashContractViolation:
                    attest_ref = None
            else:
                attest_ref = None

            media_type = art.get("media_type")
            size = int(art.get("size") or 0)

            artifacts.append(
                RunArtifact(
                    role=role,
                    cas=cas_ref,
                    attest=attest_ref,
                    media_type=str(media_type) if media_type is not None else None,
                    size=size,
                )
            )

    return RunRecord(
        run_id=run_id,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        artifacts=artifacts,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _run_to_payload(record: RunRecord) -> dict[str, Any]:
    return {
        "status": record.status.value,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "metadata": record.metadata,
        "artifacts": [
            {
                "role": art.role,
                "cas": str(art.cas),
                "attest": str(art.attest) if art.attest is not None else None,
                "media_type": art.media_type,
                "size": art.size,
            }
            for art in record.artifacts
        ],
    }

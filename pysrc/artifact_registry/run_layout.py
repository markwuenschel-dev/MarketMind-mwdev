"""Run-scoped artifact directories with retention and pinning support."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

ARTIFACTS_ROOT = Path("artifacts")
RUNS_DIR = ARTIFACTS_ROOT / "runs"
LATEST_DIR = ARTIFACTS_ROOT / "latest"
PINNED_DIR = ARTIFACTS_ROOT / "pinned"
SCRATCH_DIR = ARTIFACTS_ROOT / "scratch"

_RUN_META = "run_meta.json"
_SMOKE_MARKERS = frozenset({"smoke", "smoke_test", "ci_smoke"})


@dataclass(frozen=True, slots=True)
class ArtifactRoots:
    root: Path
    runs: Path
    latest: Path
    pinned: Path
    scratch: Path


@dataclass
class ArtifactCleanupReport:
    deleted_run_ids: list[str] = field(default_factory=list)
    kept_run_ids: list[str] = field(default_factory=list)
    pinned_run_ids: list[str] = field(default_factory=list)
    skipped_missing_meta: list[str] = field(default_factory=list)


def resolve_artifact_roots(root: Path | str | None = None) -> ArtifactRoots:
    base = Path(root) if root is not None else ARTIFACTS_ROOT
    return ArtifactRoots(
        root=base,
        runs=base / "runs",
        latest=base / "latest",
        pinned=base / "pinned",
        scratch=base / "scratch",
    )


def allocate_run_dir(
    *,
    lane: str,
    run_id: str | None = None,
    smoke: bool = False,
    roots: ArtifactRoots | None = None,
) -> Path:
    """Create artifacts/runs/<run_id>/ with standard subdirectories."""

    resolved = roots or resolve_artifact_roots()
    run_token = (
        run_id or f"{lane}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    run_dir = resolved.runs / run_token
    for sub in ("reports", "predictions", "diagnostics", "rankings", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_token,
        "lane": lane,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "smoke": smoke,
        "pinned": False,
    }
    atomic_write_json(run_dir / _RUN_META, meta)
    resolved.latest.mkdir(parents=True, exist_ok=True)
    latest_pointer = resolved.latest / f"{lane}.json"
    atomic_write_json(
        latest_pointer,
        {"run_id": run_token, "run_dir": str(run_dir), "updated_at": meta["created_at"]},
    )
    LOG.info("artifact_run_allocated", lane=lane, run_id=run_token, path=str(run_dir))
    return run_dir


def _load_run_meta(run_dir: Path) -> dict[str, object] | None:
    meta_path = run_dir / _RUN_META
    if not meta_path.is_file():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _is_pinned(run_id: str, roots: ArtifactRoots) -> bool:
    return (roots.pinned / run_id).exists() or (roots.pinned / f"{run_id}.json").is_file()


def _is_smoke_run(meta: dict[str, object]) -> bool:
    if bool(meta.get("smoke")):
        return True
    lane = str(meta.get("lane", "")).lower()
    run_id = str(meta.get("run_id", "")).lower()
    return any(marker in lane or marker in run_id for marker in _SMOKE_MARKERS)


def cleanup_runs(
    *,
    keep_latest: int = 5,
    delete_smoke: bool = True,
    roots: ArtifactRoots | None = None,
) -> ArtifactCleanupReport:
    """Delete unpinned old runs; optionally remove smoke runs."""

    resolved = roots or resolve_artifact_roots()
    report = ArtifactCleanupReport()
    if not resolved.runs.is_dir():
        return report

    run_dirs = sorted(
        (path for path in resolved.runs.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    kept_non_smoke = 0
    for run_dir in run_dirs:
        run_id = run_dir.name
        if _is_pinned(run_id, resolved):
            report.pinned_run_ids.append(run_id)
            report.kept_run_ids.append(run_id)
            continue

        meta = _load_run_meta(run_dir)
        if meta is None:
            report.skipped_missing_meta.append(run_id)
            continue

        if delete_smoke and _is_smoke_run(meta):
            shutil.rmtree(run_dir)
            report.deleted_run_ids.append(run_id)
            LOG.info("artifact_run_deleted_smoke", run_id=run_id)
            continue

        if kept_non_smoke < keep_latest:
            kept_non_smoke += 1
            report.kept_run_ids.append(run_id)
            continue

        shutil.rmtree(run_dir)
        report.deleted_run_ids.append(run_id)
        LOG.info("artifact_run_deleted_retention", run_id=run_id)

    return report

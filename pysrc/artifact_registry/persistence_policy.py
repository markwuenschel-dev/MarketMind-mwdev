"""Artifact persistence policy for meta-router and related lanes."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.contracts.meta_router import MetaRouterConfig
from pysrc.pipeline.meta_router_products import (
    RUN_META,
    SMOKE_SUMMARY,
    durable_filenames,
    intermediate_filenames,
)

STANDARD_DURABLE_PRODUCTS = durable_filenames()
INTERMEDIATE_PRODUCTS = intermediate_filenames()


@dataclass
class ArtifactPolicyReport:
    deleted_paths: list[str] = field(default_factory=list)
    kept_paths: list[str] = field(default_factory=list)
    smoke_ephemeral: bool = False


def _relative_paths(run_dir: Path) -> list[Path]:
    return [path for path in run_dir.rglob("*") if path.is_file()]


def _path_key(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def finalize_run_artifacts(
    run_dir: Path,
    config: MetaRouterConfig,
    *,
    summary: dict[str, object] | None = None,
) -> ArtifactPolicyReport:
    """Apply smoke/standard/debug artifact retention to a completed run."""

    report = ArtifactPolicyReport(
        smoke_ephemeral=config.smoke_test and not config.keep_smoke_artifacts
    )
    if not run_dir.is_dir():
        return report

    if config.smoke_test and config.keep_smoke_artifacts:
        report.kept_paths.extend(_path_key(path, run_dir) for path in _relative_paths(run_dir))
        return report

    if config.smoke_test and not config.keep_smoke_artifacts:
        payload = summary or {"schema_version": "smoke_summary.v1", "lane": "local_meta_router"}
        atomic_write_json(run_dir / SMOKE_SUMMARY, payload)
        for path in _relative_paths(run_dir):
            key = _path_key(path, run_dir)
            if key in {RUN_META, SMOKE_SUMMARY}:
                report.kept_paths.append(key)
                continue
            path.unlink()
            report.deleted_paths.append(key)
        for subdir in sorted(run_dir.rglob("*"), reverse=True):
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
        return report

    allowed = set(STANDARD_DURABLE_PRODUCTS)
    if config.persist_intermediates:
        allowed |= INTERMEDIATE_PRODUCTS

    for path in _relative_paths(run_dir):
        key = _path_key(path, run_dir)
        basename = path.name
        if basename in allowed:
            report.kept_paths.append(key)
            continue
        if config.persist_intermediates:
            report.kept_paths.append(key)
            continue
        path.unlink()
        report.deleted_paths.append(key)

    for subdir in sorted(run_dir.rglob("*"), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

    return report


def teardown_ephemeral_run_dir(run_dir: Path) -> None:
    if run_dir.is_dir():
        shutil.rmtree(run_dir)


def is_within_run_dir(path: Path, run_dir: Path) -> bool:
    try:
        path.resolve().relative_to(run_dir.resolve())
        return True
    except ValueError:
        return False


def load_run_meta(run_dir: Path) -> dict[str, object]:
    meta_path = run_dir / RUN_META
    if not meta_path.is_file():
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}

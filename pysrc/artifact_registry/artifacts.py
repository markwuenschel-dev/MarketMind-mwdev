"""
Canonical artifact I/O helpers for the ADR-002 artifact-registry surface.

The Appendix C bundle layout remains unchanged, but helper ownership lives
under ``pysrc.artifact_registry``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON artifact from disk."""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_bundle_files(bundle_dir: Path) -> list[Path]:
    """Return all regular files directly inside *bundle_dir* (non-recursive)."""
    return sorted(path for path in bundle_dir.iterdir() if path.is_file())


def assert_bundle_complete(
    bundle_dir: Path,
    required: list[str] | None = None,
) -> None:
    """Raise FileNotFoundError listing any required files absent from *bundle_dir*."""
    if required is None:
        required = [
            "plan.json",
            "env_fingerprint.json",
            "dataset_manifest.json",
            "preprocessing_report.json",
            "splits_manifest.json",
        ]
    missing = [filename for filename in required if not (bundle_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle at {bundle_dir} is incomplete. Missing: {missing}")

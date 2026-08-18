"""Panel-owned runtime helpers retained after router retirement."""

from __future__ import annotations

import os
from pathlib import Path

from pysrc.pipeline.contracts.p2 import P2Config


def resolve_sklearn_n_jobs(config: P2Config, *, parallel_workers: int = 1) -> int:
    """Resolve safe estimator parallelism without depending on a router lane."""
    if config.sklearn_n_jobs > 0:
        return config.sklearn_n_jobs
    if parallel_workers > 1:
        return 1
    return max(1, os.cpu_count() or 1)


def resolve_supervision_path(config: P2Config) -> Path:
    """Return the configured optional supervision product path."""
    return Path(config.data_path or config.default_supervision_path)


def supervision_artifact_exists(config: P2Config) -> bool:
    """Whether the optional supervision product is available."""
    return resolve_supervision_path(config).is_file()


__all__ = [
    "resolve_sklearn_n_jobs",
    "resolve_supervision_path",
    "supervision_artifact_exists",
]

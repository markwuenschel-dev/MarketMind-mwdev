"""Worker allocation for sklearn-backed pipeline stages."""

from __future__ import annotations

import os


def resolve_sklearn_n_jobs(requested_n_jobs: int, *, parallel_workers: int) -> int:
    """Resolve sklearn workers while preventing accidental nested parallelism."""

    if requested_n_jobs > 0:
        return requested_n_jobs
    if parallel_workers > 1:
        return 1
    return max(1, os.cpu_count() or 4)


__all__ = ["resolve_sklearn_n_jobs"]

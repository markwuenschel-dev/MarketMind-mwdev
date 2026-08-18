"""Tests for sklearn worker allocation in active pipeline stages."""

from __future__ import annotations

import pytest

from pysrc.pipeline.execution.sklearn_jobs import resolve_sklearn_n_jobs


@pytest.mark.determinism("d1")
def test_explicit_sklearn_job_count_wins(deterministic_seed: int) -> None:
    _ = deterministic_seed

    assert resolve_sklearn_n_jobs(3, parallel_workers=8) == 3


@pytest.mark.determinism("d1")
def test_auto_sklearn_job_count_avoids_nested_parallelism(deterministic_seed: int) -> None:
    _ = deterministic_seed

    assert resolve_sklearn_n_jobs(0, parallel_workers=2) == 1

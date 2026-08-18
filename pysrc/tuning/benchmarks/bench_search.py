"""Benchmark: measure throughput of the deterministic uniform sampler."""

from __future__ import annotations

import time

from pysrc.tuning.core.search.sampler import sample_uniform
from pysrc.tuning.core.search.space import SearchSpace
from pysrc.tuning.core.specs.search_space_spec import DimensionSpec, SearchSpaceSpec


def _make_space(n_dims: int) -> SearchSpace:
    dims = tuple(DimensionSpec(name=f"x{i}", kind="real", low=0.0, high=1.0) for i in range(n_dims))
    spec = SearchSpaceSpec(
        name="bench",
        version="1.0.0",
        model_type="linear",
        spec_hash="cas.v1:b3-256:" + "0" * 64,
        dimensions=dims,
        fixed={},
    )
    return SearchSpace.from_spec(spec)


def bench_sample_uniform(
    n_dims: int = 10,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Measure time to draw n_samples candidates from an n_dims-dimensional space."""
    space = _make_space(n_dims)
    start = time.perf_counter()
    sample_uniform(space, seed, n_samples)
    elapsed = time.perf_counter() - start
    return {
        "n_dims": float(n_dims),
        "n_samples": float(n_samples),
        "elapsed_s": elapsed,
        "samples_per_sec": n_samples / elapsed,
    }


__all__ = ["bench_sample_uniform"]

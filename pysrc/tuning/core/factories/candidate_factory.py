"""CandidateFactory: enumerate candidate hyperparameter sets from a SearchSpaceSpec."""

from __future__ import annotations

from pysrc.tuning.core.specs.search_space_spec import SearchSpaceSpec


def enumerate_grid(space: SearchSpaceSpec, seed: int) -> list[dict[str, object]]:
    """Return a random sample of the search space; real sampling delegated to core/search/sampler.py."""
    raise NotImplementedError(
        f"enumerate_grid for space '{space.name}' must be implemented via a registered sampler"
    )


__all__ = ["enumerate_grid"]

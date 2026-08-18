"""Transfer learning: extract search priors from past experiment memory."""

from __future__ import annotations

from pysrc.tuning.core.meta_learning.experiment_memory import ExperimentMemory
from pysrc.tuning.core.meta_learning.priors import HParamPrior, SearchPrior


def build_transfer_prior(
    memory: ExperimentMemory,
    space_hash: str,
    dim_names: tuple[str, ...],
) -> SearchPrior | None:
    """Construct a SearchPrior from past experiments; returns None if no history exists."""
    past = memory.by_space(space_hash)
    if not past:
        return None
    priors = tuple(HParamPrior(name=n, mean=0.0, std=1.0) for n in dim_names)
    return SearchPrior(space_hash=space_hash, priors=priors)


__all__ = ["build_transfer_prior"]

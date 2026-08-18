"""Prior distributions over hyperparameters informed by past experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HParamPrior:
    """Gaussian prior over a single hyperparameter dimension."""

    name: str
    mean: float
    std: float
    weight: float = 1.0


@dataclass(frozen=True)
class SearchPrior:
    """Collection of per-dimension priors for warm-starting search."""

    space_hash: str
    priors: tuple[HParamPrior, ...]


def uniform_prior(name: str, low: float, high: float) -> HParamPrior:
    """Return a Gaussian prior centred at the midpoint of [low, high]."""
    return HParamPrior(name=name, mean=(low + high) / 2.0, std=(high - low) / 6.0)


__all__ = ["HParamPrior", "SearchPrior", "uniform_prior"]

from __future__ import annotations

from .config import PAIRS_DEFAULT, HedgeEstimator, PairsConfig
from .pairs import StatArbPairsStrategy

# Backwards-compatible alias for legacy imports.
StatArbPairs = StatArbPairsStrategy

__all__ = [
    "StatArbPairsStrategy",
    "StatArbPairs",
    "PairsConfig",
    "PAIRS_DEFAULT",
    "HedgeEstimator",
]

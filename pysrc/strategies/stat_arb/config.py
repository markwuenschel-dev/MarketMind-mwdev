from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HedgeEstimator(StrEnum):
    OLS = "ols"
    KALMAN = "kalman"


@dataclass(frozen=True)
class PairsConfig:
    """Public config surface for stat-arb pairs.

    Phase I-D remains pairs-only and cointegration-based; do not add
    triplet or generalized-dimension fields here.
    """

    method: str = "cointegration"

    # Signal thresholds and holding horizon
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_hold_days: int = 60

    # Windows for beta, half-life, and spread z-score
    beta_window: int = 60
    half_life_window: int = 60
    zscore_window: int = 60

    # Half-life validity band (in bars)
    min_half_life: float = 1.0
    max_half_life: float = 252.0

    # Hedge-ratio estimator (Phase I-D executes OLS only)
    hedge_estimator: HedgeEstimator = HedgeEstimator.OLS


PAIRS_DEFAULT = PairsConfig()

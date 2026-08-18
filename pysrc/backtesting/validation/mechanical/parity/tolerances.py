from __future__ import annotations

from dataclasses import dataclass

from pysrc.backtesting.contracts.plan import DeterminismTier


@dataclass(frozen=True)
class ToleranceSpec:
    atol: float
    rtol: float


_TOLERANCES = {
    DeterminismTier.D0: ToleranceSpec(atol=0.0, rtol=0.0),
    DeterminismTier.D1: ToleranceSpec(atol=1e-9, rtol=1e-9),
    DeterminismTier.D2: ToleranceSpec(atol=1e-6, rtol=1e-6),
    DeterminismTier.D3: ToleranceSpec(atol=1e-4, rtol=1e-4),
}


def tolerance_for_tier(tier: DeterminismTier) -> ToleranceSpec:
    return _TOLERANCES[tier]

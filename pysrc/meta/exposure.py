"""Exposure scaling from volatility and disagreement state."""

from __future__ import annotations

import numpy as np


def exposure_scale_from_state(
    *,
    volatility_state: str | float | None,
    model_disagreement: float,
    base_scale: float = 1.0,
    disagreement_penalty: float = 0.25,
) -> float:
    """Reduce exposure when disagreement is high or volatility is elevated."""

    scale = float(base_scale)
    if isinstance(volatility_state, str):
        if volatility_state == "high":
            scale *= 0.5
        elif volatility_state == "mid":
            scale *= 0.75
    disagreement = max(0.0, float(model_disagreement))
    scale *= max(0.1, 1.0 - disagreement_penalty * min(disagreement, 1.0))
    return float(np.clip(scale, 0.0, 1.0))


__all__ = ["exposure_scale_from_state"]

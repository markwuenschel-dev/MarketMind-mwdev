from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from pysrc.backtesting.contracts.types import PitMeta


@dataclasses.dataclass(frozen=True)
class AlphaIR:
    signal: pd.Series
    information_coefficient: float | None
    realized_vol: pd.Series | None
    task_embedding: np.ndarray
    pit_provenance: PitMeta | None
    variant: str
    diagnostics: dict[str, Any]

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from pysrc.backtesting.contracts.types import PitMeta
from pysrc.strategies.momentum.alpha_ir import AlphaIR

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_alpha_ir_is_frozen_dataclass() -> None:
    alpha_ir = AlphaIR(
        signal=pd.Series([0.1, -0.1], index=["A", "B"]),
        information_coefficient=0.25,
        realized_vol=None,
        task_embedding=np.zeros(64, dtype=np.float32),
        pit_provenance=PitMeta(as_of="2024-01-01"),
        variant="xsec",
        diagnostics={"n_assets": 2},
    )
    assert alpha_ir.task_embedding.shape == (64,)
    assert alpha_ir.task_embedding.dtype == np.float32
    with pytest.raises(FrozenInstanceError):
        alpha_ir.variant = "tsmom"  # type: ignore[misc]

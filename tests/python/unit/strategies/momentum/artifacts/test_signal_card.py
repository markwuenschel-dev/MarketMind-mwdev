from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysrc.backtesting.contracts.types import PitMeta
from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.artifacts.signal_card import RunMeta, build_signal_card_payload
from pysrc.strategies.momentum.exceptions import SerializationError

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _alpha_ir(*, pit_provenance: PitMeta | None) -> AlphaIR:
    return AlphaIR(
        signal=pd.Series(
            [float("nan"), 0.25, float("inf")],
            index=["MSFT", "AAPL", "GOOG"],
        ),
        information_coefficient=None,
        realized_vol=None,
        task_embedding=np.zeros(64, dtype=np.float32),
        pit_provenance=pit_provenance,
        variant="xsec",
        diagnostics={"n_assets": 3},
    )


def test_signal_card_orders_positions_and_nulls_non_finite_weights() -> None:
    payload = build_signal_card_payload(
        _alpha_ir(pit_provenance=PitMeta(as_of="2024-01-01")),
        RunMeta(run_id="run-1"),
    )
    assert [item["asset_id"] for item in payload["weighted_positions"]] == ["AAPL", "GOOG", "MSFT"]
    assert payload["weighted_positions"][1]["weight"] is None
    assert payload["weighted_positions"][2]["weight"] is None
    assert payload["content_hash"].startswith("attest.v1:jcs-sha256:")


def test_signal_card_requires_pit_provenance() -> None:
    with pytest.raises(SerializationError, match="pit_provenance"):
        build_signal_card_payload(_alpha_ir(pit_provenance=None), RunMeta(run_id="run-1"))

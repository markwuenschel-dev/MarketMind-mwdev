from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from pysrc.pipeline.stages.market_data.sources.market_data import MarketDataManager
from pysrc.pipeline.stages.market_data.sources.sip_adjusted_panel import (
    SipAdjustedPanelSource,
    load_sip_adjusted_panel_frame,
)


def _tiny_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "instrument": ["AAA", "AAA", "AAA"],
            "adj_close": [100.0, 101.0, 102.0],
            "raw_close": [100.0, 101.0, 102.0],
        }
    )


@pytest.mark.determinism("d1")
def test_sip_source_registered() -> None:
    mgr = MarketDataManager(config={"default_source": "sip_adjusted_panel"})
    src = mgr._resolve("sip_adjusted_panel")
    assert isinstance(src, SipAdjustedPanelSource)


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_sip_source_filters_symbol_and_dates() -> None:
    with patch(
        "pysrc.pipeline.stages.market_data.sources.sip_adjusted_panel.load_sip_adjusted_panel_frame",
        return_value=_tiny_panel(),
    ):
        src = SipAdjustedPanelSource({"path": "ignored"})
        frame = await src.get_historical("AAA", "2024-01-02", "2024-01-03", eager=True)
    assert isinstance(frame, pl.DataFrame)
    assert len(frame) == 2


@pytest.mark.determinism("d1")
def test_load_sip_panel_frame_uses_horizon_one_base_panel(tmp_path: Path) -> None:
    panel = _tiny_panel()
    panel.to_parquet(tmp_path / "sample.parquet", index=False)
    captured: dict[str, int] = {}

    def _capture_base(panel_in: pd.DataFrame, config, **kwargs: object) -> pd.DataFrame:
        _ = panel_in
        _ = kwargs
        captured["horizon"] = int(config.horizon)
        return panel

    with (
        patch(
            "pysrc.pipeline.stages.market_data.sources.sip_adjusted_panel.load_sip_adjusted_panel",
            return_value=panel,
        ),
        patch(
            "pysrc.pipeline.stages.market_data.sources.sip_adjusted_panel.build_sip_base_panel",
            side_effect=_capture_base,
        ),
    ):
        load_sip_adjusted_panel_frame({"path": tmp_path, "apply_base_panel": True})
    assert captured.get("horizon") == 1

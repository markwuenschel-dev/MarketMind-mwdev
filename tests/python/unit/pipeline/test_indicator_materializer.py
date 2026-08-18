from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysrc.pipeline.materializers.indicator_panel import materialize_full_indicator_panel
from pysrc.pipeline.stages.preprocessing.indicators.engine import IndicatorEngine
from pysrc.pipeline.stages.preprocessing.indicators.schema import (
    REQUIRED_PROVIDER_INPUT_COLUMNS,
    W3B_INDICATOR_IDS,
)


def _synthetic_base_panel(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "instrument": ["AAA"] * rows,
            "adj_open": 100.0,
            "adj_high": 101.0,
            "adj_low": 99.0,
            "adj_close": 100.5,
            "adj_volume": 1_000_000.0,
            "raw_close": 100.5,
            "raw_volume": 1_000_000.0,
            "forward_return_horizon": 0.001,
        }
    )
    for column in REQUIRED_PROVIDER_INPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = 1.0
    return frame


@pytest.mark.determinism("d1")
def test_indicator_engine_compute_and_load_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    engine = IndicatorEngine()
    base = _synthetic_base_panel()
    computed = engine.compute(base, workers=1, copy_input=False)
    assert computed.indicator_columns
    assert set(computed.indicator_columns).issubset(set(W3B_INDICATOR_IDS))

    panel_path = tmp_path / "panel.parquet"
    computed.features.to_parquet(panel_path, index=False)
    loaded = engine.load(panel_path)
    assert loaded.indicator_columns == computed.indicator_columns


@pytest.mark.determinism("d1")
def test_materialize_full_indicator_panel_writes_product(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    processed_root = tmp_path / "processed"
    result = materialize_full_indicator_panel(
        {
            "enabled": True,
            "processed_data_root": str(processed_root),
            "workers": 1,
        },
        base_panel=_synthetic_base_panel(),
    )
    out_path = Path(result["path"])
    assert out_path.is_file()
    assert (processed_root / "manifest.json").is_file()
    frame = pd.read_parquet(out_path)
    assert "interval" in frame.columns
    assert "rsi_14" in frame.columns
    assert "forward_return_horizon" in frame.columns


@pytest.mark.determinism("d1")
def test_materialized_panel_resolves_panel_train_target(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    from pysrc.pipeline.contracts.p2 import P2Config
    from pysrc.pipeline.panel.panel_targets import resolve_panel_target_column

    processed_root = tmp_path / "processed"
    materialize_full_indicator_panel(
        {
            "enabled": True,
            "processed_data_root": str(processed_root),
            "workers": 1,
        },
        base_panel=_synthetic_base_panel(),
    )
    frame = pd.read_parquet(processed_root / "full_indicator_feature_panel" / "panel.parquet")
    column = resolve_panel_target_column(frame, P2Config())
    assert column == "forward_return_horizon"

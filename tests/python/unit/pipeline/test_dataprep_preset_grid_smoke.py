"""Production dataprep smoke: preprocessor preset/grid path on synthetic input_df."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysrc.pipeline.dataprep_runtime import DataPrepOrchestrator


def _ohlcv_panel(rows: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * rows,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + (pd.Series(range(rows)) * 0.01),
            "volume": 1_000_000,
        }
    )


@pytest.mark.determinism("d2")
def test_dataprep_spec_inline_production_graph_smoke(tmp_path: Path) -> None:
    """spec_inline exercises the production preprocessor graph without preset files."""
    run_cfg = {
        "execution": {"lazy": False, "backend": "polars"},
        "cache": {"version_tag": "production-smoke"},
        "data": {"input_df": _ohlcv_panel()},
        "pipeline": {
            "cleaning": {
                "combos": [
                    {"name": "default", "when": {"frequency": "daily"}, "steps": [], "order": {}}
                ],
                "use": "default",
            },
            "spec_inline": {
                "ops": [
                    {"kind": "scaling.robust", "input_col": "close", "out_col": "close_robust"}
                ],
            },
        },
    }
    orch = DataPrepOrchestrator(run_cfg=run_cfg)
    manifest = orch.run()
    assert manifest["status"] == "success"
    assert "hashes" in manifest
    assert manifest["hashes"].get("processed_hash")


@pytest.mark.determinism("d2")
def test_dataprep_preset_grid_path_rejects_missing_schema(tmp_path: Path) -> None:
    """Preset/grid branch fails loudly when schema keys are set but files are absent."""
    run_cfg = {
        "execution": {"lazy": False, "backend": "polars"},
        "cache": {"version_tag": "production-smoke"},
        "data": {"input_df": _ohlcv_panel()},
        "pipeline": {
            "cleaning": {
                "combos": [
                    {"name": "default", "when": {"frequency": "daily"}, "steps": [], "order": {}}
                ],
                "use": "default",
            },
            "preprocessor_preset": "default",
            "preprocessor_grid": "small",
            "preprocessor_schema_path": str(tmp_path / "missing_preprocessors.yaml"),
        },
    }
    orch = DataPrepOrchestrator(run_cfg=run_cfg)
    with pytest.raises(Exception):
        orch.run()

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysrc.pipeline.dataprep_runtime import DataPrepOrchestrator
from pysrc.pipeline.stages.preprocessing.indicators.schema import REQUIRED_PROVIDER_INPUT_COLUMNS
from tests.python.unit.pipeline.test_indicator_materializer import _synthetic_base_panel


def _research_run_cfg(processed_root: Path, base_panel: pd.DataFrame) -> dict:
    return {
        "execution": {"lazy": False, "backend": "polars"},
        "cache": {"version_tag": "research-test"},
        "data": {"input_df": base_panel},
        "pipeline": {
            "cleaning": {
                "governance_mode": "governed",
                "determinism_tier": "d1",
                "use": "default",
                "combos": [
                    {"name": "default", "when": {"frequency": "daily"}, "steps": [], "order": {}}
                ],
            },
            "preprocessing": {
                "steps": [{"type": "indicator_engine", "params": {"workers": 1}}],
            },
        },
        "outputs": {
            "indicator_panel": {
                "enabled": True,
                "processed_data_root": str(processed_root),
            },
        },
    }


@pytest.mark.determinism("d1")
def test_dataprep_research_fetch_clean_indicator_engine_materialize(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    processed_root = tmp_path / "processed"
    base = _synthetic_base_panel()
    for column in REQUIRED_PROVIDER_INPUT_COLUMNS:
        if column not in base.columns:
            base[column] = 1.0

    orch = DataPrepOrchestrator(run_cfg=_research_run_cfg(processed_root, base))
    manifest = orch.run()

    assert manifest["status"] == "success"
    assert "indicator_panel" in manifest
    out_path = Path(manifest["indicator_panel"]["path"])
    assert out_path.is_file()
    assert (processed_root / "manifest.json").is_file()

    frame = pd.read_parquet(out_path)
    assert "rsi_14" in frame.columns
    assert "interval" in frame.columns
    assert "forward_return_horizon" in frame.columns

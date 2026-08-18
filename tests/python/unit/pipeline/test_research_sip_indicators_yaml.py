"""Integration: research_sip_indicators.yaml wires indicator_engine preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pysrc.pipeline.dataprep_runtime import DataPrepOrchestrator
from pysrc.pipeline.stages.preprocessing.indicators.schema import (
    REQUIRED_PROVIDER_INPUT_COLUMNS,
    W3B_INDICATOR_IDS,
)
from tests.python.unit.pipeline.test_indicator_materializer import _synthetic_base_panel

_RESEARCH_YAML = (
    Path(__file__).resolve().parents[4]
    / "pysrc"
    / "pipeline"
    / "pipeline_config"
    / "research_sip_indicators.yaml"
)


@pytest.mark.determinism("d1")
def test_research_sip_indicators_yaml_indicator_engine_materialize(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    assert _RESEARCH_YAML.is_file(), f"missing config {_RESEARCH_YAML}"

    run_cfg = yaml.safe_load(_RESEARCH_YAML.read_text(encoding="utf-8"))
    processed_root = tmp_path / "processed"
    base = _synthetic_base_panel()
    for column in REQUIRED_PROVIDER_INPUT_COLUMNS:
        if column not in base.columns:
            base[column] = 1.0

    run_cfg.setdefault("execution", {})["backend"] = "polars"
    run_cfg.setdefault("cache", {})["version_tag"] = "research-yaml-test"
    run_cfg.setdefault("data", {})["input_df"] = base
    run_cfg.setdefault("outputs", {}).setdefault("indicator_panel", {})["processed_data_root"] = (
        str(processed_root)
    )
    # Synthetic panel uses adj_* columns; skip governed OHLCV schema step for this test.
    run_cfg.setdefault("pipeline", {}).setdefault("cleaning", {})["combos"] = [
        {"name": "default", "when": {"frequency": "daily"}, "steps": [], "order": {}},
    ]

    manifest = DataPrepOrchestrator(run_cfg=run_cfg).run()
    assert manifest["status"] == "success"
    assert "indicator_panel" in manifest

    out_path = Path(manifest["indicator_panel"]["path"])
    frame = pd.read_parquet(out_path)
    assert set(W3B_INDICATOR_IDS).issubset(set(frame.columns))
    assert "interval" in frame.columns
    assert frame.duplicated(subset=["date", "instrument", "interval"]).sum() == 0

    sidecar_manifest = processed_root / "manifest.json"
    assert sidecar_manifest.is_file()

"""Macro state panel fixture tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from pysrc.models.ensemble import emit_macro_state_panel, validate_macro_state_panel
from pysrc.pipeline.products import load_macro_state_panel_fixture
from pysrc.pipeline.stages.cleaning.core.fred_macro_provider import FredMacroGovernedProvider
from pysrc.pipeline.stages.cleaning.features.macro import MacroFeatureParams

_RESEARCH_MACRO_YAML = (
    Path(__file__).resolve().parents[4]
    / "research"
    / "p2"
    / "configs"
    / "research_macro_indicators.yaml"
)


@pytest.mark.determinism("d1")
def test_macro_state_panel_fixture_columns(deterministic_seed: int) -> None:
    _ = deterministic_seed
    frame = load_macro_state_panel_fixture(n_days=3)
    validate_macro_state_panel(frame)
    assert len(frame) == 3


@pytest.mark.determinism("d1")
def test_ensemble_emit_macro_state_panel(deterministic_seed: int) -> None:
    _ = deterministic_seed
    frame = emit_macro_state_panel(n_days=2)
    validate_macro_state_panel(frame)
    assert len(frame) == 2


@pytest.mark.determinism("d1")
def test_fred_macro_governed_provider_emits_pit_lineage(deterministic_seed: int) -> None:
    _ = deterministic_seed
    provider = FredMacroGovernedProvider()
    panel = pl.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "instrument": ["SPY", "SPY", "SPY"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    params = MacroFeatureParams(output_columns=("macro_dff", "macro_vix_chg"))
    governed = provider.materialize(panel, context=None, params=params)
    assert governed.frame.height == panel.height
    assert "macro_dff" in governed.frame.columns
    assert governed.lineage.get("pit_identity") == "fred_research_fixture.v1"


@pytest.mark.determinism("d1")
def test_research_macro_indicators_yaml_wires_macro_step(deterministic_seed: int) -> None:
    _ = deterministic_seed
    assert _RESEARCH_MACRO_YAML.is_file(), f"missing config {_RESEARCH_MACRO_YAML}"
    cfg = yaml.safe_load(_RESEARCH_MACRO_YAML.read_text(encoding="utf-8"))
    steps = cfg["pipeline"]["cleaning"]["combos"][0]["steps"]
    step_types = {step["step_type"] for step in steps}
    assert "feature.macro" in step_types
    assert cfg["pipeline"]["cleaning"]["governance_mode"] == "governed"

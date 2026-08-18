"""P2 research config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysrc.models.registry import resolve_model_family
from pysrc.pipeline.p2_config_loader import (
    RESEARCH_P2_CONFIGS,
    load_p2_config,
    validate_model_families,
)

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.determinism("d1")
def test_p2_configs_load_from_research_directory() -> None:
    for name in ("panel_baselines.yaml", "router_matrix.yaml", "local_meta_router.yaml"):
        path = RESEARCH_P2_CONFIGS / name
        config = load_p2_config(path)
        assert config["program"] == "p2"
        assert isinstance(config["experiment"], str)


@pytest.mark.determinism("d1")
def test_invalid_model_family_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model_family"):
        resolve_model_family("not_a_real_model")


@pytest.mark.determinism("d1")
def test_panel_baselines_resolves_registered_models() -> None:
    config = load_p2_config(RESEARCH_P2_CONFIGS / "panel_baselines.yaml")
    families = validate_model_families(config)
    assert "ridge" in families
    assert "random_forest" in families

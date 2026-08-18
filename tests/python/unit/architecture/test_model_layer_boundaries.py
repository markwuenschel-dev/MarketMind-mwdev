"""Architecture tests for consolidated model layer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.python.unit.architecture.import_boundary import imports_matching

ROOT = Path(__file__).resolve().parents[4]
PYSRC = ROOT / "pysrc"


@pytest.mark.determinism("d1")
def test_pysrc_ml_package_removed() -> None:
    assert not (PYSRC / "ml").exists(), "pysrc/ml should be absorbed into pysrc/models"


@pytest.mark.determinism("d1")
def test_no_imports_from_removed_ml_package() -> None:
    violations = imports_matching(PYSRC, prefix="pysrc.ml")
    assert violations == [], f"Found stale pysrc.ml imports: {violations[:5]}"


@pytest.mark.determinism("d1")
def test_executable_panel_families_include_expanded_matrix() -> None:
    from pysrc.models.registry import EXECUTABLE_MODEL_FAMILIES

    required = {
        "ridge",
        "elastic_net",
        "bayesian_ridge",
        "random_forest",
        "extra_trees",
        "pcr",
        "pls",
        "quantile_regression",
        "mlp",
    }
    assert required.issubset(EXECUTABLE_MODEL_FAMILIES)
    if importlib.util.find_spec("xgboost") is not None:
        assert "xgboost" in EXECUTABLE_MODEL_FAMILIES


@pytest.mark.determinism("d1")
def test_planned_model_family_fails_at_resolution() -> None:
    from pysrc.models.registry import resolve_model_family

    with pytest.raises(ValueError, match="planned but not yet executable"):
        resolve_model_family("lstm")

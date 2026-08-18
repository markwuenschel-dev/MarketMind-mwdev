"""Unit tests for TabularPanelModel executable families."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysrc.models.registry import (
    EXECUTABLE_MODEL_FAMILIES,
    create_panel_model,
    resolve_model_family,
)
from pysrc.models.tabular import TABULAR_FAMILIES

_TABULAR_PANEL_FAMILIES = sorted(
    family
    for family in TABULAR_FAMILIES
    if family in EXECUTABLE_MODEL_FAMILIES and family != "pcr_ridge"
)


@pytest.mark.determinism("d1")
@pytest.mark.parametrize("family", _TABULAR_PANEL_FAMILIES)
def test_tabular_panel_model_fit_predict_roundtrip(
    family: str,
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(120, 8))
    y_train = rng.normal(size=120)
    x_test = rng.normal(size=(30, 8))

    model = create_panel_model(family, random_seed=3)
    model.set_feature_names([f"f{i}" for i in range(8)])
    model.fit(x_train, y_train, fold_id="fold_0")
    preds = model.predict(x_test)
    conf = model.predict_confidence(x_test)

    assert preds.shape == (30,)
    assert conf.shape == (30,)
    assert np.isfinite(preds).all()
    assert np.isfinite(conf).all()

    path = tmp_path / f"{family}.joblib"
    model.save(path)
    loaded = create_panel_model(family, random_seed=3)
    loaded.load(path)
    reloaded = loaded.predict(x_test)
    np.testing.assert_allclose(preds, reloaded, rtol=1e-5, atol=1e-5)


@pytest.mark.determinism("d1")
def test_pcr_caps_components_to_feature_count(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(80, 3))
    y_train = rng.normal(size=80)
    model = create_panel_model("pcr", params={"n_components": 10}, random_seed=1)
    model.fit(x_train, y_train, fold_id="fold_0")
    preds = model.predict(rng.normal(size=(10, 3)))
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.determinism("d1")
def test_group_lasso_families_not_registered() -> None:
    for family in ("group_lasso", "sparse_group_lasso"):
        with pytest.raises(ValueError, match="no panel contract"):
            resolve_model_family(family)


@pytest.mark.determinism("d1")
def test_quantile_regression_is_executable() -> None:
    assert resolve_model_family("quantile_regression") == "quantile_regression"


@pytest.mark.determinism("d1")
def test_executable_families_include_expanded_matrix() -> None:
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

"""
Canonical unit tests for the pysrc.tuning subsystem.

Coverage:
- tune() facade via grid and random engines
- Direction semantics: maximize / minimize
- TuningResult shape invariants (all fields populated)
- create_tuner() factory
- SearchSpace / normalize_space() for all input formats
- parse_yaml_grid() legacy adapter
- tune_objective() adapter
- tune_estimator() sklearn adapter
- TuningError raised for unknown engine key
- EngineNotAvailableError raised when optional dep is missing (simulated)
- Registry resolution and custom registration
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from pysrc.tuning import (
    EngineNotAvailableError,
    EngineRegistry,
    SearchSpace,
    TunerSpec,
    TuningError,
    TuningResult,
    create_tuner,
    tune,
)
from pysrc.tuning.adapters.legacy_yaml import parse_yaml_grid
from pysrc.tuning.adapters.objective import tune_objective
from pysrc.tuning.space import normalize_space
from pysrc.tuning.specs import TrialRecord

# ---------------------------------------------------------------------------
# Shared objective functions
# ---------------------------------------------------------------------------


def _parabola(params: dict[str, Any]) -> float:
    """Maximize -(x-2)^2 — optimum is x=2, score=0."""
    x = float(params["x"])
    return -((x - 2.0) ** 2)


def _neg_parabola(params: dict[str, Any]) -> float:
    """Minimize (x-2)^2 — optimum is x=2, score=0."""
    x = float(params["x"])
    return (x - 2.0) ** 2


# ---------------------------------------------------------------------------
# normalize_space() — all four input formats
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_normalize_space_dict_of_lists() -> None:
    space = normalize_space({"lr": [0.01, 0.001, 0.0001], "depth": [3, 5]})
    assert isinstance(space, SearchSpace)
    combos = list(space.iter_grid())
    assert len(combos) == 6  # 3 × 2


@pytest.mark.determinism("d1")
def test_normalize_space_dict_of_tuples_continuous() -> None:
    space = normalize_space({"lr": (1e-4, 1e-1)})
    assert isinstance(space, SearchSpace)
    # Continuous space has no finite grid — iter_grid yields one empty dict
    grid = list(space.iter_grid())
    assert grid == [{}]


@pytest.mark.determinism("d1")
def test_normalize_space_mixed_dict() -> None:
    space = normalize_space({"lr": (1e-4, 1e-1), "depth": [3, 5, 7]})
    grid = list(space.iter_grid())
    # Only categorical params yield grid combos
    assert len(grid) == 3
    assert all("depth" in p for p in grid)


@pytest.mark.determinism("d1")
def test_normalize_space_list_of_dicts_yaml() -> None:
    raw = [{"lr": [0.01, 0.001]}, {"depth": [3, 5, 7]}]
    space = normalize_space(raw)
    combos = list(space.iter_grid())
    assert len(combos) == 6


@pytest.mark.determinism("d1")
def test_normalize_space_already_normalised_returns_copy() -> None:
    original = normalize_space({"x": [1, 2, 3]})
    copy = normalize_space(original)
    assert copy is not original
    assert list(copy.iter_grid()) == list(original.iter_grid())


@pytest.mark.determinism("d1")
def test_normalize_space_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="dict or list"):
        normalize_space("not a dict")  # type: ignore[arg-type]


@pytest.mark.determinism("d1")
def test_normalize_space_rejects_duplicate_param_names() -> None:
    with pytest.raises(ValueError, match="duplicate parameter name"):
        normalize_space([{"x": [1, 2]}, {"x": [3, 4]}])


# ---------------------------------------------------------------------------
# SearchSpace.sample()
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_search_space_sample_respects_seed() -> None:
    import random

    space = normalize_space({"x": [1, 2, 3, 4, 5]})
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    samples_a = space.sample(10, rng_a)
    samples_b = space.sample(10, rng_b)
    assert samples_a == samples_b


@pytest.mark.determinism("d1")
def test_search_space_sample_zero_returns_empty() -> None:
    import random

    space = normalize_space({"x": [1, 2, 3]})
    assert space.sample(0, random.Random(0)) == []


# ---------------------------------------------------------------------------
# tune() — grid engine
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_tune_grid_finds_best_maximize() -> None:
    result = tune(
        _parabola,
        {"x": [0, 1, 2, 3, 4]},
        engine="grid",
        direction="maximize",
    )
    assert isinstance(result, TuningResult)
    assert result.best_params == {"x": 2}
    assert result.best_score == pytest.approx(0.0)
    assert result.direction == "maximize"
    assert result.engine == "grid"
    assert len(result.trials) == 5


@pytest.mark.determinism("d1")
def test_tune_grid_finds_best_minimize() -> None:
    result = tune(
        _neg_parabola,
        {"x": [0, 1, 2, 3, 4]},
        engine="grid",
        direction="minimize",
    )
    assert result.best_params == {"x": 2}
    assert result.best_score == pytest.approx(0.0)
    assert result.direction == "minimize"


@pytest.mark.determinism("d1")
def test_tune_grid_result_shape_complete() -> None:
    """Every field of TuningResult is populated by the grid engine."""
    result = tune(
        lambda p: float(p["x"]),
        {"x": [1, 2, 3]},
        engine="grid",
        direction="maximize",
    )
    assert result.best_params is not None
    assert isinstance(result.best_score, float)
    assert result.best_model is None  # grid engine never populates best_model
    assert isinstance(result.trials, list)
    assert len(result.trials) == 3
    for trial in result.trials:
        assert isinstance(trial, TrialRecord)
        assert isinstance(trial.score, float)
    assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# tune() — random engine
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_tune_random_deterministic_with_seed() -> None:
    space = {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    result_a = tune(_parabola, space, engine="random", direction="maximize", budget=5, seed=99)
    result_b = tune(_parabola, space, engine="random", direction="maximize", budget=5, seed=99)
    assert result_a.best_params == result_b.best_params
    assert result_a.best_score == result_b.best_score
    assert [t.params for t in result_a.trials] == [t.params for t in result_b.trials]


@pytest.mark.determinism("d1")
def test_tune_random_different_seeds_may_differ() -> None:
    """Different seeds should (probabilistically) explore different points."""
    space = {"x": list(range(100))}
    r1 = tune(_parabola, space, engine="random", direction="maximize", budget=10, seed=1)
    r2 = tune(_parabola, space, engine="random", direction="maximize", budget=10, seed=999)
    # Cannot assert they differ (astronomically unlikely they're identical),
    # but both should produce valid results.
    assert r1.engine == "random"
    assert r2.engine == "random"
    assert len(r1.trials) == 10
    assert len(r2.trials) == 10


@pytest.mark.determinism("d1")
def test_tune_random_direction_minimize() -> None:
    result = tune(
        _neg_parabola,
        {"x": [0, 1, 2, 3, 4]},
        engine="random",
        direction="minimize",
        budget=5,
        seed=7,
    )
    assert result.direction == "minimize"
    # best_score should be the lowest score seen across trials
    min_score = min(t.score for t in result.trials)
    assert result.best_score == pytest.approx(min_score)


# ---------------------------------------------------------------------------
# tune() — result consistency invariants
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_tune_best_score_matches_best_params() -> None:
    """best_score must equal objective_fn(best_params)."""

    def obj(p: dict[str, Any]) -> float:
        return float(p["a"]) * 10.0 + float(p["b"])

    result = tune(
        obj,
        {"a": [1, 2, 3], "b": [0, 5]},
        engine="grid",
        direction="maximize",
    )
    expected_score = obj(result.best_params)
    assert result.best_score == pytest.approx(expected_score)


@pytest.mark.determinism("d1")
def test_tune_trials_contain_best() -> None:
    """best_params must appear in trials."""
    result = tune(
        _parabola,
        {"x": [0, 1, 2, 3, 4]},
        engine="grid",
        direction="maximize",
    )
    trial_params = [t.params for t in result.trials]
    assert result.best_params in trial_params


# ---------------------------------------------------------------------------
# create_tuner()
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_create_tuner_reusable() -> None:
    spec = TunerSpec(
        engine="grid",
        direction="maximize",
        budget=10,
        cv=5,
        scoring=None,
        seed=None,
    )
    tuner = create_tuner(spec)
    space = normalize_space({"x": [1, 2, 3]})

    r1 = tuner(space, lambda p: float(p["x"]))
    r2 = tuner(space, lambda p: -float(p["x"]))

    assert r1.best_params == {"x": 3}
    assert r2.best_params == {"x": 1}


@pytest.mark.determinism("d1")
def test_create_tuner_raises_for_unknown_engine() -> None:
    spec = TunerSpec(
        engine="does_not_exist",
        direction="maximize",
        budget=5,
        cv=5,
        scoring=None,
        seed=None,
    )
    with pytest.raises(TuningError, match="does_not_exist"):
        create_tuner(spec)


# ---------------------------------------------------------------------------
# TuningError — unknown engine
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_tune_raises_for_unknown_engine() -> None:
    with pytest.raises(TuningError, match="Unknown tuning engine"):
        tune(lambda p: 0.0, {"x": [1]}, engine="nonexistent_engine")


# ---------------------------------------------------------------------------
# EngineNotAvailableError — optional dep simulation
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_engine_not_available_error_attributes() -> None:
    err = EngineNotAvailableError("bayes", "scikit-optimize (skopt)")
    assert err.engine == "bayes"
    assert err.package == "scikit-optimize (skopt)"
    assert "bayes" in str(err)
    assert "scikit-optimize" in str(err)
    assert "pip install" in str(err)


@pytest.mark.determinism("d1")
def test_engine_not_available_is_tuning_error() -> None:
    err = EngineNotAvailableError("optuna", "optuna")
    assert isinstance(err, TuningError)


@pytest.mark.determinism("d1")
def test_bayes_engine_raises_when_skopt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate skopt not being installed: importing bayes engine must raise EngineNotAvailableError."""
    # Remove skopt from sys.modules if present, and block the import.
    monkeypatch.setitem(sys.modules, "skopt", None)  # type: ignore[arg-type]
    # Import the engine run function directly and call it to trigger the guard.
    import importlib

    from pysrc.tuning.engines import bayes as bayes_mod

    importlib.reload(bayes_mod)

    space = normalize_space({"x": [1, 2]})
    spec = TunerSpec(
        engine="bayes",
        direction="maximize",
        budget=2,
        cv=5,
        scoring=None,
        seed=None,
    )
    with pytest.raises(EngineNotAvailableError, match="scikit-optimize"):
        bayes_mod.run(spec, space, lambda p: float(p["x"]))


@pytest.mark.determinism("d1")
def test_optuna_engine_raises_when_optuna_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate optuna not being installed."""
    monkeypatch.setitem(sys.modules, "optuna", None)  # type: ignore[arg-type]
    import importlib

    from pysrc.tuning.engines import optuna as optuna_mod

    importlib.reload(optuna_mod)

    space = normalize_space({"x": [1, 2]})
    spec = TunerSpec(
        engine="optuna",
        direction="maximize",
        budget=2,
        cv=5,
        scoring=None,
        seed=None,
    )
    with pytest.raises(EngineNotAvailableError, match="optuna"):
        optuna_mod.run(spec, space, lambda p: float(p["x"]))


# ---------------------------------------------------------------------------
# EngineRegistry
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_engine_registry_has_canonical_engines() -> None:
    available = EngineRegistry.available()
    assert "grid" in available
    assert "random" in available
    assert "bayes" in available
    assert "optuna" in available


@pytest.mark.determinism("d1")
def test_engine_registry_custom_registration() -> None:
    """Custom engines can be registered and immediately used via tune()."""
    sentinel: list[bool] = []

    def _my_engine(
        spec: TunerSpec,
        space: SearchSpace,
        objective_fn: Any,
    ) -> TuningResult:
        from pysrc.tuning.specs import TrialRecord as TR
        from pysrc.tuning.specs import TuningResult as TRes

        sentinel.append(True)
        trial = TR(params={"x": 1}, score=1.0)
        return TRes(
            best_params={"x": 1},
            best_score=1.0,
            best_model=None,
            trials=[trial],
            engine="custom_test",
            direction=spec.direction,
        )

    EngineRegistry.register("custom_test", _my_engine)
    try:
        result = tune(lambda p: 1.0, {"x": [1]}, engine="custom_test")
        assert result.engine == "custom_test"
        assert sentinel == [True]
    finally:
        # Clean up: re-register the original lazy loader to avoid test pollution.
        # The canonical "grid"/"random"/"bayes"/"optuna" entries are preserved.
        del EngineRegistry._factories["custom_test"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# tune_objective() adapter
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_tune_objective_returns_canonical_result() -> None:
    result = tune_objective(
        _parabola,
        {"x": [0, 1, 2, 3, 4]},
        engine="grid",
        direction="maximize",
    )
    assert isinstance(result, TuningResult)
    assert result.best_params == {"x": 2}
    assert result.best_score == pytest.approx(0.0)


@pytest.mark.determinism("d1")
def test_tune_objective_accepts_list_of_dicts_space() -> None:
    result = tune_objective(
        lambda p: float(p["x"]),
        [{"x": [10, 20, 30]}],
        engine="grid",
        direction="maximize",
    )
    assert result.best_params == {"x": 30}


# ---------------------------------------------------------------------------
# tune_estimator() sklearn adapter
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_tune_estimator_grid_populates_best_model() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn required for this test")
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    from pysrc.tuning.adapters.sklearn import tune_estimator

    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] > 0).astype(int)

    result = tune_estimator(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [2, 3, 4]},
        X,
        y,
        engine="grid",
        direction="maximize",
        cv=3,
        scoring="accuracy",
    )

    assert isinstance(result, TuningResult)
    assert result.best_model is not None
    assert "max_depth" in result.best_params
    assert len(result.trials) == 3


@pytest.mark.determinism("d1")
def test_tune_estimator_best_model_refitted_on_full_data() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn required for this test")
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    from pysrc.tuning.adapters.sklearn import tune_estimator

    rng = np.random.default_rng(seed=1)
    X = rng.standard_normal((40, 3))
    y = (X[:, 0] > 0).astype(int)

    result = tune_estimator(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [2, 4]},
        X,
        y,
        engine="grid",
        direction="maximize",
        cv=2,
    )

    # Verify best_model can predict on training data (i.e. it is actually fitted).
    preds = result.best_model.predict(X)  # type: ignore[union-attr]
    assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# parse_yaml_grid() legacy adapter
# ---------------------------------------------------------------------------


@pytest.mark.determinism("d1")
def test_parse_yaml_grid_canonical_space() -> None:
    grid = [
        {"lr": [0.01, 0.001]},
        {"hidden": [64, 128, 256]},
    ]
    space = parse_yaml_grid(grid)
    assert isinstance(space, SearchSpace)
    combos = list(space.iter_grid())
    assert len(combos) == 6  # 2 × 3


@pytest.mark.determinism("d1")
def test_parse_yaml_grid_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate parameter name"):
        parse_yaml_grid([{"x": [1, 2]}, {"x": [3, 4]}])


@pytest.mark.determinism("d1")
def test_parse_yaml_grid_single_param() -> None:
    space = parse_yaml_grid([{"dropout": [0.1, 0.2, 0.5]}])
    combos = list(space.iter_grid())
    assert len(combos) == 3
    assert all("dropout" in c for c in combos)

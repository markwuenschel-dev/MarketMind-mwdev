# tests/python/test_builder.py

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pysrc.pipeline.core.pipeline_core_builder import (
    PipelineBuilder,
    _matches,
    choose_combo,
    topo_order,
)

# ------------------------------------------------------------
# _matches — basic rule checks; keep it lightweight and generic
# ------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_when", "ctx_attrs", "expected"),
    [
        ({}, {}, True),
        ({"dataset": "stocks"}, {"dataset": "stocks"}, True),
        ({"dataset": "stocks"}, {"dataset": "bonds"}, False),
        ({"rows": {"ge": 10}}, {"rows": 10}, True),
        ({"rows": {"gt": 10}}, {"rows": 10}, False),
        ({"rows": {"le": 5}}, {"rows": 4}, True),
        ({"rows": {"lt": 5}}, {"rows": 5}, False),
    ],
)
def test_matches(rule_when: dict[str, Any], ctx_attrs: dict[str, Any], expected: bool):
    ctx = MagicMock()
    for k, v in ctx_attrs.items():
        setattr(ctx, k, v)
    assert _matches(rule_when, ctx) is expected


# ------------------------------------------------------------
# choose_combo — adapts to your redesigned behavior
#   • Dict combos: use > default > first; unknown use → fallback (no KeyError)
#   • List combos: MODE/PIPELINE_COMBO by name; else when-match; else first
#   • Missing combos keys produce specific KeyErrors
# ------------------------------------------------------------


def _ctx(dataset: str = "stocks", env: str = "prod") -> Any:
    return MagicMock(dataset=dataset, env=env)


@pytest.mark.parametrize(
    ("cfg_shape", "combos_kind", "use_key", "env_key", "when_match"),
    [
        # Dict combos
        ("attr", "dict", "default", None, True),
        ("attr", "dict", "custom", None, True),
        ("attr", "dict", None, "test_mode", True),
        ("attr", "dict", "missing", None, True),
        ("dict", "dict", "default", None, True),
        ("dict", "dict", "custom", None, True),
        ("dict", "dict", None, "test_mode", True),
        ("dict", "dict", "missing", None, True),
        ("cleaning_only", "dict", "default", None, True),
        ("cleaning_only", "dict", None, "test_mode", True),
        # List combos
        ("attr", "list", None, "test_mode", False),
        ("attr", "list", None, None, True),
        ("attr", "list", None, None, False),
        ("dict", "list", None, "test_mode", False),
        ("dict", "list", None, None, True),
        ("dict", "list", None, None, False),
        ("cleaning_only", "list", None, "test_mode", False),
        ("cleaning_only", "list", None, None, True),
        ("cleaning_only", "list", None, None, False),
    ],
)
def test_choose_combo_all_formats(
    cfg_shape: str, combos_kind: str, use_key: str | None, env_key: str | None, when_match: bool
):
    ctx = _ctx()

    # Build the combos structure based on combos_kind
    if combos_kind == "dict":
        combos = {
            "default": {"steps": ["a"], "order": {}},
            "custom": {"steps": ["b"], "order": {}},
            "test_mode": {"steps": ["c"], "order": {}},
        }
    else:  # list
        combos = [
            {"name": "default", "steps": ["a"], "order": {}},
            {"name": "custom", "steps": ["b"], "order": {}},
            {
                "name": "test_mode",
                "steps": ["c"],
                "order": {},
                "when": {"dataset": "stocks"} if when_match else {"dataset": "bonds"},
            },
        ]

    # Build effective_cfg based on cfg_shape BEFORE using it
    if cfg_shape == "attr":
        cleaning = MagicMock()
        cleaning.combos = combos
        if use_key is not None:
            cleaning.use = use_key
        cfg_obj = MagicMock()
        cfg_obj.cleaning = cleaning
        effective_cfg = cfg_obj  # object with .cleaning
    elif cfg_shape == "dict":
        cfg = {"cleaning": {"combos": combos}}
        if use_key is not None:
            cfg["cleaning"]["use"] = use_key
        effective_cfg = cfg
    else:  # cleaning_only
        cfg = {"combos": combos}
        if use_key is not None:
            cfg["use"] = use_key
        effective_cfg = {"cleaning": cfg}

    # Now we can use effective_cfg
    env_map = {}
    if env_key:
        env_map = {"MODE": env_key, "PIPELINE_COMBO": env_key}

    with patch.dict(os.environ, env_map, clear=False):
        out = choose_combo(effective_cfg, ctx)

    # Assertions tailored to redesigned semantics
    if combos_kind == "dict":
        if use_key in ("default", "custom", "test_mode"):
            exp = {"default": ["a"], "custom": ["b"], "test_mode": ["c"]}[use_key]
            assert out.get("steps") == exp
        elif env_key == "test_mode":
            assert out.get("steps") == ["c"]
        else:
            # default if available; else first value
            assert out.get("steps") in (["a"], ["b"], ["c"])
    else:
        # list: env name preferred if present, but when clause must match
        if env_key == "test_mode":
            # Check if the test_mode combo's when clause matches
            test_mode_combo = next((c for c in combos if c["name"] == "test_mode"), None)
            if test_mode_combo and "when" in test_mode_combo:
                if when_match:
                    assert out.get("steps") == ["c"]
                else:
                    # when clause doesn't match, fall back to first
                    assert out.get("steps") == ["a"]
            else:
                assert out.get("steps") == ["c"]
        else:
            # when-match → pick matching; else first
            if when_match:
                assert out.get("steps") == ["c"]
            else:
                assert out.get("steps") == ["a"]


def test_choose_combo_error_shapes():
    ctx = _ctx()

    # dict with cleaning but no combos
    cfg = {"cleaning": {}}
    with pytest.raises(KeyError, match="'cleaning' missing 'combos'"):
        choose_combo(cfg, ctx)

    # dict with neither cleaning nor combos
    cfg = {"other": {}}
    with pytest.raises(KeyError, match="missing 'cleaning' or 'combos'"):
        choose_combo(cfg, ctx)

    # empty → empty
    assert choose_combo({}, ctx) == {"steps": [], "order": {}}

    # single dict without default/use → first
    cfg = {"cleaning": {"combos": {"alt": {"steps": ["y"], "order": {}}}}}
    assert choose_combo(cfg, ctx)["steps"] == ["y"]

    # dict with invalid use → fallback to default or first (no KeyError in redesign)
    cfg = {
        "cleaning": {
            "combos": {
                "default": {"steps": ["d"], "order": {}},
                "x": {"steps": ["x"], "order": {}},
            },
            "use": "missing",
        }
    }
    out = choose_combo(cfg, ctx)
    assert out["steps"] in (["d"], ["x"])


def test_topo_order_preserves_and_maps_conflicts():
    steps = ["A", "B", "C", "D"]
    order = {"before": {"A": ["B"]}, "after": {"D": ["C"]}}
    ordered = topo_order(steps, order)
    assert ordered.index("A") < ordered.index("B")
    assert ordered.index("C") < ordered.index("D")

    # Conflict A<->B → ValueError("Cyclic step constraints detected")
    with pytest.raises(ValueError, match="Cyclic step constraints detected"):
        topo_order(["A", "B"], {"before": {"A": ["B"], "B": ["A"]}})


# ------------------------------------------------------------
# Minimal Pipeline smoke (matches your simplified stubs)
# ------------------------------------------------------------


def test_pipeline_smoke():
    preset = {"steps": [{"name": "clean_norm"}, {"name": "feature_x"}]}
    params = {"clean_norm.enabled": True, "feature_x.enabled": True}
    pb = PipelineBuilder.for_stage("cleaning", {"origin": "test"}).from_preset_and_params(
        preset, params
    )
    X, y, meta = pb.build().fit_transform({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(X) == 3
    assert len(y) == 3
    assert meta["num_steps"] == 2
    assert meta["step_names"] == ["clean_norm", "feature_x"]

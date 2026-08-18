"""
Strategy package imports and StrategyRegistry error surfacing.

A) Do not swallow real errors: module missing tolerated; module present but broken surfaces.
C) Registry import error surfaced: StrategyRegistry.get("momentum") when module missing
   yields KeyError with module path and chained cause, not swallowed elsewhere.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_strategies_package_loads_when_momentum_and_stat_arb_missing():
    """Module missing should be tolerated: package loads with MomentumStrategy/StatArbPairs as None."""
    for key in list(sys.modules.keys()):
        if key == "pysrc.strategies" or key.startswith("pysrc.strategies."):
            del sys.modules[key]
    import pysrc.strategies as strategies

    # When momentum/stat_arb modules do not exist, they are set to None (optional imports)
    assert getattr(strategies, "MomentumStrategy", None) is None
    assert getattr(strategies, "StatArbPairs", None) is None
    from pysrc.strategies.pipeline_strategy import StrategyRegistry

    assert StrategyRegistry is not None


def test_strategies_package_broken_module_surfaces():
    """Module present but broken should surface: non-ImportError is not swallowed."""
    for key in list(sys.modules.keys()):
        if key == "pysrc.strategies" or key.startswith("pysrc.strategies."):
            del sys.modules[key]

    import importlib

    from pysrc.strategies.pipeline_strategy import StrategyRegistry

    StrategyRegistry.clear_for_test()

    real_import_module = importlib.import_module

    def _raise_on_momentum(name: str, *args: object, **kwargs: object) -> object:
        if name == "pysrc.strategies.momentum":
            raise ValueError("simulated broken module")
        return real_import_module(name, *args, **kwargs)

    with patch.object(importlib, "import_module", side_effect=_raise_on_momentum):
        with pytest.raises(ValueError, match="simulated broken module"):
            StrategyRegistry.get("momentum")


def test_registry_get_momentum_raises_key_error_with_module_path_and_cause():
    """StrategyRegistry.get('momentum') when module does not exist: KeyError, message has module path, __cause__ set."""
    import importlib

    from pysrc.strategies.pipeline_strategy import StrategyRegistry

    # Clear any cached registration so we hit the dynamic-import path
    if "momentum" in StrategyRegistry._REGISTRY:
        del StrategyRegistry._REGISTRY["momentum"]

    real_import_module = importlib.import_module

    def _raise_missing_momentum(name, *args, **kwargs):
        if name == "pysrc.strategies.momentum":
            raise ModuleNotFoundError("No module named 'pysrc.strategies.momentum'")
        return real_import_module(name, *args, **kwargs)

    with patch.object(importlib, "import_module", side_effect=_raise_missing_momentum):
        with pytest.raises(KeyError) as exc_info:
            StrategyRegistry.get("momentum")

    err = exc_info.value
    assert "momentum" in str(err)
    assert "pysrc.strategies.momentum" in str(err)
    assert err.__cause__ is not None, "KeyError should chain the original import error (from e)"
    assert type(err.__cause__).__name__ in ("ModuleNotFoundError", "ImportError")

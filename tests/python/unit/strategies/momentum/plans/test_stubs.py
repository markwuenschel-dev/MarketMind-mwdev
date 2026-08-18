from __future__ import annotations

import importlib

import pytest

from pysrc.strategies.momentum.plans.ensemble import build_plan as build_ensemble_plan
from pysrc.strategies.momentum.plans.ml import build_plan as build_ml_plan

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_phase_stub_plans_raise() -> None:
    with pytest.raises(NotImplementedError, match="Phase II/III stub"):
        build_ensemble_plan({})
    with pytest.raises(NotImplementedError, match="Phase III stub"):
        build_ml_plan({})


def test_control_package_raises_invariant_message() -> None:
    with pytest.raises(NotImplementedError, match="momentum.control is a Phase III stub"):
        importlib.import_module("pysrc.strategies.momentum.control")

from __future__ import annotations

import importlib

import pytest

from pysrc.strategies.pipeline_strategy import StrategyRegistry
from pysrc.strategies.stat_arb import PairsConfig, StatArbPairsStrategy

pytestmark = pytest.mark.determinism("d0")


def test_stat_arb_package_spine_is_import_safe_and_pairs_only() -> None:
    package = importlib.import_module("pysrc.strategies.stat_arb")

    assert package.StatArbPairsStrategy is StatArbPairsStrategy
    assert isinstance(package.PAIRS_DEFAULT, PairsConfig)
    assert StrategyRegistry.get("stat_arb_pairs") is StatArbPairsStrategy

    importlib.import_module("pysrc.strategies.stat_arb.artifacts")
    importlib.import_module("pysrc.strategies.stat_arb.artifacts.schemas")
    signal_card = importlib.import_module("pysrc.strategies.stat_arb.artifacts.signal_card")
    importlib.import_module("pysrc.strategies.stat_arb.dimensions")
    importlib.import_module("pysrc.strategies.stat_arb.dimensions.base")
    deferred_pairs = importlib.import_module("pysrc.strategies.stat_arb.dimensions.pairs")
    triplets = importlib.import_module("pysrc.strategies.stat_arb.dimensions.triplets")
    control = importlib.import_module("pysrc.strategies.stat_arb.control.controller")
    multi_ou = importlib.import_module("pysrc.strategies.stat_arb.control.multi_ou")
    formation = importlib.import_module("pysrc.strategies.stat_arb.formation")

    with pytest.raises(NotImplementedError, match="Phase I-Db stub"):
        deferred_pairs.DeferredPairsDimensionAdapter()
    with pytest.raises(NotImplementedError, match="Phase II stub"):
        triplets.TripletsDimension()
    with pytest.raises(NotImplementedError, match="Phase I-Db stub"):
        control.build_controller()
    with pytest.raises(NotImplementedError, match="Phase II stub"):
        multi_ou.solve_multi_ou_control()
    with pytest.raises(NotImplementedError, match="Phase I-Db stub"):
        formation.form_stat_arb_basket()
    with pytest.raises(NotImplementedError, match="Phase I-Db stub"):
        signal_card.render_signal_card()

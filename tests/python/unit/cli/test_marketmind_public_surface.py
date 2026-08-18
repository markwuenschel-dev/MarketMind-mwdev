"""Public command-surface tests for the unified MarketMind CLI."""

from __future__ import annotations

import pytest

from pysrc.cli.marketmind import cli


@pytest.mark.determinism("d1")
def test_legacy_lane_groups_are_not_public(deterministic_seed: int) -> None:
    _ = deterministic_seed

    legacy_groups = {"phase2", "router", "w3", "indicators", "w4", "supervision"}

    assert legacy_groups.isdisjoint(cli.commands)


@pytest.mark.determinism("d1")
def test_active_stage_groups_remain_public(deterministic_seed: int) -> None:
    _ = deterministic_seed

    assert {
        "dataprep",
        "panel",
        "candidate-portfolios",
        "meta-router",
        "backtest",
        "artifacts",
    } <= set(cli.commands)

"""Public product command-surface contract."""

from __future__ import annotations

import pytest

from pysrc.cli.marketmind import cli


@pytest.mark.determinism("d1")
def test_product_domains_match_the_supported_execution_graph(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed

    assert set(cli.commands) == {
        "run",
        "dataprep",
        "panel",
        "tuning",
        "strategies",
        "candidate-portfolios",
        "meta-router",
        "backtest",
        "artifacts",
        "capabilities",
        "config",
    }

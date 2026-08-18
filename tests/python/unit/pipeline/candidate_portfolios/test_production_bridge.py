"""Gate 5 production bridge tests."""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.pipeline.candidate_portfolios.production_bridge import (
    build_direct_candidate_products,
    build_production_candidate_products,
    build_production_parity_report,
    predictions_to_trade_intent_envelope,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def _synthetic_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in ("fold_0", "fold_1"):
        for date in ("2024-01-02", "2024-01-03"):
            for instrument, pred in [("AAA", 0.3), ("BBB", 0.2), ("CCC", -0.1)]:
                rows.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "model_id": "model_a",
                        "prediction": pred,
                        "fold_id": fold_id,
                        "split": "test",
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ("2024-01-02", "2024-01-03"):
        for instrument, ret in [("AAA", 0.02), ("BBB", 0.01), ("CCC", -0.005)]:
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "forward_return_1d": ret,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.determinism("d1")
def test_trade_intent_envelope_carries_fold_and_split(deterministic_seed: int) -> None:
    _ = deterministic_seed
    predictions = _synthetic_predictions()
    intents = predictions_to_trade_intent_envelope(
        predictions,
        strategy_id="prediction_threshold_model_a",
        source_product_id="model_prediction_panel",
        lineage={"run_id": "test-run"},
        top_k=2,
        source_model_id="model_a",
    )

    assert not intents.empty
    assert {"fold_id", "split"}.issubset(intents.columns)
    assert set(intents["fold_id"].astype(str)) == {"fold_0", "fold_1"}


@pytest.mark.determinism("d1")
def test_strategy_and_direct_paths_have_finite_returns(deterministic_seed: int) -> None:
    _ = deterministic_seed
    predictions = _synthetic_predictions()
    panel = _synthetic_panel()
    spec = PortfolioSpec(top_k=2, single_name_cap=0.5, cost_bps=0.0, capacity_constraints=False)

    intents = predictions_to_trade_intent_envelope(
        predictions,
        strategy_id="prediction_threshold_model_a",
        source_product_id="model_prediction_panel",
        lineage={"run_id": "test-run"},
        top_k=2,
    )
    strategy_positions, strategy_outputs = build_production_candidate_products(
        intents,
        panel,
        spec,
    )
    direct_positions, direct_outputs = build_direct_candidate_products(
        predictions,
        panel,
        spec,
        model_id="model_a",
    )

    assert not strategy_positions.empty
    assert not direct_positions.empty
    assert strategy_outputs["net_return"].notna().all()
    assert direct_outputs["net_return"].notna().all()
    assert strategy_positions["candidate_id"].nunique() >= 1
    assert direct_positions["candidate_id"].nunique() == 1


@pytest.mark.determinism("d1")
def test_parity_report_passes_on_matching_paths(deterministic_seed: int) -> None:
    _ = deterministic_seed
    predictions = _synthetic_predictions()
    panel = _synthetic_panel()
    spec = PortfolioSpec(top_k=2, single_name_cap=0.5, cost_bps=0.0, capacity_constraints=False)
    strategy_id = "prediction_threshold_model_a"

    intents = predictions_to_trade_intent_envelope(
        predictions,
        strategy_id=strategy_id,
        source_product_id="model_prediction_panel",
        lineage={"run_id": "test-run"},
        top_k=2,
    )
    strategy_positions, strategy_outputs = build_production_candidate_products(
        intents,
        panel,
        spec,
    )
    direct_positions, direct_outputs = build_direct_candidate_products(
        predictions,
        panel,
        spec,
        model_id="model_a",
    )

    from pysrc.pipeline.candidate_portfolios.viability import _candidate_economics_by_fold

    parity = build_production_parity_report(
        direct_by_fold=_candidate_economics_by_fold(direct_positions, panel, spec),
        strategy_by_fold=_candidate_economics_by_fold(strategy_positions, panel, spec),
        model_id="model_a",
        strategy_id=strategy_id,
        sharpe_tol=0.25,
        cum_log_tol=0.25,
    )

    assert parity["parity_pass"] is True
    assert parity["by_fold"]

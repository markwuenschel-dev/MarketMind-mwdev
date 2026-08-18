"""Aggregate candidate portfolio pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pysrc.pipeline.candidate_portfolios.build_positions import predictions_to_candidate_positions
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.candidate_portfolios.trade_intent_positions import (
    trade_intents_to_candidate_positions,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def build_candidate_portfolio_products(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = predictions_to_candidate_positions(
        predictions,
        top_k=portfolio_spec.top_k,
        single_name_cap=portfolio_spec.single_name_cap,
    )
    outputs = simulate_candidate_portfolios(
        positions,
        panel,
        cost_bps=portfolio_spec.cost_bps,
        capacity_limit=1.0 if portfolio_spec.capacity_constraints else 10.0,
    )
    return positions, outputs


def build_candidate_portfolio_products_from_trade_intents(
    trade_intents: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    fold_id: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate products from the canonical strategy output boundary."""

    positions = trade_intents_to_candidate_positions(
        trade_intents,
        top_k=portfolio_spec.top_k,
        single_name_cap=portfolio_spec.single_name_cap,
        fold_id=fold_id,
        split=split,
    )
    outputs = simulate_candidate_portfolios(
        positions,
        panel,
        cost_bps=portfolio_spec.cost_bps,
        capacity_limit=1.0 if portfolio_spec.capacity_constraints else 10.0,
    )
    return positions, outputs


def write_candidate_portfolio_products(
    run_dir: Path,
    positions: pd.DataFrame,
    outputs: pd.DataFrame,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    pred_dir = run_dir / "predictions"
    diag_dir = run_dir / "diagnostics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    pos_path = pred_dir / "candidate_position_panel.parquet"
    out_path = diag_dir / "candidate_portfolio_output_panel.parquet"
    positions.to_parquet(pos_path, index=False)
    outputs.to_parquet(out_path, index=False)
    return {"candidate_position_panel": pos_path, "candidate_portfolio_output_panel": out_path}


__all__ = [
    "build_candidate_portfolio_products",
    "build_candidate_portfolio_products_from_trade_intents",
    "write_candidate_portfolio_products",
]

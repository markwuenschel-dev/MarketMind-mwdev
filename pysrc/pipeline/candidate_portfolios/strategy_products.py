"""Candidate positions assembled from strategy intent envelopes."""

from __future__ import annotations

import pandas as pd

from pysrc.contracts.meta_router import CANDIDATE_POSITION_PANEL_COLUMNS, DEFAULT_CANDIDATE_ID
from pysrc.pipeline.candidate_portfolios.trade_intent_positions import (
    trade_intents_to_candidate_positions,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def trade_intent_envelope_to_candidate_positions(
    trade_intents: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
) -> pd.DataFrame:
    """Size intents while retaining fold and split as stage-owned envelope fields."""

    required = {"fold_id", "split"}
    missing = sorted(required - set(trade_intents.columns))
    if missing:
        raise ValueError(f"trade intent envelope missing columns: {missing}")

    position_parts: list[pd.DataFrame] = []
    for (fold_id, split), group in trade_intents.groupby(["fold_id", "split"], sort=True):
        positions = trade_intents_to_candidate_positions(
            group,
            top_k=portfolio_spec.top_k,
            single_name_cap=portfolio_spec.single_name_cap,
            fold_id=str(fold_id),
            split=str(split),
        )
        if not positions.empty:
            position_parts.append(positions)

    if not position_parts:
        return pd.DataFrame(columns=list(CANDIDATE_POSITION_PANEL_COLUMNS))

    positions = pd.concat(position_parts, ignore_index=True)
    return _append_default_blend(positions, single_name_cap=portfolio_spec.single_name_cap)


def _append_default_blend(
    positions: pd.DataFrame,
    *,
    single_name_cap: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (date, fold_id, split), group in positions.groupby(["date", "fold_id", "split"], sort=True):
        tickers = sorted(group["ticker"].astype(str).unique())
        if not tickers:
            continue
        weight = min(single_name_cap, 1.0 / len(tickers))
        rows.append(
            pd.DataFrame(
                {
                    "date": [date] * len(tickers),
                    "candidate_id": [DEFAULT_CANDIDATE_ID] * len(tickers),
                    "ticker": tickers,
                    "target_weight": [weight] * len(tickers),
                    "fold_id": [fold_id] * len(tickers),
                    "split": [split] * len(tickers),
                }
            )
        )
    if not rows:
        return positions
    return pd.concat([positions, *rows], ignore_index=True)[list(CANDIDATE_POSITION_PANEL_COLUMNS)]


__all__ = ["trade_intent_envelope_to_candidate_positions"]

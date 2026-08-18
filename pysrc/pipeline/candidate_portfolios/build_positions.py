"""Build candidate position panels from model predictions."""

from __future__ import annotations

import pandas as pd

from pysrc.contracts.meta_router import (
    CANDIDATE_POSITION_PANEL_COLUMNS,
    CASH_CANDIDATE_ID,
    DEFAULT_CANDIDATE_ID,
)


def predictions_to_candidate_positions(
    predictions: pd.DataFrame,
    *,
    top_k: int = 20,
    single_name_cap: float = 0.10,
) -> pd.DataFrame:
    """Cross-sectional rank per date → equal-weight long-only positions per model."""

    required = {"date", "instrument", "model_id", "prediction", "fold_id", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"model_prediction_panel missing columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    for (_date, model_id, fold_id, split), group in predictions.groupby(
        ["date", "model_id", "fold_id", "split"], sort=True
    ):
        ranked = group.sort_values("prediction", ascending=False).head(top_k)
        n = max(1, len(ranked))
        weight = min(single_name_cap, 1.0 / n)
        part = ranked[["date", "instrument"]].copy()
        part["candidate_id"] = str(model_id)
        part["ticker"] = part["instrument"].astype(str)
        part["target_weight"] = weight
        part["fold_id"] = fold_id
        part["split"] = split
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=list(CANDIDATE_POSITION_PANEL_COLUMNS))

    positions = pd.concat(rows, ignore_index=True)
    positions = positions[list(CANDIDATE_POSITION_PANEL_COLUMNS)]

    # Default equal blend across models per date
    blend_rows: list[pd.DataFrame] = []
    for (date, fold_id, split), group in positions.groupby(["date", "fold_id", "split"], sort=True):
        tickers = group["ticker"].unique()
        if len(tickers) == 0:
            continue
        w = 1.0 / len(tickers)
        blend_rows.append(
            pd.DataFrame(
                {
                    "date": [date] * len(tickers),
                    "candidate_id": [DEFAULT_CANDIDATE_ID] * len(tickers),
                    "ticker": tickers,
                    "target_weight": [min(single_name_cap, w)] * len(tickers),
                    "fold_id": [fold_id] * len(tickers),
                    "split": [split] * len(tickers),
                }
            )
        )
    if blend_rows:
        positions = pd.concat(
            [positions, pd.concat(blend_rows, ignore_index=True)], ignore_index=True
        )

    # Cash has no positions (weight panel only for active candidates)
    return positions


def candidate_ids_from_positions(positions: pd.DataFrame) -> list[str]:
    ids = sorted(positions["candidate_id"].astype(str).unique().tolist())
    if CASH_CANDIDATE_ID not in ids:
        ids.append(CASH_CANDIDATE_ID)
    return ids


__all__ = ["candidate_ids_from_positions", "predictions_to_candidate_positions"]

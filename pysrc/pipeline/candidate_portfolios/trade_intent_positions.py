"""Candidate sizing for the canonical strategy ``TradeIntent`` product."""

from __future__ import annotations

import pandas as pd

from pysrc.contracts.meta_router import CANDIDATE_POSITION_PANEL_COLUMNS
from pysrc.contracts.trade_intent import TradeDirection


def trade_intents_to_candidate_positions(
    trade_intents: pd.DataFrame,
    *,
    top_k: int = 20,
    single_name_cap: float = 0.10,
    fold_id: str,
    split: str,
) -> pd.DataFrame:
    """Size active strategy opinions into comparable candidate position panels.

    ``TradeIntent`` remains deliberately pre-sizing. Fold and split describe
    the stage invocation, not the strategy's decision, so they are injected at
    this boundary rather than added to the shared strategy contract.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if not 0.0 < single_name_cap <= 1.0:
        raise ValueError("single_name_cap must be in (0, 1]")
    if not fold_id or not split:
        raise ValueError("fold_id and split must be non-empty")

    required = {
        "date",
        "instrument",
        "strategy_id",
        "score",
        "direction",
        "eligible",
        "abstain",
    }
    missing = required - set(trade_intents.columns)
    if missing:
        raise ValueError(f"trade_intent_panel missing columns: {sorted(missing)}")

    valid_directions = {direction.value for direction in TradeDirection}
    directions = trade_intents["direction"].astype(str)
    unknown = sorted(set(directions) - valid_directions)
    if unknown:
        raise ValueError(f"trade_intent_panel has unknown directions: {unknown}")

    active_mask = (
        trade_intents["eligible"].astype(bool)
        & ~trade_intents["abstain"].astype(bool)
        & directions.isin((TradeDirection.LONG.value, TradeDirection.SHORT.value))
    )
    active = trade_intents.loc[active_mask].copy()
    if active.empty:
        return pd.DataFrame(columns=list(CANDIDATE_POSITION_PANEL_COLUMNS))

    active["score"] = pd.to_numeric(active["score"], errors="coerce")
    if active["score"].isna().any() or active["score"].isin((float("inf"), float("-inf"))).any():
        raise ValueError("active trade intents require finite scores")

    long_scores = active.loc[active["direction"] == TradeDirection.LONG.value, "score"]
    short_scores = active.loc[active["direction"] == TradeDirection.SHORT.value, "score"]
    if (long_scores <= 0.0).any() or (short_scores >= 0.0).any():
        raise ValueError("trade intent score signs must agree with direction")

    active["_absolute_score"] = active["score"].abs()
    active = active.sort_values(
        ["date", "strategy_id", "_absolute_score", "instrument"],
        ascending=[True, True, False, True],
        kind="stable",
    )

    rows: list[pd.DataFrame] = []
    for (_, strategy_id), group in active.groupby(["date", "strategy_id"], sort=True):
        selected = group.head(top_k).copy()
        weight = min(single_name_cap, 1.0 / len(selected))
        selected["candidate_id"] = str(strategy_id)
        selected["ticker"] = selected["instrument"].astype(str)
        selected["target_weight"] = selected["direction"].map(
            {
                TradeDirection.LONG.value: weight,
                TradeDirection.SHORT.value: -weight,
            }
        )
        selected["fold_id"] = fold_id
        selected["split"] = split
        rows.append(selected)

    positions = pd.concat(rows, ignore_index=True)
    return positions[list(CANDIDATE_POSITION_PANEL_COLUMNS)]


__all__ = ["trade_intents_to_candidate_positions"]

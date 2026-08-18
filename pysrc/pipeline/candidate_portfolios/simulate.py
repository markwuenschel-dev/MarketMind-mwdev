"""Simulate candidate portfolio paths with costs and constraints."""

from __future__ import annotations

import pandas as pd

from pysrc.contracts.meta_router import (
    CANDIDATE_PORTFOLIO_OUTPUT_PANEL_COLUMNS,
    MetaRouterConfig,
)
from pysrc.portfolio.labels import compute_weight_path_series, forward_return_lookup


def simulate_candidate_portfolios(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    forward_horizon_days: int = 1,
    capacity_limit: float = 1.0,
) -> pd.DataFrame:
    """Return candidate_portfolio_output_panel aggregated by date and model."""

    config = MetaRouterConfig(
        cost_bps=cost_bps,
        forward_horizon_days=forward_horizon_days,
        smoke_test=True,
    )
    returns_lookup = forward_return_lookup(panel)
    rows: list[dict[str, object]] = []

    for candidate_id, weights in positions.groupby("candidate_id"):
        calendar = sorted(weights["date"].astype(str).unique())
        series = compute_weight_path_series(
            weights[["date", "ticker", "target_weight"]],
            returns_lookup,
            config,
            calendar=calendar,
        )
        gross_exposure = (
            weights.groupby("date")["target_weight"]
            .apply(lambda value: float(value.abs().sum()))
            .reindex(calendar, fill_value=0.0)
        )
        capacity_used = (gross_exposure / max(capacity_limit, 1e-9)).clip(0.0, 1.0)
        for date in calendar:
            row = series.loc[series["date"] == date].iloc[0]
            rows.append(
                {
                    "model_id": str(candidate_id),
                    "date": date,
                    "gross_return": float(row["gross_return"]),
                    "net_return": float(row["net_return"]),
                    "turnover": float(row["turnover"]),
                    "cost": float(row["cost_estimate"]),
                    "drawdown": float(row["drawdown"]),
                    "gross_exposure": float(gross_exposure.get(date, 0.0)),
                    "cash_weight": float(max(0.0, 1.0 - gross_exposure.get(date, 0.0))),
                    "capacity_used": float(capacity_used.get(date, 0.0)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=list(CANDIDATE_PORTFOLIO_OUTPUT_PANEL_COLUMNS))
    out = pd.DataFrame(rows)
    return out[list(CANDIDATE_PORTFOLIO_OUTPUT_PANEL_COLUMNS)]


__all__ = ["simulate_candidate_portfolios"]

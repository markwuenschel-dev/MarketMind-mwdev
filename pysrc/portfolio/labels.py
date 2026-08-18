"""Evaluate candidate portfolios with forward realized returns and a cost model.

Net utility per date (held close ``t`` over the forward horizon):

    U_t = net_t - gamma/2 * net_t^2 - lambda * turnover_t
          - eta * drawdown_increment_t - cash_hurdle

    net_t = gross_t - cost_rate * turnover_t
    gross_t = sum_i w_{i,t} * forward_return_{i,t}

The router label is ``delta_utility_vs_default = U_t^m - U_t^default``.
Forward returns are targets only; they never enter meta-state features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from pysrc.contracts.meta_router import (
    CASH_CANDIDATE_ID,
    MetaRouterConfig,
    validate_portfolio_label_frame,
)

# Forward-return column name produced by the (retired) meta_router candidate outputs
# that this portfolio-label evaluation consumes. Inlined when that lane was demolished.
FORWARD_RETURN_COLUMN = "forward_return_1d"


def forward_return_lookup(panel: pd.DataFrame) -> pd.DataFrame:
    """Slim (date, ticker, forward_return) frame from the panel slice."""

    frame = panel[["date", "instrument", FORWARD_RETURN_COLUMN]].rename(
        columns={"instrument": "ticker"}
    )
    return frame.dropna(subset=["date"]).reset_index(drop=True)


def compute_weight_path_series(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    config: MetaRouterConfig,
    *,
    calendar: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-date gross/net/utility series for an arbitrary weight path.

    ``weights`` is (date, ticker, target_weight); dates absent from it (e.g.
    days routed to cash) earn ``cash_daily_return`` and pay the cost of having
    unwound the previous book. Missing forward returns are treated as 0
    (delisting fallback) and counted in ``missing_forward_count``.
    """

    dates = sorted(set(calendar) if calendar is not None else set(forward_returns["date"]))
    date_set = {str(d) for d in dates}
    tickers = set(weights["ticker"].astype(str).unique())
    scoped_returns = forward_returns.loc[
        forward_returns["date"].astype(str).isin(date_set)
        & forward_returns["ticker"].astype(str).isin(tickers)
    ].drop_duplicates(subset=["date", "ticker"], keep="last")
    wide_weights = (
        weights.pivot_table(index="date", columns="ticker", values="target_weight", fill_value=0.0)
        .reindex(dates, fill_value=0.0)
        .sort_index()
    )
    wide_forward = (
        scoped_returns.pivot_table(index="date", columns="ticker", values=FORWARD_RETURN_COLUMN)
        .reindex(index=dates, columns=wide_weights.columns)
        .sort_index()
    )

    held = wide_weights.to_numpy(dtype=np.float64)
    forwards = wide_forward.to_numpy(dtype=np.float64)
    missing_mask = np.isnan(forwards) & (held != 0.0)
    forwards = np.nan_to_num(forwards, nan=0.0)

    gross = (held * forwards).sum(axis=1)
    previous = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    turnover = np.abs(held - previous).sum(axis=1)
    cost_rate = config.cost_bps / 1e4
    cost = cost_rate * turnover

    invested = np.abs(held).sum(axis=1) > 0.0
    net = gross - cost
    net[~invested] = config.cash_daily_return - cost[~invested]

    cumulative = np.cumsum(np.log1p(np.clip(net, -0.99, None)))
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    drawdown_increment = np.clip(np.diff(drawdown, prepend=0.0), 0.0, None)

    risk_penalty = 0.5 * config.gamma_risk * net**2
    utility = (
        net
        - risk_penalty
        - config.lambda_turnover * turnover
        - config.eta_drawdown * drawdown_increment
        - config.cash_hurdle_daily
    )

    return pd.DataFrame(
        {
            "date": dates,
            "gross_return": gross,
            "cost_estimate": cost,
            "net_return": net,
            "turnover": turnover,
            "drawdown": drawdown,
            "drawdown_increment": drawdown_increment,
            "risk_penalty": risk_penalty,
            "net_utility": utility,
            "missing_forward_count": missing_mask.sum(axis=1).astype(np.int64),
        }
    )


def _cash_series(dates: Sequence[str], config: MetaRouterConfig) -> pd.DataFrame:
    net = np.full(len(dates), config.cash_daily_return, dtype=np.float64)
    utility = net - 0.5 * config.gamma_risk * net**2 - config.cash_hurdle_daily
    return pd.DataFrame(
        {
            "date": list(dates),
            "gross_return": net,
            "cost_estimate": 0.0,
            "net_return": net,
            "turnover": 0.0,
            "drawdown": 0.0,
            "drawdown_increment": 0.0,
            "risk_penalty": 0.5 * config.gamma_risk * net**2,
            "net_utility": utility,
            "missing_forward_count": 0,
        }
    )


def build_portfolio_labels(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    config: MetaRouterConfig,
) -> pd.DataFrame:
    """Label every candidate (including cash) at date_candidate grain."""

    forward_returns = forward_return_lookup(panel)
    calendar = sorted(weights["date"].unique())

    pieces: list[pd.DataFrame] = []
    for candidate_id, group in weights.groupby("candidate_id", sort=True):
        series = compute_weight_path_series(group, forward_returns, config, calendar=calendar)
        series.insert(1, "candidate_id", candidate_id)
        pieces.append(series)

    cash = _cash_series(calendar, config)
    cash.insert(1, "candidate_id", CASH_CANDIDATE_ID)
    pieces.append(cash)

    labels = pd.concat(pieces, ignore_index=True)
    default_utility = labels.loc[
        labels["candidate_id"] == config.default_candidate_id, ["date", "net_utility"]
    ].rename(columns={"net_utility": "_default_utility"})
    if default_utility.empty:
        raise ValueError(f"Default candidate {config.default_candidate_id!r} missing from labels")
    labels = labels.merge(default_utility, on="date", how="left")
    labels["delta_utility_vs_default"] = labels["net_utility"] - labels["_default_utility"]
    labels = labels.drop(columns=["_default_utility"])
    labels = labels.drop(columns=["missing_forward_count"])
    labels = labels.sort_values(["candidate_id", "date"], kind="mergesort").reset_index(drop=True)
    validate_portfolio_label_frame(labels)
    return labels


def summarize_series(
    series: pd.DataFrame,
    *,
    dates: Sequence[str] | None = None,
    trading_days_per_year: float = 252.0,
) -> Mapping[str, float]:
    """Headline metrics for one per-date series, optionally restricted to dates."""

    frame = series
    if dates is not None:
        frame = frame.loc[frame["date"].isin(set(dates))]
    if frame.empty:
        return {"n_dates": 0.0}

    net = frame["net_return"].to_numpy(dtype=np.float64)
    utility = frame["net_utility"].to_numpy(dtype=np.float64)
    cumulative = np.cumsum(np.log1p(np.clip(net, -0.99, None)))
    drawdown = np.maximum.accumulate(cumulative) - cumulative

    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = float(net.mean() / std * np.sqrt(trading_days_per_year)) if std > 0 else 0.0
    positive = utility[utility > 0]
    top5_share = float(np.sort(positive)[-5:].sum() / positive.sum()) if positive.sum() > 0 else 0.0
    return {
        "n_dates": float(len(frame)),
        "sum_net_utility": float(utility.sum()),
        "mean_net_utility": float(utility.mean()),
        "mean_net_return": float(net.mean()),
        "net_sharpe": sharpe,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "mean_turnover": float(frame["turnover"].mean()),
        "top5_day_utility_share": top5_share,
    }


__all__ = [
    "build_portfolio_labels",
    "compute_weight_path_series",
    "forward_return_lookup",
    "summarize_series",
]

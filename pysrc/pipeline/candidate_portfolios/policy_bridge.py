"""PDR-002 policy bridge: panel predictions → local policy selector → routed economics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pysrc.contracts.meta_router import (
    CANDIDATE_POSITION_PANEL_COLUMNS,
    DEFAULT_CANDIDATE_ID,
    MetaRouterConfig,
)
from pysrc.meta.bocpd_service import BOCPDRegimeService
from pysrc.meta.local_policy_selector import (
    route_decisions,
    train_local_policy_selector,
)
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import annualized_log_rv_from_returns
from pysrc.pipeline.candidate_portfolios.build_positions import predictions_to_candidate_positions
from pysrc.pipeline.candidate_portfolios.promotion_stats import _sharpe_from_returns
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.candidate_portfolios.viability import (
    _attach_fold_id_to_outputs,
    _capacity_limit,
    _panel_slice_for_simulation,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec
from pysrc.portfolio.labels import build_portfolio_labels

_POLICY_BOCPD_CONFIG = BOCPDConfig(cold_start_burn_in=5)
_FOLD1_GAP_CLOSE_MIN = 0.3
_BOUNDARY_FLAG_CODES: dict[str, float] = {
    "cold_start": 0.0,
    "stable": 1.0,
    "transition": 2.0,
    "change_point": 3.0,
}


def _parse_decision_ts(date_value: object) -> datetime:
    text = str(date_value)
    if "T" in text:
        decision_ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        decision_ts = datetime.fromisoformat(f"{text}T00:00:00+00:00")
    if decision_ts.tzinfo is None:
        return decision_ts.replace(tzinfo=UTC)
    return decision_ts.astimezone(UTC)


def _bocpd_regime_features_from_default_returns(
    labels: pd.DataFrame,
    *,
    bocpd_config: BOCPDConfig = _POLICY_BOCPD_CONFIG,
) -> pd.DataFrame:
    """PIT BOCPD labels from equal_blend net returns (data <= date T only)."""

    default_returns = (
        labels.loc[
            labels["candidate_id"].astype(str) == DEFAULT_CANDIDATE_ID,
            ["date", "net_return"],
        ]
        .drop_duplicates(subset=["date"])
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if default_returns.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "regime_id",
                "state_change_probability",
                "state_transition_probability",
                "state_posterior_entropy",
                "state_bocpd_boundary_code",
                "state_cold_start",
            ]
        )

    dates = default_returns["date"].astype(str).tolist()
    returns = default_returns["net_return"].astype(float).to_numpy(dtype=np.float64)
    n = int(returns.size)
    log_rv = np.full(n, np.nan, dtype=np.float64)
    for idx in range(bocpd_config.vol_window - 1, n):
        log_rv[idx] = annualized_log_rv_from_returns(returns, idx, bocpd_config.vol_window)

    finite_indices = np.flatnonzero(np.isfinite(log_rv))
    if finite_indices.size == 0:
        return pd.DataFrame(
            {
                "date": dates,
                "regime_id": "unknown__unknown__bocpd_stable",
                "state_change_probability": 0.0,
                "state_transition_probability": 0.0,
                "state_posterior_entropy": 0.0,
                "state_bocpd_boundary_code": 0.0,
                "state_cold_start": 1.0,
            }
        )

    init_count = max(1, min(int(bocpd_config.cold_start_burn_in), int(finite_indices.size)))
    init_hist = log_rv[finite_indices[:init_count]].astype(np.float64)
    service = BOCPDRegimeService(bocpd_config)
    service.initialize(init_hist)

    rows: list[dict[str, object]] = []
    for idx in finite_indices.tolist():
        record = service.update(
            _parse_decision_ts(dates[idx]),
            float(log_rv[idx]),
            log_return=float(returns[idx]),
            pit_boundary_idx=idx,
            log_rv_history=log_rv[: idx + 1],
            returns_history=returns[: idx + 1],
        )
        rows.append(
            {
                "date": dates[idx],
                "regime_id": record.regime_id,
                "state_change_probability": float(record.change_probability),
                "state_transition_probability": float(record.transition_probability),
                "state_posterior_entropy": float(record.posterior_entropy),
                "state_bocpd_boundary_code": _BOUNDARY_FLAG_CODES.get(record.boundary_flag, 1.0),
                "state_cold_start": 1.0 if record.cold_start else 0.0,
            }
        )

    regime = pd.DataFrame(rows)
    all_dates = pd.DataFrame({"date": dates})
    return (
        all_dates.merge(regime, on="date", how="left")
        .ffill()
        .fillna(
            {
                "regime_id": "unknown__unknown__bocpd_stable",
                "state_change_probability": 0.0,
                "state_transition_probability": 0.0,
                "state_posterior_entropy": 0.0,
                "state_bocpd_boundary_code": 0.0,
                "state_cold_start": 1.0,
            }
        )
    )


def load_prediction_panel(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "predictions" / "model_prediction_panel.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    return pd.read_parquet(path)


def _load_candidate_positions(
    run_dir: Path,
    predictions: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    date_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Prefer pre-built candidate positions; avoid re-ranking the full prediction panel."""

    pos_path = Path(run_dir) / "predictions" / "candidate_position_panel.parquet"
    if pos_path.is_file():
        positions = pd.read_parquet(pos_path)
        if date_filter is not None:
            positions = positions.loc[positions["date"].astype(str).isin(date_filter)]
        return positions
    frame = predictions
    if date_filter is not None:
        frame = frame.loc[frame["date"].astype(str).isin(date_filter)]
    return predictions_to_candidate_positions(
        frame,
        top_k=portfolio_spec.top_k,
        single_name_cap=portfolio_spec.single_name_cap,
    )


def _cross_sectional_panel_state_features(panel_slice: pd.DataFrame) -> pd.DataFrame:
    """PIT cross-sectional aggregates from lagged per-instrument returns (date T only)."""

    from pysrc.portfolio.labels import FORWARD_RETURN_COLUMN

    empty_columns = (
        "date",
        "state_cs_median_lagged_ret",
        "state_cs_breadth_positive",
        "state_cs_return_dispersion",
        "state_cs_universe_count",
    )
    if panel_slice.empty or FORWARD_RETURN_COLUMN not in panel_slice.columns:
        return pd.DataFrame(columns=list(empty_columns))

    work = panel_slice.loc[:, ["date", "instrument", FORWARD_RETURN_COLUMN]].copy()
    work["date"] = work["date"].astype(str)
    work = work.sort_values(["instrument", "date"], kind="mergesort")
    work["_lagged_ret"] = work.groupby("instrument", sort=True)[FORWARD_RETURN_COLUMN].shift(1)

    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", sort=True):
        vals = group["_lagged_ret"].astype(float)
        finite = vals[np.isfinite(vals.to_numpy(dtype=np.float64))]
        n = int(finite.size)
        if n == 0:
            rows.append(
                {
                    "date": str(date),
                    "state_cs_median_lagged_ret": 0.0,
                    "state_cs_breadth_positive": 0.0,
                    "state_cs_return_dispersion": 0.0,
                    "state_cs_universe_count": 0.0,
                }
            )
            continue
        rows.append(
            {
                "date": str(date),
                "state_cs_median_lagged_ret": float(finite.median()),
                "state_cs_breadth_positive": float((finite > 0).mean()),
                "state_cs_return_dispersion": float(finite.std(ddof=0)) if n > 1 else 0.0,
                "state_cs_universe_count": float(n),
            }
        )
    return pd.DataFrame(rows)


def build_policy_state_features(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    config: MetaRouterConfig | None = None,
    panel_slice: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach date-level state_* features for the local policy selector."""

    pred_summary = (
        predictions.groupby(["date", "model_id"], sort=True)["prediction"].mean().reset_index()
    )
    frame = labels.copy()
    disp = (
        pred_summary.groupby("date", sort=True)["prediction"]
        .agg(prediction_dispersion=lambda s: float(s.astype(float).std(ddof=0)))
        .reset_index()
    )
    default_returns = (
        frame.loc[
            frame["candidate_id"].astype(str) == DEFAULT_CANDIDATE_ID,
            ["date", "net_return"],
        ]
        .drop_duplicates(subset=["date"])
        .sort_values("date", kind="mergesort")
    )
    market = default_returns.rename(columns={"net_return": "_def_ret"}).copy()
    market["state_market_ret_1d"] = market["_def_ret"].astype(float).shift(1).fillna(0.0)
    market["state_market_trend_20"] = (
        market["_def_ret"].astype(float).rolling(20, min_periods=5).mean().fillna(0.0)
    )
    market["state_market_vol_20"] = (
        market["_def_ret"].astype(float).rolling(20, min_periods=5).std().fillna(0.0)
    )
    market["state_realized_vol"] = market["state_market_vol_20"]
    state = disp.merge(
        market.loc[
            :,
            [
                "date",
                "state_market_ret_1d",
                "state_market_trend_20",
                "state_market_vol_20",
                "state_realized_vol",
            ],
        ],
        on="date",
        how="left",
    )
    state["state_prediction_dispersion"] = state["prediction_dispersion"].fillna(0.0)
    state["state_dispersion"] = state["state_prediction_dispersion"]
    state = state.drop(columns=["prediction_dispersion"], errors="ignore")
    frame = frame.merge(
        state.loc[
            :,
            [
                "date",
                "state_market_ret_1d",
                "state_market_trend_20",
                "state_market_vol_20",
                "state_realized_vol",
                "state_prediction_dispersion",
                "state_dispersion",
            ],
        ],
        on="date",
        how="left",
    )
    for column in (
        "state_market_ret_1d",
        "state_market_trend_20",
        "state_market_vol_20",
        "state_realized_vol",
        "state_prediction_dispersion",
        "state_dispersion",
    ):
        frame[column] = frame[column].fillna(0.0)
    if not pred_summary.empty:
        pred_map = (
            pred_summary.assign(candidate_id=pred_summary["model_id"].astype(str))
            .rename(columns={"prediction": "cand_mean_prediction"})
            .loc[:, ["date", "candidate_id", "cand_mean_prediction"]]
        )
        frame = frame.merge(pred_map, on=["date", "candidate_id"], how="left")
        frame["cand_mean_prediction"] = frame["cand_mean_prediction"].fillna(0.0)
    utility = frame.loc[:, ["date", "candidate_id", "net_utility"]].copy()
    utility = utility.sort_values(["candidate_id", "date"], kind="mergesort")
    utility["cand_recent_utility_20"] = (
        utility.groupby("candidate_id", sort=True)["net_utility"]
        .transform(lambda s: s.astype(float).rolling(20, min_periods=5).mean().shift(1))
        .fillna(0.0)
    )
    frame = frame.merge(
        utility.loc[:, ["date", "candidate_id", "cand_recent_utility_20"]],
        on=["date", "candidate_id"],
        how="left",
    )
    frame["cand_recent_utility_20"] = frame["cand_recent_utility_20"].fillna(0.0)
    regime = _bocpd_regime_features_from_default_returns(labels)
    frame = frame.merge(regime, on="date", how="left")
    frame["regime_id"] = frame["regime_id"].fillna("unknown__unknown__bocpd_stable")
    for column in (
        "state_change_probability",
        "state_transition_probability",
        "state_posterior_entropy",
        "state_bocpd_boundary_code",
        "state_cold_start",
    ):
        frame[column] = frame[column].fillna(0.0)
    if panel_slice is not None and not panel_slice.empty:
        cs = _cross_sectional_panel_state_features(panel_slice)
        if not cs.empty:
            frame = frame.merge(cs, on="date", how="left")
            for column in (
                "state_cs_median_lagged_ret",
                "state_cs_breadth_positive",
                "state_cs_return_dispersion",
                "state_cs_universe_count",
            ):
                frame[column] = frame[column].fillna(0.0)
    if config is not None and config.state_features:
        missing = [c for c in config.state_features if c not in frame.columns]
        if missing:
            raise ValueError(
                f"build_policy_state_features: missing configured state_features {missing}"
            )
    return _assign_policy_splits(frame)


def _assign_policy_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """When predictions are OOS-only (split=test), use walk-forward fold_id for train/test."""

    out = frame.copy()
    if "split" not in out.columns or "fold_id" not in out.columns:
        return out
    labeled_split = out["split"].dropna().astype(str)
    if labeled_split.empty or labeled_split.nunique() != 1:
        return out
    if labeled_split.iloc[0] != "test":
        return out
    folds = sorted(
        f for f in out["fold_id"].astype(str).dropna().unique().tolist() if f and f != "nan"
    )
    if len(folds) < 2:
        return out
    train_folds = set(folds[:-1])
    out["split"] = np.where(out["fold_id"].astype(str).isin(train_folds), "train", "test")
    return out


def build_policy_training_frame(
    run_dir: Path,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    *,
    target_column: str = "forward_return_horizon",
    config: MetaRouterConfig | None = None,
) -> pd.DataFrame:
    """Build date×candidate training rows with delta_utility_vs_default labels."""

    predictions = load_prediction_panel(run_dir)
    panel_slice = _panel_slice_for_simulation(Path(panel_path), target_column=target_column)
    weights = _load_candidate_positions(run_dir, predictions, portfolio_spec)
    if weights.empty:
        raise ValueError("No candidate weights built from prediction panel")
    router_config = config or MetaRouterConfig(default_candidate_id=DEFAULT_CANDIDATE_ID)
    labels = build_portfolio_labels(weights, panel_slice, router_config)
    if "split" not in labels.columns:
        split_map = weights.loc[:, ["date", "candidate_id", "split"]].drop_duplicates()
        labels = labels.merge(split_map, on=["date", "candidate_id"], how="left")
    if "fold_id" not in labels.columns:
        fold_map = weights.loc[:, ["date", "candidate_id", "fold_id"]].drop_duplicates()
        labels = labels.merge(fold_map, on=["date", "candidate_id"], how="left")
    panel_for_features = _panel_slice_for_simulation(Path(panel_path), target_column=target_column)
    return build_policy_state_features(
        predictions,
        labels,
        config=router_config,
        panel_slice=panel_for_features,
    )


def routed_positions_from_decisions(
    decisions: pd.DataFrame,
    positions_by_candidate: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Select per-date positions for the routed candidate."""

    parts: list[pd.DataFrame] = []
    for row in decisions.itertuples(index=False):
        candidate_id = str(row.chosen_candidate_id)
        date = str(row.date)
        pos = positions_by_candidate.get(candidate_id)
        if pos is None or pos.empty:
            continue
        day = pos.loc[pos["date"].astype(str) == date].copy()
        if day.empty:
            continue
        day["gate_id"] = "local_policy_selector"
        parts.append(day)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _parse_model_weights(weights_json: object) -> dict[str, float]:
    import json

    if weights_json is None:
        return {}
    try:
        raw = json.loads(str(weights_json))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if float(v) > 0.0}


def decisions_use_weighted_blend(decisions: pd.DataFrame) -> bool:
    """True when any decision row carries multi-candidate model_weights_json."""

    if "model_weights_json" not in decisions.columns:
        return False
    for weights_json in decisions["model_weights_json"].tolist():
        weights = _parse_model_weights(weights_json)
        if len(weights) > 1:
            return True
    return False


def routed_positions_from_weighted_decisions(
    decisions: pd.DataFrame,
    positions_by_candidate: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Blend per-date positions using model_weights_json (multi-candidate mix)."""

    parts: list[pd.DataFrame] = []
    for row in decisions.itertuples(index=False):
        date = str(row.date)
        weights = _parse_model_weights(getattr(row, "model_weights_json", "{}"))
        if not weights:
            candidate_id = str(
                getattr(row, "chosen_candidate_id", getattr(row, "selected_candidate_id", ""))
            )
            if not candidate_id:
                continue
            weights = {candidate_id: 1.0}
        total = sum(weights.values())
        if total <= 0.0:
            continue
        day_parts: list[pd.DataFrame] = []
        for candidate_id, weight in weights.items():
            pos = positions_by_candidate.get(str(candidate_id))
            if pos is None or pos.empty:
                continue
            day = pos.loc[pos["date"].astype(str) == date].copy()
            if day.empty:
                continue
            day["target_weight"] = day["target_weight"].astype(float) * (float(weight) / total)
            day_parts.append(day)
        if not day_parts:
            continue
        blended = pd.concat(day_parts, ignore_index=True)
        group_cols = [c for c in ("date", "fold_id", "split", "ticker") if c in blended.columns]
        if group_cols:
            blended = (
                blended.groupby(group_cols, sort=True, as_index=False)
                .agg(target_weight=("target_weight", "sum"))
                .assign(candidate_id="weighted_blend")
            )
        else:
            blended["candidate_id"] = "weighted_blend"
        blended["gate_id"] = str(getattr(row, "gate_id", "weighted_blend"))
        parts.append(blended)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out[list(CANDIDATE_POSITION_PANEL_COLUMNS)]


def build_fold_attribution_report(
    run_dir: Path,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    *,
    fold_id: str = "fold_1",
    target_column: str = "forward_return_horizon",
) -> dict[str, Any]:
    """Per-fold return distribution and concentration metrics (fold_1 skepticism)."""

    predictions = load_prediction_panel(run_dir)
    panel_slice = _panel_slice_for_simulation(Path(panel_path), target_column=target_column)
    capacity = _capacity_limit(portfolio_spec)
    cost = portfolio_spec.cost_bps
    focus = ("xgboost", DEFAULT_CANDIDATE_ID)
    by_candidate: dict[str, dict[str, float | int]] = {}
    for model_id in focus:
        model_preds = predictions.loc[
            (predictions["model_id"].astype(str) == model_id)
            & (predictions["fold_id"].astype(str) == fold_id)
        ]
        positions = predictions_to_candidate_positions(
            model_preds,
            top_k=portfolio_spec.top_k,
            single_name_cap=portfolio_spec.single_name_cap,
        )
        simulated = simulate_candidate_portfolios(
            positions,
            panel_slice,
            cost_bps=cost,
            capacity_limit=capacity,
        )
        if simulated.empty:
            continue
        net = simulated["net_return"].astype(float).to_numpy()
        net = net[np.isfinite(net)]
        weights = positions.groupby("date")["target_weight"].apply(
            lambda s: float((s.astype(float) ** 2).sum())
        )
        hhi = float(weights.mean()) if len(weights) else 0.0
        by_candidate[model_id] = {
            "n_days": int(net.size),
            "daily_return_std": float(np.std(net)) if net.size else 0.0,
            "net_sharpe": _sharpe_from_returns(net),
            "position_hhi_mean": hhi,
        }
    return {
        "schema_version": "fold_attribution_report.v1",
        "run_id": Path(run_dir).name,
        "fold_id": fold_id,
        "by_candidate": by_candidate,
    }


def _economics_for_mask(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    mask: pd.Series,
) -> dict[str, float]:
    cand = frame.loc[(frame["candidate_id"].astype(str) == candidate_id) & mask]
    if cand.empty:
        return {"net_sharpe": 0.0, "cumulative_log_return": 0.0, "n_days": 0}
    net = cand["net_return"].astype(float).to_numpy()
    net = net[np.isfinite(net)]
    return {
        "net_sharpe": _sharpe_from_returns(net),
        "cumulative_log_return": float(np.sum(np.log1p(net))) if net.size else 0.0,
        "n_days": int(net.size),
    }


def _fold1_head_to_head(
    training_frame: pd.DataFrame,
    routed_outputs: pd.DataFrame,
    *,
    fold_id: str = "fold_1",
) -> dict[str, Any]:
    fold_mask = training_frame["fold_id"].astype(str) == fold_id
    routed_fold = routed_outputs.loc[routed_outputs["fold_id"].astype(str) == fold_id]
    routed_net = routed_fold["net_return"].astype(float).to_numpy()
    routed_net = routed_net[np.isfinite(routed_net)]
    routed_econ = {
        "net_sharpe": _sharpe_from_returns(routed_net),
        "cumulative_log_return": float(np.sum(np.log1p(routed_net))) if routed_net.size else 0.0,
        "n_days": int(routed_net.size),
    }
    xgb_econ = _economics_for_mask(training_frame, candidate_id="xgboost", mask=fold_mask)
    eq_econ = _economics_for_mask(training_frame, candidate_id=DEFAULT_CANDIDATE_ID, mask=fold_mask)
    xgb_vs_eq_gap = float(xgb_econ["net_sharpe"] - eq_econ["net_sharpe"])
    remaining_vs_xgb = float(xgb_econ["net_sharpe"] - routed_econ["net_sharpe"])
    gap_closed = float(xgb_vs_eq_gap - remaining_vs_xgb)
    return {
        "fold_id": fold_id,
        "routed": routed_econ,
        "xgboost": xgb_econ,
        "equal_blend": eq_econ,
        "xgboost_vs_equal_blend_gap": xgb_vs_eq_gap,
        "gap_closed_vs_equal_blend": gap_closed,
        "gap_close_pass": gap_closed >= _FOLD1_GAP_CLOSE_MIN,
    }


def _evaluate_pdr002_gate1(
    routed_test_sharpe: float,
    baseline_test_econ: dict[str, dict[str, float]],
    fold1_head_to_head: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    eq_test = baseline_test_econ.get(DEFAULT_CANDIDATE_ID, {})
    eq_test_sharpe = float(eq_test.get("net_sharpe", 0.0))
    beats_equal_blend_test = routed_test_sharpe >= eq_test_sharpe
    fold1_gap_close = bool(fold1_head_to_head.get("gap_close_pass", False))
    policy_smoke_pass = beats_equal_blend_test or fold1_gap_close
    return policy_smoke_pass, {
        "beats_equal_blend_test": beats_equal_blend_test,
        "fold1_gap_close_pass": fold1_gap_close,
        "routed_test_sharpe": routed_test_sharpe,
        "equal_blend_test_sharpe": eq_test_sharpe,
        "fold1_gap_closed": float(fold1_head_to_head.get("gap_closed_vs_equal_blend", 0.0)),
        "fold1_gap_close_min": _FOLD1_GAP_CLOSE_MIN,
    }


def _evaluate_pdr002_gate2(
    routed_test_sharpe: float,
    baseline_test_econ: dict[str, dict[str, float]],
    fold1_head_to_head: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Gate 2: best sweep config beats equal_blend test Sharpe OR fold1 gap_close >= 0.3."""

    passed, details = _evaluate_pdr002_gate1(
        routed_test_sharpe,
        baseline_test_econ,
        fold1_head_to_head,
    )
    return passed, {**details, "policy_sweep_pass": passed}


_POLICY_SWEEP_RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0)
_POLICY_SWEEP_SWITCH_MARGINS: tuple[float, ...] = (0.0, 0.05, 0.1)
_POLICY_SWEEP_UNCERTAINTY_KS: tuple[float, ...] = (0.5, 1.0, 2.0)
_POLICY_SWEEP_COST_BUFFERS: tuple[float, ...] = (0.0, 0.01)


def _policy_sweep_grid() -> list[dict[str, float]]:
    return [
        {
            "selector_ridge_alpha": alpha,
            "switch_margin": margin,
            "selector_uncertainty_k": uncertainty_k,
            "selector_cost_buffer": cost_buffer,
        }
        for alpha in _POLICY_SWEEP_RIDGE_ALPHAS
        for margin in _POLICY_SWEEP_SWITCH_MARGINS
        for uncertainty_k in _POLICY_SWEEP_UNCERTAINTY_KS
        for cost_buffer in _POLICY_SWEEP_COST_BUFFERS
    ]


def _run_policy_allocation_for_config(
    training_frame: pd.DataFrame,
    predictions_panel: pd.DataFrame,
    panel_slice: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    run_dir: Path,
    config: MetaRouterConfig,
    *,
    capacity: float,
) -> dict[str, Any]:
    from pysrc.meta.local_policy_selector import predict_candidate_deltas

    selector = train_local_policy_selector(training_frame, config)
    predictions = predict_candidate_deltas(selector, training_frame)
    decisions = route_decisions(predictions, config)

    route_dates = set(decisions["date"].astype(str).tolist())
    all_positions = _load_candidate_positions(
        run_dir,
        predictions_panel,
        portfolio_spec,
        date_filter=route_dates,
    )
    positions_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate_id in sorted(all_positions["candidate_id"].astype(str).unique().tolist()):
        positions_by_candidate[candidate_id] = all_positions.loc[
            all_positions["candidate_id"].astype(str) == candidate_id
        ].copy()

    routed_positions = routed_positions_from_decisions(decisions, positions_by_candidate)
    routed_outputs = simulate_candidate_portfolios(
        routed_positions,
        panel_slice,
        cost_bps=portfolio_spec.cost_bps,
        capacity_limit=capacity,
    )
    routed_outputs = _attach_fold_id_to_outputs(routed_outputs, routed_positions)

    test_mask = training_frame["split"].astype(str) == "test"
    test_dates = set(training_frame.loc[test_mask, "date"].astype(str).tolist())
    routed_test = routed_outputs.loc[routed_outputs["date"].astype(str).isin(test_dates)]
    routed_net = routed_test["net_return"].astype(float).to_numpy()
    routed_net = routed_net[np.isfinite(routed_net)]

    baseline_econ: dict[str, dict[str, float]] = {}
    for candidate_id in ("xgboost", DEFAULT_CANDIDATE_ID):
        cand = training_frame.loc[
            (training_frame["candidate_id"].astype(str) == candidate_id) & test_mask
        ]
        if cand.empty:
            continue
        net = cand["net_return"].astype(float).to_numpy()
        net = net[np.isfinite(net)]
        baseline_econ[candidate_id] = {
            "net_sharpe": _sharpe_from_returns(net),
            "cumulative_log_return": float(np.sum(np.log1p(net))) if net.size else 0.0,
            "n_days": int(net.size),
        }

    fold1_head_to_head = _fold1_head_to_head(training_frame, routed_outputs, fold_id="fold_1")
    routed_test_sharpe = _sharpe_from_returns(routed_net)
    return {
        "routed_positions": routed_positions,
        "routed_test_economics": {
            "net_sharpe": routed_test_sharpe,
            "cumulative_log_return": float(np.sum(np.log1p(routed_net)))
            if routed_net.size
            else 0.0,
            "n_days": int(routed_net.size),
        },
        "baseline_test_economics": baseline_econ,
        "fold1_head_to_head": fold1_head_to_head,
        "routed_test_sharpe": routed_test_sharpe,
    }


def _write_portfolio_target_plans(
    routed_positions: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    from pysrc.pipeline.candidate_portfolios.production_backtest import (
        candidate_positions_to_portfolio_target_plans,
    )

    target_plans = candidate_positions_to_portfolio_target_plans(
        routed_positions,
        strategy_id="local_policy_selector",
    )
    out_path = run_dir / "reports" / "portfolio_target_plans.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "portfolio_target_plans.v1",
        "strategy_id": "local_policy_selector",
        "plan_count": len(target_plans),
        "plans": [plan.to_backtest_context() for plan in target_plans],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "portfolio_target_plan_count": len(target_plans),
        "portfolio_target_plans_path": str(out_path),
    }


def run_policy_smoke_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    target_column: str = "forward_return_horizon",
    emit_target_plans: bool = False,
    config: MetaRouterConfig | None = None,
    report_name: str | None = None,
) -> dict[str, Any]:
    """Train local policy selector and compare routed vs baseline economics."""

    run_dir = Path(run_dir)
    training_frame = build_policy_training_frame(
        run_dir,
        panel_path,
        portfolio_spec,
        target_column=target_column,
        config=config,
    )
    if config is None:
        config = MetaRouterConfig(
            default_candidate_id=DEFAULT_CANDIDATE_ID,
            selector_ridge_alpha=1.0,
            selector_uncertainty_k=1.0,
            selector_cost_buffer=0.0,
            switch_margin=0.0,
        )
    predictions_panel = load_prediction_panel(run_dir)
    panel_slice = _panel_slice_for_simulation(Path(panel_path), target_column=target_column)
    capacity = _capacity_limit(portfolio_spec)
    allocation = _run_policy_allocation_for_config(
        training_frame,
        predictions_panel,
        panel_slice,
        portfolio_spec,
        run_dir,
        config,
        capacity=capacity,
    )
    baseline_econ = allocation["baseline_test_economics"]
    fold1_head_to_head = allocation["fold1_head_to_head"]
    routed_test_sharpe = float(allocation["routed_test_sharpe"])
    routed_positions = allocation["routed_positions"]
    routed_test_econ = allocation["routed_test_economics"]

    fold_attribution = build_fold_attribution_report(
        run_dir,
        panel_path,
        portfolio_spec,
        fold_id="fold_1",
        target_column=target_column,
    )
    policy_smoke_pass, gate1 = _evaluate_pdr002_gate1(
        routed_test_sharpe,
        baseline_econ,
        fold1_head_to_head,
    )

    report: dict[str, Any] = {
        "schema_version": "policy_allocation_report.v2",
        "run_id": run_dir.name,
        "default_candidate_id": DEFAULT_CANDIDATE_ID,
        "routed_test_economics": routed_test_econ,
        "baseline_test_economics": baseline_econ,
        "fold_attribution": fold_attribution,
        "fold1_head_to_head": fold1_head_to_head,
        "pdr002_gate1": gate1,
        "policy_smoke_pass": policy_smoke_pass,
    }
    if emit_target_plans and not routed_positions.empty:
        report.update(_write_portfolio_target_plans(routed_positions, run_dir))
    out_name = report_name or "policy_allocation_report.v2.json"
    out_path = run_dir / "reports" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def run_policy_sweep(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    target_column: str = "forward_return_horizon",
    report_name: str | None = None,
) -> dict[str, Any]:
    """Grid-search local policy selector hyperparameters; evaluate PDR-002 Gate 2."""

    from pysrc.artifact_registry._atomic import atomic_write_json

    run_dir = Path(run_dir)
    training_frame = build_policy_training_frame(
        run_dir,
        panel_path,
        portfolio_spec,
        target_column=target_column,
    )
    predictions_panel = load_prediction_panel(run_dir)
    panel_slice = _panel_slice_for_simulation(Path(panel_path), target_column=target_column)
    capacity = _capacity_limit(portfolio_spec)

    sweep_results: list[dict[str, Any]] = []
    for params in _policy_sweep_grid():
        config = MetaRouterConfig(
            default_candidate_id=DEFAULT_CANDIDATE_ID,
            selector_ridge_alpha=params["selector_ridge_alpha"],
            selector_uncertainty_k=params["selector_uncertainty_k"],
            selector_cost_buffer=params["selector_cost_buffer"],
            switch_margin=params["switch_margin"],
        )
        allocation = _run_policy_allocation_for_config(
            training_frame,
            predictions_panel,
            panel_slice,
            portfolio_spec,
            run_dir,
            config,
            capacity=capacity,
        )
        sweep_results.append({**params, **allocation})

    best = max(sweep_results, key=lambda row: float(row["routed_test_sharpe"]))
    policy_sweep_pass, gate2 = _evaluate_pdr002_gate2(
        float(best["routed_test_sharpe"]),
        best["baseline_test_economics"],
        best["fold1_head_to_head"],
    )

    report: dict[str, Any] = {
        "schema_version": "policy_sweep_results.v1",
        "run_id": run_dir.name,
        "default_candidate_id": DEFAULT_CANDIDATE_ID,
        "grid": {
            "selector_ridge_alpha": list(_POLICY_SWEEP_RIDGE_ALPHAS),
            "switch_margin": list(_POLICY_SWEEP_SWITCH_MARGINS),
            "selector_uncertainty_k": list(_POLICY_SWEEP_UNCERTAINTY_KS),
            "selector_cost_buffer": list(_POLICY_SWEEP_COST_BUFFERS),
        },
        "grid_size": len(sweep_results),
        "results": [
            {
                "selector_ridge_alpha": row["selector_ridge_alpha"],
                "switch_margin": row["switch_margin"],
                "selector_uncertainty_k": row["selector_uncertainty_k"],
                "selector_cost_buffer": row["selector_cost_buffer"],
                "routed_test_economics": row["routed_test_economics"],
                "fold1_head_to_head": row["fold1_head_to_head"],
                "routed_test_sharpe": row["routed_test_sharpe"],
            }
            for row in sweep_results
        ],
        "best_config": {
            "selector_ridge_alpha": best["selector_ridge_alpha"],
            "switch_margin": best["switch_margin"],
            "selector_uncertainty_k": best["selector_uncertainty_k"],
            "selector_cost_buffer": best["selector_cost_buffer"],
        },
        "best_routed_test_economics": best["routed_test_economics"],
        "best_fold1_head_to_head": best["fold1_head_to_head"],
        "pdr002_gate2": gate2,
        "policy_sweep_pass": policy_sweep_pass,
    }
    out_name = report_name or "policy_sweep_results.json"
    out_path = run_dir / "reports" / out_name
    atomic_write_json(out_path, report)
    report["report_path"] = str(out_path)
    return report


__all__ = [
    "build_fold_attribution_report",
    "build_policy_state_features",
    "build_policy_training_frame",
    "run_policy_smoke_for_model_matrix_run",
    "run_policy_sweep",
]

"""Canonical product schemas for the local MetaRouter production lane.

Grain vocabulary:

- ``ticker_date_candidate`` — (date, ticker, candidate_id) model scores.
- ``date_candidate`` — (date, candidate_id) portfolios, labels, training rows.
- ``date`` — route decisions and final allocations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

CASH_CANDIDATE_ID: Final[str] = "cash"
DEFAULT_CANDIDATE_ID: Final[str] = "equal_blend"

RouteAction = Literal["route", "abstain_default", "hold_inertia"]

CandidateOutputGrain = Literal["ticker_date_candidate", "date_candidate_portfolio"]

FORBIDDEN_FEATURE_PATTERNS: Final[tuple[str, ...]] = (
    "future_",
    "hindsight",
    "oracle",
    "forward_",
    "_net_utility",
)

DATE_STATE_PREFIX: Final[str] = "state_"
CANDIDATE_STATE_PREFIX: Final[str] = "cand_"
MACRO_STATE_PREFIX: Final[str] = "macro_"
TRAINING_TARGET_COLUMN: Final[str] = "delta_utility_vs_default"

# Model prediction product (canonical milestone panel)
MODEL_PREDICTION_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "model_id",
    "model_family",
    "instrument",
    "date",
    "interval",
    "fold_id",
    "split",
    "prediction",
    "prediction_rank",
    "confidence",
    "target_name",
)

# Candidate portfolio position product
CANDIDATE_POSITION_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "candidate_id",
    "ticker",
    "target_weight",
    "fold_id",
    "split",
)

CANDIDATE_PORTFOLIO_OUTPUT_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "date",
    "gross_return",
    "net_return",
    "turnover",
    "cost",
    "drawdown",
    "gross_exposure",
    "cash_weight",
    "capacity_used",
)

# MetaRouter state products
REGIME_STATE_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "regime",
    "volatility_state",
    "liquidity_state",
    "model_disagreement",
    "mean_confidence",
    "prediction_dispersion",
)

MACRO_STATE_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "interval",
    "risk_on_probability",
    "risk_off_probability",
    "expected_volatility",
    "liquidity_stress",
    "macro_regime_probabilities",
    "sector_tilts",
    "asset_class_tilts",
    "confidence",
)

# MetaRouter decision product
META_ROUTER_DECISION_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "fold_id",
    "split",
    "gate_id",
    "selected_candidate_id",
    "exposure_scale",
    "abstain_probability",
    "action",
    "model_weights_json",
)

# Final portfolio target product (post-gate routed weights)
META_ROUTER_PORTFOLIO_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "target_weight",
    "fold_id",
    "split",
    "gate_id",
)

# Legacy lane projections (indicator-children path only)
CANDIDATE_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "candidate_id",
    "score",
    "rank",
    "confidence",
    "feature_policy",
    "model_family",
    "run_id",
)

CANDIDATE_PORTFOLIO_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "candidate_id",
    "ticker",
    "target_weight",
)

PORTFOLIO_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "candidate_id",
    "n_names",
    "gross_exposure",
    "expected_turnover",
    "liquidity_proxy",
)

PORTFOLIO_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "candidate_id",
    "gross_return",
    "cost_estimate",
    "net_return",
    "turnover",
    "drawdown",
    "drawdown_increment",
    "risk_penalty",
    "net_utility",
    "delta_utility_vs_default",
)

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "candidate_id",
    "split",
    "predicted_delta_utility",
    "residual_std",
)

ROUTE_DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "split",
    "chosen_candidate_id",
    "action",
    "predicted_delta",
    "decision_buffer",
    "previous_candidate_id",
)


class CandidateOutputRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    ticker: str
    candidate_id: str
    score: float
    rank: int | None = None
    confidence: float | None = None
    feature_policy: str
    model_family: str
    run_id: str


class CandidatePortfolioRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    candidate_id: str
    ticker: str
    target_weight: float = Field(ge=-1.0, le=1.0)


class PortfolioLabelRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    candidate_id: str
    gross_return: float
    cost_estimate: float
    net_return: float
    turnover: float
    drawdown: float
    drawdown_increment: float
    risk_penalty: float
    net_utility: float
    delta_utility_vs_default: float


class MetaStateRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str


class MetaRouterTrainingRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str
    candidate_id: str
    split: Literal["train", "embargo", "test"]
    delta_utility_vs_default: float


class MetaRouterPredictionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    candidate_id: str
    split: str
    predicted_delta_utility: float
    residual_std: float


class RouteDecisionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    split: str
    chosen_candidate_id: str
    action: RouteAction
    predicted_delta: float
    decision_buffer: float
    previous_candidate_id: str | None = None


class MetaRouterConfig(BaseModel):
    """Configuration for the local MetaRouter lane."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int = 42
    smoke_test: bool = False
    keep_smoke_artifacts: bool = False
    persist_intermediates: bool = False
    processed_data_root: str = "data/processed"
    output_dir: str | None = None

    experiment_id: str | None = None
    experiment_config_path: str | None = None
    experiment_config_hash: str | None = None

    use_model_matrix: bool = False
    model_matrix_run_id: str | None = None
    use_legacy_indicator_children: bool = True

    macro_state_channel_enabled: bool = False

    gating_baselines: tuple[str, ...] = (
        "cash",
        "equal_weight_blend",
        "validation_weighted_blend",
        "regime_lookup",
        "recent_winner_selector",
        "linear_gate",
        "tree_gate",
        "boosted_tree_gate",
        "neural_gate",
        "reptile_neural_gate",
    )

    state_features: tuple[str, ...] | None = None
    forbidden_feature_patterns: tuple[str, ...] | None = None

    date_start: str | None = "2021-01-01"
    date_end: str | None = None
    max_tickers: int = Field(default=150, ge=2)
    min_ticker_coverage: float = Field(default=0.90, gt=0.0, le=1.0)

    include_ridge_child: bool = True
    ridge_alpha: float = Field(default=1.0, gt=0.0)
    ridge_refit_interval_days: int = Field(default=63, ge=1)
    ridge_min_train_days: int = Field(default=63, ge=10)

    top_k: int = Field(default=20, ge=1)
    single_name_cap: float = Field(default=0.10, gt=0.0, le=1.0)
    liquidity_floor_dollar_ratio: float | None = None

    forward_horizon_days: int = Field(default=1, ge=1)
    cost_bps: float = Field(default=5.0, ge=0.0)
    gamma_risk: float = Field(default=2.0, ge=0.0)
    lambda_turnover: float = Field(default=0.0005, ge=0.0)
    eta_drawdown: float = Field(default=0.05, ge=0.0)
    cash_daily_return: float = 0.0
    cash_hurdle_daily: float = Field(default=0.0, ge=0.0)

    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    embargo_days: int = Field(default=5, ge=0)
    warmup_days: int = Field(default=60, ge=0)

    selector_ridge_alpha: float = Field(default=1.0, gt=0.0)
    selector_uncertainty_k: float = Field(default=0.5, ge=0.0)
    selector_cost_buffer: float = Field(default=1e-4, ge=0.0)
    switch_margin: float = Field(default=5e-4, ge=0.0)
    inertia_rho: float = Field(default=0.30, ge=0.0, le=1.0)
    softmax_temperature: float = Field(default=400.0, gt=0.0)
    recent_perf_window_days: int = Field(default=20, ge=1)
    default_candidate_id: str = DEFAULT_CANDIDATE_ID


def require_columns(frame: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def validate_unique_grain(frame: pd.DataFrame, keys: Sequence[str], *, context: str) -> None:
    require_columns(frame, keys, context=context)
    duplicate_count = int(frame.duplicated(subset=list(keys)).sum())
    if duplicate_count:
        raise ValueError(
            f"{context}: grain {tuple(keys)} not unique ({duplicate_count} duplicate rows)"
        )


def validate_candidate_output_frame(frame: pd.DataFrame) -> None:
    require_columns(frame, CANDIDATE_OUTPUT_COLUMNS, context="candidate_outputs")
    validate_unique_grain(frame, ("date", "ticker", "candidate_id"), context="candidate_outputs")


def validate_candidate_portfolio_frame(frame: pd.DataFrame) -> None:
    require_columns(frame, CANDIDATE_PORTFOLIO_COLUMNS, context="candidate_portfolios")
    validate_unique_grain(frame, ("date", "candidate_id", "ticker"), context="candidate_portfolios")


def validate_portfolio_label_frame(frame: pd.DataFrame) -> None:
    require_columns(frame, PORTFOLIO_LABEL_COLUMNS, context="portfolio_labels")
    validate_unique_grain(frame, ("date", "candidate_id"), context="portfolio_labels")


def validate_macro_state_panel(frame: pd.DataFrame) -> None:
    require_columns(frame, MACRO_STATE_PANEL_COLUMNS, context="macro_state_panel")
    validate_unique_grain(frame, ("date", "interval"), context="macro_state_panel")


def select_feature_columns(
    frame: pd.DataFrame,
    *,
    whitelist: Sequence[str] | None = None,
) -> list[str]:
    candidates = sorted(
        column
        for column in frame.columns
        if column.startswith((DATE_STATE_PREFIX, CANDIDATE_STATE_PREFIX, MACRO_STATE_PREFIX))
    )
    if whitelist:
        allowed = set(whitelist)
        return [c for c in candidates if c in allowed]
    return candidates


def forbidden_patterns_for_config(
    config: MetaRouterConfig | None = None,
) -> tuple[str, ...]:
    if config is not None and config.forbidden_feature_patterns:
        return config.forbidden_feature_patterns
    return FORBIDDEN_FEATURE_PATTERNS


def scan_forbidden_features(
    feature_columns: Sequence[str],
    *,
    patterns: Sequence[str] | None = None,
) -> list[str]:
    active = patterns if patterns is not None else FORBIDDEN_FEATURE_PATTERNS
    return [column for column in feature_columns if any(pattern in column for pattern in active)]


def validate_training_frame(
    frame: pd.DataFrame,
    *,
    config: MetaRouterConfig | None = None,
) -> list[str]:
    require_columns(
        frame,
        ("date", "candidate_id", "split", TRAINING_TARGET_COLUMN),
        context="training_frame",
    )
    validate_unique_grain(frame, ("date", "candidate_id"), context="training_frame")
    whitelist = config.state_features if config is not None else None
    patterns = forbidden_patterns_for_config(config)
    feature_columns = select_feature_columns(frame, whitelist=whitelist)
    if not feature_columns:
        raise ValueError("training_frame: no state_/cand_/macro_ feature columns present")
    forbidden = scan_forbidden_features(feature_columns, patterns=patterns)
    if forbidden:
        raise ValueError(f"training_frame: forbidden leakage feature names {forbidden}")
    numeric_features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric_features.to_numpy(dtype=np.float64))
    invalid = sorted(
        column
        for column, has_invalid in zip(feature_columns, ~finite.all(axis=0), strict=True)
        if has_invalid
    )
    if invalid:
        raise ValueError(f"training_frame: non-finite feature values {invalid}")
    return feature_columns


__all__ = [
    "CANDIDATE_OUTPUT_COLUMNS",
    "CANDIDATE_PORTFOLIO_COLUMNS",
    "CANDIDATE_PORTFOLIO_OUTPUT_PANEL_COLUMNS",
    "CANDIDATE_POSITION_PANEL_COLUMNS",
    "CASH_CANDIDATE_ID",
    "DEFAULT_CANDIDATE_ID",
    "FORBIDDEN_FEATURE_PATTERNS",
    "MACRO_STATE_PANEL_COLUMNS",
    "MACRO_STATE_PREFIX",
    "META_ROUTER_DECISION_PANEL_COLUMNS",
    "META_ROUTER_PORTFOLIO_OUTPUT_COLUMNS",
    "MODEL_PREDICTION_PANEL_COLUMNS",
    "PORTFOLIO_DIAGNOSTIC_COLUMNS",
    "PORTFOLIO_LABEL_COLUMNS",
    "PREDICTION_COLUMNS",
    "REGIME_STATE_PANEL_COLUMNS",
    "ROUTE_DECISION_COLUMNS",
    "TRAINING_TARGET_COLUMN",
    "CandidateOutputRow",
    "CandidatePortfolioRow",
    "MetaRouterConfig",
    "MetaRouterPredictionRow",
    "MetaRouterTrainingRow",
    "MetaStateRow",
    "PortfolioLabelRow",
    "RouteDecisionRow",
    "require_columns",
    "scan_forbidden_features",
    "select_feature_columns",
    "validate_candidate_output_frame",
    "validate_candidate_portfolio_frame",
    "validate_macro_state_panel",
    "validate_portfolio_label_frame",
    "validate_training_frame",
    "validate_unique_grain",
]

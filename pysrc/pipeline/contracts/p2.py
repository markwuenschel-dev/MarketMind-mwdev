"""
Pydantic v2 contracts for the entire P2 broad-reset pipeline.
All JSON artifacts must validate against these models (or thin wrappers).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from pysrc.contracts.candidate_spec import CandidateSpec


def _default_supervision_root() -> str:
    return "data/processed/target_label_panel"


def _default_w3_b_root() -> str:
    return "data/processed/full_indicator_feature_panel"


def _default_panel_model_output_dir() -> str:
    return "artifacts/runs"


# =============================================================================
# ENUMS & LITERALS
# =============================================================================

SurfaceStatus = Literal[
    "available",
    "available_primary",
    "blocked_no_pit_safe_source",
    "blocked_no_validated_surface",
    "blocked_no_provenance",
    "blocked_dependency_missing",
]

ModelStatus = Literal[
    "available",
    "deferred_dependency_or_later_stage",
    "blocked",
]

CandidateStatus = Literal["active", "deferred", "blocked", "generated"]

RouterTarget = Literal[
    "child_utility_regression",
    "pairwise_advantage_regression",
]

DecisionRule = Literal[
    "free_routing",
    "train_default_override",
    "simple_ensemble_override",
    "majority_seed_consensus_override",
    "low_disagreement_override",
    "child_restricted_override",
    "no_switch_baseline",
]

NarrowClassification = Literal[
    "READY_FOR_PORTFOLIO_TEST",
    "NARROWED_OUT_BASELINE",
    "FAILED_TRAINING",
]

RunStatus = Literal[
    "RUNNABLE",
    "SKIPPED_MISSING_INPUT",
    "FAILED_LOAD",
    "FAILED_LEAKAGE_CHECK",
    "FAILED_TRAINING",
    "NARROWED_OUT_BASELINE",
    "BASELINE_UPLIFT_PASS",
    "READY_FOR_PORTFOLIO_TEST",
]

SplitPolicy = Literal[
    "time_aware_holdout_30pct",
    "time_aware_holdout_20pct",
    "purged_walk_forward_k3",
    "w4a_fold_split",
]

FeaturePolicy = Literal[
    "all_ta_child_regime",
    "w4b_date_level_v1",
    "full_indicator_universe_v1",
]

NarrowDataMode = Literal["artifact_v1", "panel_v2"]

PanelModelTarget = Literal[
    "forward_return",
    "cost_adjusted_forward_return",
    "cross_sectional_forward_return_rank",
    "positive_forward_return_label",
    "portfolio_adjusted_advantage",
    "abstain_or_trade_label",
    "allocation_score",
]

PanelTrainMemoryMode = Literal["auto", "low_memory", "in_memory"]


class RunPhase(StrEnum):
    MAP = "MAP"
    MATRIX = "MATRIX"
    NARROW = "NARROW"
    PORTFOLIO = "PORTFOLIO"
    FULL = "FULL"


# =============================================================================
# RUN METADATA (lineage & reproducibility)
# =============================================================================


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        default_factory=lambda: f"p2br_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    phase: RunPhase
    git_commit: str | None = None
    python_version: str | None = None
    random_seed: int = 42
    config_hash: str | None = None
    notes: str = "p2_broad_reset_v0.2_sip_anchored"


# =============================================================================
# PANEL & DERIVED SURFACE REGISTRIES (emitted by P2-MAP)
# =============================================================================


class PrimaryDataPanelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_id: str
    provider: str
    dataset_name: str
    path: str
    date_start: str
    date_end: str
    row_count: int
    instrument_count: int
    trading_day_count: int
    status: Literal["available_primary", "blocked_dependency_missing"] = "available_primary"
    adjusted_prices_for: list[str] = Field(default_factory=lambda: ["labels", "return_signals"])
    raw_prices_volume_for: list[str] = Field(
        default_factory=lambda: [
            "price_filters",
            "liquidity_filters",
            "tradability_filters",
        ]
    )
    pit_safe: bool = True
    reason: str | None = None


class DerivedSupervisionSurfaceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_id: str
    family: str
    status: SurfaceStatus
    supervision_path: str | None = None
    bundle_id: str | None = None
    bundle_surface_id: str | None = None
    schema_version: str = "w4_a.router_supervision.v1"
    derived_from_panel: str = "us_equities_sip_adjusted_daily_v1"
    role: Literal["primary_narrow", "stress_report_only", "embedded_features"] = "primary_narrow"
    reason: str | None = None
    pit_safe: bool = True


class SurfaceRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_id: str
    family: str
    status: SurfaceStatus
    reason: str | None = None
    path: str | None = None
    row_count: int | None = None
    feature_count: int | None = None
    available_at_decision_time: bool = True
    pit_safe: bool = True
    columns_sample: list[str] = Field(default_factory=list)
    derived_from_panel: str | None = None


class ModelFamilyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: ModelStatus
    reason: str | None = None
    supported_router_targets: list[RouterTarget] = Field(default_factory=list)
    supported_decision_rules: list[DecisionRule] = Field(default_factory=list)
    requires_extra: list[str] = Field(default_factory=list)
    default_hyperparams: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_legacy_rules(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        rules = values.get("supported_decision_rules")
        if isinstance(rules, list):
            values["supported_decision_rules"] = [
                "train_default_override" if rule == "default_to_best_child_override" else rule
                for rule in rules
            ]
        return values


class SignalFamilyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    signals: list[str]
    status: SurfaceStatus
    reason: str | None = None


class ArchitectureMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    primary_panel: PrimaryDataPanelEntry | None = None
    derived_surfaces: list[DerivedSupervisionSurfaceEntry] = Field(default_factory=list)
    surfaces: list[SurfaceRegistryEntry]
    summary: dict[str, Any] = Field(default_factory=dict)


class ModelFamilyRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    families: list[ModelFamilyEntry]


class SignalFamilyRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    families: list[SignalFamilyEntry]


class SurfaceMarketRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    primary_panel: PrimaryDataPanelEntry | None = None
    surfaces: list[SurfaceRegistryEntry]
    blocked_count: int
    available_count: int
    available_primary_count: int = 0


# =============================================================================
# CANDIDATE MATRIX
# =============================================================================


class CandidateMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    candidates: list[CandidateSpec]
    summary: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# NARROWING RESULTS
# =============================================================================


class SurfaceStressResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluated: bool
    reason: str | None = None
    utility: float | None = None


class BaselineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    test_utility: float
    test_sharpe_like: float | None = None
    win_rate_vs_cash: float | None = None
    notes: str = ""


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    test_utility: float
    best_child_test_utility: float
    delta_vs_best_child: float
    classification: NarrowClassification
    status: RunStatus
    notes: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class NarrowingReport(BaseModel):
    """Research-first narrowing report (no governance gate fields)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    input_surface: str
    supervision_path: str | None
    grain: str
    split_policy: SplitPolicy
    feature_policy: FeaturePolicy
    candidate_count: int
    baselines: dict[str, float]
    results: list[CandidateResult]
    survivors: list[str]
    next_action: str


class SurvivorManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    survivors: list[CandidateResult]
    ready_for_portfolio_test: bool
    recommendation: str
    next_step: Literal["P2-PORTFOLIO", "investigate_failures", "expand_matrix"]


class HumanSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: RunMetadata
    markdown: str
    key_findings: list[str]
    survivor_table: str | None = None


# =============================================================================
# CONFIG
# =============================================================================


class P2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_seed: int = 42
    pairwise_seed_start: int = 42
    pairwise_seed_count: int = Field(default=50, ge=1)
    smoke_test: bool = False
    data_path: str | None = None
    test_size: float = 0.30
    min_delta_for_override: float = 0.005
    confidence_level: float = 0.80
    output_dir: str = "artifacts/phase_ii/p2_broad_reset"
    run_dir: str | None = None
    max_candidates_first_run: int = 40
    panel_root: str = "data/massive/us_stocks_sip/adjusted_day_panel_v1"
    panel_manifest_path: str | None = None
    supervision_root: str = Field(default_factory=_default_supervision_root)
    supervision_bundle: str = "w3b_ta"
    supervision_surface: str = "surface_b"
    narrow_data_mode: NarrowDataMode = "artifact_v1"
    # 0 = auto (all CPU cores, capped by candidate count); 1 = serial
    narrow_workers: int = 0
    # 0 = auto (all cores when serial, 1 when narrow_workers > 1); else explicit n_jobs
    sklearn_n_jobs: int = 0
    w3_b_root: str = Field(default_factory=_default_w3_b_root)
    processed_data_root: str = "data/processed"
    panel_model_output_dir: str = Field(default_factory=_default_panel_model_output_dir)
    panel_coverage_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    panel_target: PanelModelTarget = "forward_return"
    panel_target_horizon_days: int | None = Field(default=None, ge=0)
    panel_include_w4a_features: bool = False
    panel_model_families: tuple[str, ...] = ("ridge", "random_forest")
    panel_train_memory_mode: PanelTrainMemoryMode = "auto"
    panel_train_chunk_rows: int = Field(default=250_000, ge=1)
    panel_train_scratch_dir: str | None = None
    panel_train_memory_safety_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    panel_train_peak_rss_limit_bytes: int = Field(default=8 * 1024 * 1024 * 1024, ge=0)
    # Per model/fold worker wall-clock limit; 0 disables timeout enforcement.
    panel_train_worker_timeout_seconds: int = Field(default=14_400, ge=0)
    panel_preserve_scratch_on_failure: bool = False
    panel_preserve_fragments_on_failure: bool = False
    # Cap training rows per fold for all model families (0 = use all rows).
    panel_train_max_rows_per_fold: int = Field(default=0, ge=0)
    # Quantile cap follows the same 0 = all rows convention.
    panel_quantile_max_train_rows: int = Field(default=0, ge=0)
    # Walk-forward fold count for panel train-matrix (0 = default: 3 real / 2 smoke).
    panel_walk_forward_folds: int = Field(default=0, ge=0)
    # Cap instruments for canary runs (0 = all instruments in panel).
    panel_max_instruments: int = Field(default=0, ge=0)
    portfolio_reference_capital: float = Field(default=1_000_000.0, gt=0.0)
    portfolio_adv_participation: float = Field(default=0.05, gt=0.0, le=1.0)
    portfolio_cost_bps: float = Field(default=5.0, ge=0.0)
    portfolio_rebalance_bars: int = Field(default=1, ge=1)
    portfolio_capital_grid: tuple[float, ...] = (
        10_000.0,
        100_000.0,
        1_000_000.0,
        5_000_000.0,
        10_000_000.0,
        25_000_000.0,
        50_000_000.0,
        100_000_000.0,
    )
    portfolio_cost_grid_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0)
    portfolio_rebalance_grid_bars: tuple[int, ...] = (1, 2, 5, 10, 21)
    portfolio_execution_delay_grid_bars: tuple[int, ...] = (0, 1, 2)
    portfolio_adv_grid: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10)
    portfolio_gross_cap_grid: tuple[float, ...] = (0.50, 0.75, 1.0)
    portfolio_single_name_cap_grid: tuple[float, ...] = (0.25, 0.50, 1.0)
    portfolio_volatility_target_grid: tuple[float | None, ...] = (None, 0.05, 0.10, 0.15, 0.20)

    # Experiment provenance (research YAML)
    experiment_id: str | None = None
    experiment_config_path: str | None = None
    experiment_config_hash: str | None = None

    @property
    def pairwise_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.pairwise_seed_start, self.pairwise_seed_start + self.pairwise_seed_count)
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def default_supervision_path(self) -> str:
        return str(
            Path(self.supervision_root)
            / "datasets"
            / self.supervision_bundle
            / self.supervision_surface
            / "router_supervision_rows.parquet"
        )

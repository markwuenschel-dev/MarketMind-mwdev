"""Canonical meta-router product registry: ID, path, durability, schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DurabilityClass = Literal["durable", "intermediate", "ephemeral", "legacy"]

# Canonical product filenames
MODEL_PREDICTION_PANEL: Final[str] = "model_prediction_panel.parquet"
CANDIDATE_POSITION_PANEL: Final[str] = "candidate_position_panel.parquet"
CANDIDATE_PORTFOLIO_OUTPUT_PANEL: Final[str] = "candidate_portfolio_output_panel.parquet"
REGIME_STATE_PANEL: Final[str] = "regime_state_panel.parquet"
META_ROUTER_TRAINING_PANEL: Final[str] = "meta_router_training_panel.parquet"
META_ROUTER_DECISION_PANEL: Final[str] = "meta_router_decision_panel.parquet"
META_ROUTER_PORTFOLIO_OUTPUT: Final[str] = "meta_router_portfolio_output.parquet"
META_ROUTER_EVALUATION_REPORT: Final[str] = "meta_router_evaluation_report.json"
RUN_META: Final[str] = "run_meta.json"
SMOKE_SUMMARY: Final[str] = "smoke_summary.json"


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One run-scoped product: identity, filename, layout, durability, schema."""

    product_id: str
    filename: str
    durability: DurabilityClass
    subdir: str = ""
    schema_ref: str | None = None
    legacy_key: str | None = None


# Canonical subdir layout (milestone / model-matrix path)
CANONICAL_ARTIFACTS: Final[dict[str, ArtifactSpec]] = {
    "model_prediction_panel": ArtifactSpec(
        product_id="model_prediction_panel",
        filename=MODEL_PREDICTION_PANEL,
        durability="durable",
        subdir="predictions",
        schema_ref="MODEL_PREDICTION_PANEL_COLUMNS",
        legacy_key="candidate_outputs",
    ),
    "candidate_position_panel": ArtifactSpec(
        product_id="candidate_position_panel",
        filename=CANDIDATE_POSITION_PANEL,
        durability="intermediate",
        subdir="predictions",
        schema_ref="CANDIDATE_POSITION_PANEL_COLUMNS",
        legacy_key="candidate_portfolios",
    ),
    "candidate_portfolio_output_panel": ArtifactSpec(
        product_id="candidate_portfolio_output_panel",
        filename=CANDIDATE_PORTFOLIO_OUTPUT_PANEL,
        durability="durable",
        subdir="diagnostics",
        schema_ref="CANDIDATE_PORTFOLIO_OUTPUT_PANEL_COLUMNS",
    ),
    "regime_state_panel": ArtifactSpec(
        product_id="regime_state_panel",
        filename=REGIME_STATE_PANEL,
        durability="intermediate",
        subdir="diagnostics",
        schema_ref="REGIME_STATE_PANEL_COLUMNS",
    ),
    "meta_router_training_panel": ArtifactSpec(
        product_id="meta_router_training_panel",
        filename=META_ROUTER_TRAINING_PANEL,
        durability="intermediate",
        subdir="diagnostics",
        legacy_key="training_frame",
    ),
    "meta_router_decision_panel": ArtifactSpec(
        product_id="meta_router_decision_panel",
        filename=META_ROUTER_DECISION_PANEL,
        durability="durable",
        subdir="diagnostics",
        schema_ref="META_ROUTER_DECISION_PANEL_COLUMNS",
        legacy_key="route_decisions",
    ),
    "meta_router_portfolio_output": ArtifactSpec(
        product_id="meta_router_portfolio_output",
        filename=META_ROUTER_PORTFOLIO_OUTPUT,
        durability="durable",
        subdir="predictions",
        schema_ref="META_ROUTER_PORTFOLIO_OUTPUT_COLUMNS",
    ),
    "meta_router_evaluation_report": ArtifactSpec(
        product_id="meta_router_evaluation_report",
        filename=META_ROUTER_EVALUATION_REPORT,
        durability="durable",
        subdir="reports",
    ),
}

# Legacy flat layout at run root (indicator-children path)
LEGACY_FLAT_ARTIFACTS: Final[dict[str, ArtifactSpec]] = {
    "panel_slice": ArtifactSpec(
        product_id="panel_slice",
        filename="panel_slice.parquet",
        durability="intermediate",
    ),
    "candidate_outputs": ArtifactSpec(
        product_id="candidate_outputs",
        filename="candidate_outputs.parquet",
        durability="intermediate",
        legacy_key="candidate_outputs",
    ),
    "candidate_portfolios": ArtifactSpec(
        product_id="candidate_portfolios",
        filename="candidate_portfolio_outputs.parquet",
        durability="intermediate",
        legacy_key="candidate_portfolios",
    ),
    "candidate_portfolio_diagnostics": ArtifactSpec(
        product_id="candidate_portfolio_diagnostics",
        filename="candidate_portfolio_diagnostics.parquet",
        durability="intermediate",
    ),
    "portfolio_labels": ArtifactSpec(
        product_id="portfolio_labels",
        filename="portfolio_labels.parquet",
        durability="intermediate",
        schema_ref="PORTFOLIO_LABEL_COLUMNS",
    ),
    "training_frame": ArtifactSpec(
        product_id="training_frame",
        filename="meta_router_training_frame.parquet",
        durability="intermediate",
        legacy_key="training_frame",
    ),
    "training_frame_audit": ArtifactSpec(
        product_id="training_frame_audit",
        filename="training_frame_audit.json",
        durability="intermediate",
    ),
    "predictions": ArtifactSpec(
        product_id="predictions",
        filename="meta_router_predictions.parquet",
        durability="intermediate",
        schema_ref="PREDICTION_COLUMNS",
    ),
    "selector_coefficients": ArtifactSpec(
        product_id="selector_coefficients",
        filename="selector_coefficients.json",
        durability="intermediate",
    ),
    "route_decisions": ArtifactSpec(
        product_id="route_decisions",
        filename="route_decisions.csv",
        durability="intermediate",
        legacy_key="route_decisions",
    ),
    "switch_attribution": ArtifactSpec(
        product_id="switch_attribution",
        filename="switch_attribution.json",
        durability="intermediate",
    ),
    "candidate_baselines": ArtifactSpec(
        product_id="candidate_baselines",
        filename="candidate_baselines.json",
        durability="intermediate",
    ),
    "portfolio_utility_summary": ArtifactSpec(
        product_id="portfolio_utility_summary",
        filename="portfolio_utility_summary.json",
        durability="intermediate",
    ),
    "summary_markdown": ArtifactSpec(
        product_id="summary_markdown",
        filename="meta_router_summary.md",
        durability="ephemeral",
    ),
    "split_audit": ArtifactSpec(
        product_id="split_audit",
        filename="split_audit.json",
        durability="intermediate",
    ),
    "inner_validation_sweep_json": ArtifactSpec(
        product_id="inner_validation_sweep_json",
        filename="inner_validation_sweep.json",
        durability="intermediate",
    ),
    "inner_validation_sweep_csv": ArtifactSpec(
        product_id="inner_validation_sweep_csv",
        filename="inner_validation_sweep.csv",
        durability="intermediate",
    ),
    "selected_config": ArtifactSpec(
        product_id="selected_config",
        filename="selected_config.json",
        durability="intermediate",
    ),
    "router_vs_ridge_decomposition": ArtifactSpec(
        product_id="router_vs_ridge_decomposition",
        filename="router_vs_ridge_decomposition.json",
        durability="intermediate",
    ),
}

LEGACY_ARTIFACT_ALIASES: Final[dict[str, str]] = {
    spec.legacy_key: spec.filename
    for spec in (*CANONICAL_ARTIFACTS.values(), *LEGACY_FLAT_ARTIFACTS.values())
    if spec.legacy_key is not None
}


def artifact_spec(key: str) -> ArtifactSpec:
    if key in LEGACY_FLAT_ARTIFACTS:
        return LEGACY_FLAT_ARTIFACTS[key]
    if key in CANONICAL_ARTIFACTS:
        return CANONICAL_ARTIFACTS[key]
    raise KeyError(f"Unknown artifact key: {key}")


def artifact_path(run_dir: Path | str, key: str) -> Path:
    """Resolve path for a legacy flat key or canonical product id."""

    spec = artifact_spec(key)
    base = Path(run_dir)
    if spec.subdir:
        return base / spec.subdir / spec.filename
    return base / spec.filename


def artifact_filenames() -> dict[str, str]:
    """Legacy flat key → filename (backward compatible with reports.py)."""

    return {key: spec.filename for key, spec in LEGACY_FLAT_ARTIFACTS.items()}


def durable_filenames() -> frozenset[str]:
    names = {RUN_META, SMOKE_SUMMARY}
    for spec in CANONICAL_ARTIFACTS.values():
        if spec.durability == "durable":
            names.add(spec.filename)
    return frozenset(names)


def intermediate_filenames() -> frozenset[str]:
    names: set[str] = set()
    for spec in (*CANONICAL_ARTIFACTS.values(), *LEGACY_FLAT_ARTIFACTS.values()):
        if spec.durability == "intermediate":
            names.add(spec.filename)
    return frozenset(names)


ARTIFACT_FILENAMES: Final[dict[str, str]] = artifact_filenames()


@dataclass(frozen=True, slots=True)
class MetaRouterProductPaths:
    run_dir: Path

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def diagnostics_dir(self) -> Path:
        return self.run_dir / "diagnostics"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    def model_prediction_panel(self) -> Path:
        return artifact_path(self.run_dir, "model_prediction_panel")

    def candidate_position_panel(self) -> Path:
        return artifact_path(self.run_dir, "candidate_position_panel")

    def candidate_portfolio_output_panel(self) -> Path:
        return artifact_path(self.run_dir, "candidate_portfolio_output_panel")

    def regime_state_panel(self) -> Path:
        return artifact_path(self.run_dir, "regime_state_panel")

    def meta_router_training_panel(self) -> Path:
        return artifact_path(self.run_dir, "meta_router_training_panel")

    def meta_router_decision_panel(self) -> Path:
        return artifact_path(self.run_dir, "meta_router_decision_panel")

    def meta_router_portfolio_output(self) -> Path:
        return artifact_path(self.run_dir, "meta_router_portfolio_output")

    def evaluation_report(self) -> Path:
        return artifact_path(self.run_dir, "meta_router_evaluation_report")


def resolve_meta_router_products(run_dir: Path | str) -> MetaRouterProductPaths:
    return MetaRouterProductPaths(run_dir=Path(run_dir))


__all__ = [
    "ArtifactSpec",
    "ARTIFACT_FILENAMES",
    "CANONICAL_ARTIFACTS",
    "CANDIDATE_PORTFOLIO_OUTPUT_PANEL",
    "CANDIDATE_POSITION_PANEL",
    "DurabilityClass",
    "LEGACY_ARTIFACT_ALIASES",
    "LEGACY_FLAT_ARTIFACTS",
    "META_ROUTER_DECISION_PANEL",
    "META_ROUTER_EVALUATION_REPORT",
    "META_ROUTER_PORTFOLIO_OUTPUT",
    "META_ROUTER_TRAINING_PANEL",
    "MODEL_PREDICTION_PANEL",
    "MetaRouterProductPaths",
    "REGIME_STATE_PANEL",
    "RUN_META",
    "SMOKE_SUMMARY",
    "artifact_filenames",
    "artifact_path",
    "artifact_spec",
    "durable_filenames",
    "intermediate_filenames",
    "resolve_meta_router_products",
]

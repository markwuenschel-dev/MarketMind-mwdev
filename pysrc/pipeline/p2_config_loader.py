"""P2 experiment YAML resolution and pydantic experiment contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pysrc.contracts.meta_router import MetaRouterConfig
from pysrc.models.registry import resolve_model_family
from pysrc.pipeline.contracts.p2 import P2Config

RESEARCH_P2_CONFIGS = Path("research/p2/configs")
DEFAULT_ROUTER_CONFIG = RESEARCH_P2_CONFIGS / "router_matrix.yaml"
DEFAULT_PANEL_CONFIG = RESEARCH_P2_CONFIGS / "panel_baselines.yaml"
DEFAULT_BROAD_MODEL_MATRIX_CONFIG = RESEARCH_P2_CONFIGS / "broad_model_matrix.yaml"
DEFAULT_PANEL_MODEL_MATRIX_CONFIG = RESEARCH_P2_CONFIGS / "panel_model_matrix.yaml"
DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG = RESEARCH_P2_CONFIGS / "candidate_portfolios.yaml"
DEFAULT_META_ROUTER_CONFIG = RESEARCH_P2_CONFIGS / "local_meta_router.yaml"
DEFAULT_REPTILE_CONFIG = RESEARCH_P2_CONFIGS / "reptile_regime_adaptation.yaml"


class ModelEntrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    params: dict[str, Any] = Field(default_factory=dict)
    sequence_length: int | None = None


class DataSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = "full_indicator_feature_panel"
    target: str = "forward_return"
    feature_policy: str = "full_indicator_universe_v1"
    split_policy: str = "walk_forward"
    target_horizon_days: int | None = None
    sequence_lengths: list[int] = Field(default_factory=lambda: [20, 60, 120])


class PortfolioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_bps: float = 5.0
    liquidity_constraints: bool = True
    capacity_constraints: bool = True
    cash_allowed: bool = True
    top_k: int = 20
    single_name_cap: float = 0.10


class MetaRouterExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    program: str = "p2"
    experiment: str = "local_meta_router"
    data: DataSpec = Field(default_factory=DataSpec)
    models: list[ModelEntrySpec] = Field(default_factory=list)
    portfolio: PortfolioSpec = Field(default_factory=PortfolioSpec)
    meta_router: dict[str, Any] = Field(default_factory=dict)
    candidates: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    gating_baselines: tuple[str, ...] = (
        "cash",
        "equal_weight_blend",
        "validation_weighted_blend",
        "regime_lookup",
        "linear_gate",
        "tree_gate",
        "neural_gate",
        "reptile_neural_gate",
    )


def load_p2_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"P2 config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P2 config must be a mapping: {config_path}")
    return payload


def config_content_hash(path: Path | str) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def resolve_config_path(name_or_path: str | None, *, default: Path) -> Path:
    if name_or_path is None:
        return default
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    shorthand = RESEARCH_P2_CONFIGS / f"{name_or_path}.yaml"
    if shorthand.is_file():
        return shorthand
    if not name_or_path.endswith(".yaml"):
        alt = RESEARCH_P2_CONFIGS / f"{name_or_path}.yaml"
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"P2 config not found: {name_or_path}")


def parse_meta_router_experiment(yaml_dict: dict[str, Any]) -> MetaRouterExperimentSpec:
    models_raw = yaml_dict.get("models") or []
    models: list[ModelEntrySpec] = []
    if isinstance(models_raw, list):
        for entry in models_raw:
            if isinstance(entry, dict) and isinstance(entry.get("family"), str):
                models.append(
                    ModelEntrySpec(
                        family=entry["family"],
                        params=dict(entry.get("params") or {}),
                        sequence_length=entry.get("sequence_length"),
                    )
                )
    data_raw = yaml_dict.get("data") if isinstance(yaml_dict.get("data"), dict) else {}
    portfolio_raw = (
        yaml_dict.get("portfolio") if isinstance(yaml_dict.get("portfolio"), dict) else {}
    )
    baselines = yaml_dict.get("gating_baselines") or yaml_dict.get("meta_router", {}).get(
        "baselines", ()
    )
    if isinstance(baselines, list):
        baseline_tuple = tuple(str(x) for x in baselines)
    else:
        baseline_tuple = MetaRouterExperimentSpec.model_fields["gating_baselines"].default
    return MetaRouterExperimentSpec(
        program=str(yaml_dict.get("program", "p2")),
        experiment=str(yaml_dict.get("experiment", "local_meta_router")),
        data=DataSpec.model_validate(data_raw),
        models=models,
        portfolio=PortfolioSpec.model_validate(portfolio_raw),
        meta_router=dict(yaml_dict.get("meta_router") or {}),
        candidates=dict(yaml_dict.get("candidates") or {}),
        evaluation=dict(yaml_dict.get("evaluation") or {}),
        gating_baselines=baseline_tuple,
    )


def yaml_to_p2_config(
    yaml_dict: dict[str, Any],
    *,
    cli_overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> P2Config:
    spec = parse_meta_router_experiment(yaml_dict)
    families = tuple(m.family for m in spec.models) if spec.models else ("ridge", "random_forest")
    target = spec.data.target
    panel_target = (
        target
        if target
        in {
            "forward_return",
            "cost_adjusted_forward_return",
            "cross_sectional_forward_return_rank",
            "positive_forward_return_label",
            "portfolio_adjusted_advantage",
            "abstain_or_trade_label",
            "allocation_score",
        }
        else "forward_return"
    )
    updates: dict[str, Any] = {
        "panel_model_families": families,
        "panel_target": panel_target,
        "panel_target_horizon_days": spec.data.target_horizon_days,
        "experiment_id": spec.experiment,
    }
    if config_path is not None:
        updates["experiment_config_path"] = str(config_path)
        updates["experiment_config_hash"] = config_content_hash(config_path)
    if spec.data.source == "full_indicator_feature_panel":
        updates["processed_data_root"] = "data/processed"
    training_raw = yaml_dict.get("training") if isinstance(yaml_dict.get("training"), dict) else {}
    if isinstance(training_raw.get("memory_mode"), str):
        updates["panel_train_memory_mode"] = str(training_raw["memory_mode"])
    if isinstance(training_raw.get("chunk_rows"), int):
        updates["panel_train_chunk_rows"] = int(training_raw["chunk_rows"])
    if "scratch_dir" in training_raw:
        scratch_dir = training_raw.get("scratch_dir")
        updates["panel_train_scratch_dir"] = None if scratch_dir is None else str(scratch_dir)
    if isinstance(training_raw.get("memory_safety_fraction"), (int, float)):
        updates["panel_train_memory_safety_fraction"] = float(
            training_raw["memory_safety_fraction"]
        )
    if isinstance(training_raw.get("peak_rss_limit_bytes"), int):
        updates["panel_train_peak_rss_limit_bytes"] = int(training_raw["peak_rss_limit_bytes"])
    if isinstance(training_raw.get("worker_timeout_seconds"), int):
        updates["panel_train_worker_timeout_seconds"] = int(training_raw["worker_timeout_seconds"])
    if isinstance(training_raw.get("preserve_scratch_on_failure"), bool):
        updates["panel_preserve_scratch_on_failure"] = bool(
            training_raw["preserve_scratch_on_failure"]
        )
    if isinstance(training_raw.get("preserve_fragments_on_failure"), bool):
        updates["panel_preserve_fragments_on_failure"] = bool(
            training_raw["preserve_fragments_on_failure"]
        )
    if isinstance(training_raw.get("max_train_rows_per_fold"), int):
        updates["panel_train_max_rows_per_fold"] = int(training_raw["max_train_rows_per_fold"])
    if isinstance(training_raw.get("quantile_max_train_rows"), int):
        updates["panel_quantile_max_train_rows"] = int(training_raw["quantile_max_train_rows"])
    if cli_overrides:
        updates.update(cli_overrides)
    return P2Config.model_validate(updates)


def yaml_to_meta_router_config(
    yaml_dict: dict[str, Any],
    *,
    cli_overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> MetaRouterConfig:
    spec = parse_meta_router_experiment(yaml_dict)
    updates: dict[str, Any] = {
        "experiment_id": spec.experiment,
        "use_model_matrix": bool(spec.models),
        "gating_baselines": spec.gating_baselines,
        "cost_bps": spec.portfolio.cost_bps,
        "top_k": spec.portfolio.top_k,
        "single_name_cap": spec.portfolio.single_name_cap,
        "default_candidate_id": str(spec.candidates.get("default_id", "equal_blend")),
    }
    if spec.models:
        updates["use_legacy_indicator_children"] = False
    meta = spec.meta_router
    if isinstance(meta.get("state_features"), list):
        updates["state_features"] = tuple(str(x) for x in meta["state_features"])
    forbidden = meta.get("forbidden_feature_patterns")
    if isinstance(forbidden, list):
        updates["forbidden_feature_patterns"] = tuple(str(x) for x in forbidden)
    if config_path is not None:
        updates["experiment_config_path"] = str(config_path)
        updates["experiment_config_hash"] = config_content_hash(config_path)
    if cli_overrides:
        updates.update(cli_overrides)
    return MetaRouterConfig.model_validate(updates)


def resolve_meta_router_battery_gate_ids(
    config: MetaRouterConfig,
    yaml_dict: dict[str, Any],
) -> tuple[str, ...]:
    """Merge gating_baselines with evaluation.comparators (deduped, order preserved)."""

    gates: list[str] = list(config.gating_baselines)
    evaluation = yaml_dict.get("evaluation")
    if isinstance(evaluation, dict):
        comparators = evaluation.get("comparators")
        if isinstance(comparators, list):
            for gate_id in comparators:
                text = str(gate_id)
                if text not in gates:
                    gates.append(text)
    return tuple(gates)


def resolve_meta_router_evaluation_criteria(yaml_dict: dict[str, Any] | None) -> dict[str, Any]:
    """Parse evaluation pass criteria from P2 YAML (defaults: xgboost on fold_2)."""

    evaluation = yaml_dict.get("evaluation") if isinstance(yaml_dict, dict) else None
    if not isinstance(evaluation, dict):
        evaluation = {}
    return {
        "pass_baseline": str(evaluation.get("pass_baseline", "best_base_model")),
        "pass_fold": str(evaluation.get("pass_fold", "fold_2")),
        "evaluate_fold": str(evaluation.get("evaluate_fold", evaluation.get("pass_fold", "fold_2"))),
        "full_contract": bool(evaluation.get("full_contract", False)),
    }


def validate_model_families(config: dict[str, Any]) -> list[str]:
    models = config.get("models")
    if not isinstance(models, list):
        return []
    return [
        resolve_model_family(str(entry["family"]))
        for entry in models
        if isinstance(entry, dict) and "family" in entry
    ]


def load_meta_router_experiment(path: Path | str) -> MetaRouterExperimentSpec:
    return parse_meta_router_experiment(load_p2_config(path))


__all__ = [
    "DEFAULT_BROAD_MODEL_MATRIX_CONFIG",
    "DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG",
    "DEFAULT_META_ROUTER_CONFIG",
    "DEFAULT_PANEL_CONFIG",
    "DEFAULT_REPTILE_CONFIG",
    "DEFAULT_ROUTER_CONFIG",
    "MetaRouterExperimentSpec",
    "RESEARCH_P2_CONFIGS",
    "config_content_hash",
    "load_meta_router_experiment",
    "load_p2_config",
    "parse_meta_router_experiment",
    "resolve_meta_router_battery_gate_ids",
    "resolve_meta_router_evaluation_criteria",
    "resolve_config_path",
    "validate_model_families",
    "yaml_to_meta_router_config",
    "yaml_to_p2_config",
]

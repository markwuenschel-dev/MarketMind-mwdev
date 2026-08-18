"""Configuration objects for W3-B technical-indicator experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _coerce_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"{label} must be numeric, got {type(value).__name__}")
    return float(value)


def _float_from_mapping(
    payload: Mapping[str, object],
    key: str,
    *,
    default: float | None = None,
) -> float:
    if key not in payload:
        if default is None:
            raise KeyError(key)
        return default
    return _coerce_float(payload[key], key)


@dataclass(frozen=True, slots=True)
class IndicatorLibraryConfig:
    """Provider-level settings shared by diagnostics, child policies, and runners."""

    lag_bars: int = 1
    redundancy_rank_corr_threshold: float = 0.85
    panel_float32: bool = True
    save_long_indicator_rows: bool = False
    save_selected_rows_only: bool = True
    robust_indicator_scaling: bool = True
    robust_scale_iqr_multiplier: float = 3.0
    robust_scale_min_iqr: float = 1e-6

    def __post_init__(self) -> None:
        if self.lag_bars < 1:
            raise ValueError("lag_bars must be >= 1")
        if not 0.0 < self.redundancy_rank_corr_threshold < 1.0:
            raise ValueError("redundancy_rank_corr_threshold must be in (0, 1)")
        if self.save_long_indicator_rows:
            raise ValueError("W3-B must not save full long indicator rows")
        if self.robust_scale_iqr_multiplier <= 0.0:
            raise ValueError("robust_scale_iqr_multiplier must be > 0")
        if self.robust_scale_min_iqr <= 0.0:
            raise ValueError("robust_scale_min_iqr must be > 0")


@dataclass(frozen=True, slots=True)
class W3BPandasTAConfig:
    """Top-level W3-B orchestration defaults."""

    output_dir: Path = Path("artifacts/phase_ii/w3_b_pandas_ta")
    surfaces: tuple[str, ...] = (
        "surface_b",
        "surface_c",
        "surface_d",
        "surface_e2_top10",
        "surface_e2_top5",
    )
    xgboost_train_row_cap: int = 50_000
    xgboost_predict_batch_rows: int = 250_000
    use_xgboost: bool = True
    loader_backend: Literal["polars", "pandas"] = "polars"
    workers: int = 1
    xgboost_n_jobs: int = 1
    seed: int = 6203
    timestamp_utc: str = "2026-05-16T00:00:00Z"
    release_input_panel_after_surfaces: bool = True
    low_memory_mode: bool = False
    spill_dir: Path | None = None
    indicator_library: IndicatorLibraryConfig = IndicatorLibraryConfig()
    pipeline_indicator_panel_path: Path | None = None

    def __post_init__(self) -> None:
        if self.xgboost_train_row_cap < 1:
            raise ValueError("xgboost_train_row_cap must be >= 1")
        if self.xgboost_predict_batch_rows < 1:
            raise ValueError("xgboost_predict_batch_rows must be >= 1")
        if self.loader_backend not in {"polars", "pandas"}:
            raise ValueError("loader_backend must be one of {'polars', 'pandas'}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.xgboost_n_jobs < 1:
            raise ValueError("xgboost_n_jobs must be >= 1")
        if not self.surfaces:
            raise ValueError("surfaces must be non-empty")


@dataclass(frozen=True, slots=True)
class PenaltyMultiplierSpec:
    """Score-time liquidity penalty parameters derived from the Surface D audit."""

    formula: str
    reference_liquidity: float
    penalty_term_cap: float
    strength: float
    min_multiplier: float
    calibration: dict[str, float]

    def __post_init__(self) -> None:
        if self.reference_liquidity <= 0.0:
            raise ValueError("reference_liquidity must be > 0")
        if self.penalty_term_cap <= 0.0:
            raise ValueError("penalty_term_cap must be > 0")
        if self.strength <= 0.0:
            raise ValueError("strength must be > 0")
        if not 0.0 <= self.min_multiplier <= 1.0:
            raise ValueError("min_multiplier must be in [0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PenaltyMultiplierSpec:
        calibration_raw = payload.get("calibration", {})
        calibration: dict[str, float] = {}
        if isinstance(calibration_raw, Mapping):
            for key, value in calibration_raw.items():
                calibration[str(key)] = _coerce_float(value, f"calibration[{key!s}]")
        return cls(
            formula=str(payload.get("formula", "clipped_ratio")),
            reference_liquidity=_float_from_mapping(payload, "reference_liquidity"),
            penalty_term_cap=_float_from_mapping(payload, "penalty_term_cap"),
            strength=_float_from_mapping(payload, "strength"),
            min_multiplier=_float_from_mapping(payload, "min_multiplier", default=0.0),
            calibration=calibration,
        )


@dataclass(frozen=True, slots=True)
class W3CLiquidityChildConfig:
    """Top-level W3-C orchestration; penalty spec must come from Surface D audit artifact."""

    output_dir: Path = Path("artifacts/phase_ii/w3_c_liquidity_child")
    surfaces: tuple[str, ...] = (
        "surface_b",
        "surface_c",
        "surface_d",
        "surface_e2_top10",
        "surface_e2_top5",
    )
    surface_d_audit_path: Path = Path(
        "artifacts/phase_ii/w3_c_liquidity_child/audit/surface_d_failure_audit.json"
    )
    w3_b_report_path: Path = Path(
        "artifacts/phase_ii/w3_b_pandas_ta/child_policies/ta_child_policy_report.json"
    )
    penalty_multiplier_spec: PenaltyMultiplierSpec | None = None
    xgboost_train_row_cap: int = 50_000
    xgboost_predict_batch_rows: int = 250_000
    use_xgboost: bool = True
    workers: int = 1
    xgboost_n_jobs: int = 1
    seed: int = 6203
    timestamp_utc: str = "2026-05-16T00:00:00Z"
    indicator_library: IndicatorLibraryConfig = IndicatorLibraryConfig()

    def __post_init__(self) -> None:
        if self.xgboost_train_row_cap < 1:
            raise ValueError("xgboost_train_row_cap must be >= 1")
        if self.xgboost_predict_batch_rows < 1:
            raise ValueError("xgboost_predict_batch_rows must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.xgboost_n_jobs < 1:
            raise ValueError("xgboost_n_jobs must be >= 1")
        if not self.surfaces:
            raise ValueError("surfaces must be non-empty")

    def with_penalty_spec(self, spec: PenaltyMultiplierSpec) -> W3CLiquidityChildConfig:
        return W3CLiquidityChildConfig(
            output_dir=self.output_dir,
            surfaces=self.surfaces,
            surface_d_audit_path=self.surface_d_audit_path,
            w3_b_report_path=self.w3_b_report_path,
            penalty_multiplier_spec=spec,
            xgboost_train_row_cap=self.xgboost_train_row_cap,
            xgboost_predict_batch_rows=self.xgboost_predict_batch_rows,
            use_xgboost=self.use_xgboost,
            workers=self.workers,
            xgboost_n_jobs=self.xgboost_n_jobs,
            seed=self.seed,
            timestamp_utc=self.timestamp_utc,
            indicator_library=self.indicator_library,
        )

    def require_penalty_spec(self) -> PenaltyMultiplierSpec:
        if self.penalty_multiplier_spec is None:
            raise ValueError(
                "W3-C penalty_multiplier_spec is required; run Surface D audit first or "
                "load surface_d_failure_audit.json via load_w3_c_config_from_audit()."
            )
        return self.penalty_multiplier_spec

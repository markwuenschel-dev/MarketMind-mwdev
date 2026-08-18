"""
Pipeline configuration with observability.

This module adds:
- Prometheus metrics for config load latency and errors
- Re-exports all config models and functions

All configuration logic lives in pysrc.pipeline.pipeline_config.loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pysrc.core.runtime.prometheus_metrics import Counter as _Counter
from pysrc.core.runtime.prometheus_metrics import Histogram as _Histogram
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.pipeline_config.loader import (
    ATR,
    # Paths
    ESG,
    FRED,
    MACD,
    # Technical indicators
    RSI,
    VWAP,
    Alerting,
    Alpaca,
    # Alternative data sources
    AlternativeData,
    AnomalyDetection,
    Backtesting,
    Bloomberg,
    Bollinger,
    Calendar,
    # Other sections
    Cleaning,
    CleaningCombo,
    # Preprocessing
    Clip,
    Compliance,
    Credentials,
    CSVSource,
    CustomFeatures,
    DashboardConfig,
    DataAnonymization,
    DataSource,
    DateRange,
    DistributedProcessing,
    Encryption,
    ErrorHandling,
    ESGNormalized,
    Fallback,
    FileOutput,
    InfluxDBOutput,
    InfluxSource,
    InteractiveBrokers,
    Logging,
    MetricAggregation,
    Model,
    ModelArchitecture,
    Normalization,
    Outputs,
    # Models (re-export for backward compatibility)
    PipelineConfig,
    PositionSizing,
    Preprocessing,
    RateLimit,
    RealTimeMarketData,
    RetryPolicy,
    RiskManagement,
    Section,
    Security,
    Sentiment,
    Streaming,
    TechnicalIndicators,
    Twitter,
    ValidationThresholds,
    Weather,
)
from pysrc.pipeline.pipeline_config.loader import (
    get_config as _get_config,
)
from pysrc.pipeline.pipeline_config.loader import (
    get_dataset as _get_dataset,
)
from pysrc.pipeline.pipeline_config.loader import (
    get_runtime_config as _get_runtime_config,
)

# Import EVERYTHING from the single source of truth
from pysrc.pipeline.pipeline_config.loader import (
    # Main API
    load_config as _load_config,
)
from pysrc.pipeline.pipeline_config.loader import (
    reload_config as _reload_config,
)
from pysrc.pipeline.pipeline_config.loader import (
    reset_config_cache as _reset_config_cache,
)
from pysrc.pipeline.pipeline_config.loader import (
    validate_runtime_requirements as _validate_runtime_requirements,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Prometheus Metrics (only in this wrapper layer)
# --------------------------------------------------------------------------------------

_CFG_LATENCY = _Histogram(
    "marketmind_pipeline_config_load_seconds",
    "Latency for loading/validating pipeline config",
    ["phase"],
)

_CFG_ERRORS = _Counter(
    "marketmind_pipeline_config_errors_total",
    "Errors during config load/validation",
    ["phase"],
)


# --------------------------------------------------------------------------------------
# Public API with Telemetry
# --------------------------------------------------------------------------------------


def load_config(
    path: Path | None = None,
    schema_path: Path | None = None,
    *,
    apply_env: bool = True,
    env_prefix: str = "MARKETMIND__",
    list_strategy: str = "replace",
) -> PipelineConfig:
    """
    Load and validate pipeline configuration with telemetry.

    See pysrc.pipeline.pipeline_config.load_config for full documentation.
    """
    with _CFG_LATENCY.labels(phase="total").time():
        try:
            return _load_config(
                path=path,
                schema_path=schema_path,
                apply_env=apply_env,
                env_prefix=env_prefix,
                list_strategy=list_strategy,
            )
        except FileNotFoundError:
            _CFG_ERRORS.labels(phase="fs").inc()
            raise
        except Exception as e:
            # Determine error phase from exception type
            if "YAML" in str(e):
                _CFG_ERRORS.labels(phase="yaml").inc()
            elif "schema" in str(e).lower():
                _CFG_ERRORS.labels(phase="jsonschema").inc()
            elif "Pydantic" in str(e) or "validation" in str(e).lower():
                _CFG_ERRORS.labels(phase="pydantic").inc()
            else:
                _CFG_ERRORS.labels(phase="unknown").inc()
            raise


def get_config(path: Path | None = None) -> PipelineConfig:
    """
    Get cached config singleton with telemetry.

    See pysrc.pipeline.pipeline_config.get_config for full documentation.
    """
    # Only measure on first load (singleton returns immediately after)
    try:
        return _get_config(path)
    except Exception:
        _CFG_ERRORS.labels(phase="singleton").inc()
        raise


def reload_config(path: Path | None = None) -> PipelineConfig:
    """
    Force reload config with telemetry.

    See pysrc.pipeline.pipeline_config.reload_config for full documentation.
    """
    with _CFG_LATENCY.labels(phase="reload").time():
        try:
            return _reload_config(path)
        except Exception:
            _CFG_ERRORS.labels(phase="reload").inc()
            raise


def reset_config_cache() -> None:
    """
    Clear config cache (used by tests).

    See pysrc.pipeline.pipeline_config.reset_config_cache for full documentation.
    """
    _reset_config_cache()


def get_runtime_config() -> PipelineConfig:
    """
    Get a fresh copy of the config.

    See pysrc.pipeline.pipeline_config.get_runtime_config for full documentation.
    """
    return _get_runtime_config()


def get_dataset(**kwargs) -> Any:
    """
    Load dataset as Polars DataFrame from configured data source.

    See pysrc.pipeline.pipeline_config.get_dataset for full documentation.
    """
    try:
        return _get_dataset(**kwargs)
    except Exception:
        _CFG_ERRORS.labels(phase="dataset").inc()
        raise


def validate_runtime_requirements(conf: PipelineConfig | None = None) -> list[str]:
    """
    Validate runtime requirements and return list of issues.

    See pysrc.pipeline.pipeline_config.validate_runtime_requirements for full documentation.
    """
    return _validate_runtime_requirements(conf)


# --------------------------------------------------------------------------------------
# Legacy Compatibility
# --------------------------------------------------------------------------------------


# Some tests may still reference these old patterns
def get_config_singleton():
    """Legacy: Use get_config() instead."""
    import warnings

    warnings.warn(
        "get_config_singleton() is deprecated, use get_config()", DeprecationWarning, stacklevel=2
    )
    return get_config()


__all__ = [
    # Main API
    "load_config",
    "get_config",
    "reload_config",
    "reset_config_cache",
    "get_runtime_config",
    "get_dataset",
    "validate_runtime_requirements",
    # Main model
    "PipelineConfig",
    # All section models
    "Section",
    "DataSource",
    "CSVSource",
    "InfluxSource",
    "RSI",
    "MACD",
    "ATR",
    "Bollinger",
    "VWAP",
    "TechnicalIndicators",
    "Clip",
    "Normalization",
    "Calendar",
    "Sentiment",
    "ESGNormalized",
    "CustomFeatures",
    "Preprocessing",
    "Cleaning",
    "CleaningCombo",
    "Streaming",
    "ErrorHandling",
    "RetryPolicy",
    "ValidationThresholds",
    "Fallback",
    "Alerting",
    "Model",
    "ModelArchitecture",
    "Logging",
    "FileOutput",
    "InfluxDBOutput",
    "Outputs",
    "MetricAggregation",
    "DashboardConfig",
    "Security",
    "Encryption",
    "Credentials",
    "Compliance",
    "DataAnonymization",
    "Backtesting",
    "RiskManagement",
    "DateRange",
    "PositionSizing",
    "DistributedProcessing",
    "AlternativeData",
    "Twitter",
    "Alpaca",
    "ESG",
    "FRED",
    "Bloomberg",
    "Weather",
    "RateLimit",
    "RealTimeMarketData",
    "InteractiveBrokers",
    "AnomalyDetection",
]

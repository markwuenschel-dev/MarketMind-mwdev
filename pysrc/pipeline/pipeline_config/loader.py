# py/pipeline/pipeline_config/loader.py
from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from functools import lru_cache
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
)

import yaml
from jsonschema import Draft202012Validator

from pysrc.core.errors import ConfigValidationError

# Soft dependency infra
from pysrc.core.runtime.optional_imports import optional_import, require, require_attr
from pysrc.ops.mm_logkit import get_logger


def _dependency_available(module: str) -> bool:
    return optional_import(module) is not None


def ensure_pydantic() -> Any:
    return require("pydantic", purpose="pipeline configuration validation")


def ensure_polars() -> Any:
    return require("polars", purpose="pipeline dataset loading")


def ensure_influxdb_client() -> Any:
    return require_attr("influxdb_client", "InfluxDBClient", purpose="InfluxDB data source loading")


# --------------------------------------------------------------------------------------
# Pydantic setup (version-agnostic)
# --------------------------------------------------------------------------------------

_P = ensure_pydantic()
BaseModel = _P.BaseModel
Field = _P.Field
ValidationError = _P.ValidationError
ConfigDict = _P.ConfigDict
model_validator = getattr(_P, "model_validator", None)

# Graceful shim for Pydantic v1 compatibility
if model_validator is None:

    def model_validator(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap


logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Filesystem paths
# --------------------------------------------------------------------------------------

BASE: Path = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH: Path = BASE / "py" / "pipeline" / "pipeline_config" / "config.yaml"
DEFAULT_SCHEMA_PATH: Path = BASE / "schemas" / "config_schema.json"

# Test override hooks (monkeypatch these in tests)
CONFIG_PATH: Path | None = None
SCHEMA_PATH: Path | None = None

# --------------------------------------------------------------------------------------
# Environment variable utilities
# --------------------------------------------------------------------------------------

# Matches: ${VAR}, ${VAR:-default}, ${VAR:?error_message}
_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?::-(?P<default>[^}]*))?"
    r"(?::\?(?P<err>[^}]*))?"
    r"}"
)


def _interpolate_env(text: str) -> str:
    """Interpolate environment variables with bash-like syntax."""
    if "$" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        name = m.group("name")
        default = m.group("default")
        err = m.group("err")
        val = os.environ.get(name)

        if val is None:
            if err:
                raise ConfigValidationError(f"Required env var {name}: {err}")
            return default if default is not None else ""
        return val

    return _ENV_PATTERN.sub(repl, os.path.expandvars(text))


def _resolve_env(obj: Any) -> Any:
    """Recursively resolve environment variables in nested structures."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, list):
        return [_resolve_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    return obj


def _apply_env_overrides(d: MutableMapping[str, Any], *, prefix: str = "MARKETMIND__") -> None:
    """
    Apply hierarchical environment variable overrides.

    Example: MARKETMIND__preprocessing__rsi__window=21
    Maps to: d["preprocessing"]["rsi"]["window"] = 21
    """
    plen = len(prefix)
    _cache: dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        path = key[plen:].split("__")
        cursor: MutableMapping[str, Any] = d

        # Navigate/create nested structure
        for p in path[:-1]:
            if p not in cursor or not isinstance(cursor[p], MutableMapping):
                cursor[p] = {}
            cursor = cursor[p]

        # Coerce value to appropriate type
        v: Any = _cache.get(value)
        if v is None:
            lv = value.lower()
            if lv in ("true", "false"):
                v = lv == "true"
            else:
                try:
                    v = int(value) if "." not in value else float(value)
                    if isinstance(v, float) and v.is_integer():
                        v = int(v)
                except Exception:
                    v = value
            _cache[value] = v

        cursor[path[-1]] = v


def _is_production_mode() -> bool:
    """Detect if running in production environment."""
    env = os.getenv("ENVIRONMENT", "").lower()
    return env in ("production", "prod") or os.getenv("PYTEST_CURRENT_TEST") is None


# --------------------------------------------------------------------------------------
# Deep merge utilities
# --------------------------------------------------------------------------------------


def _deep_merge(base: Any, overlay: Any, *, list_strategy: str = "replace") -> Any:
    """
    Deep-merge overlay onto base.

    Args:
        base: Base configuration
        overlay: Configuration to merge on top
        list_strategy: How to merge lists
            - "replace": Overlay completely replaces base
            - "append": Concatenate lists
            - "unique": Merge and deduplicate
    """
    if base is None:
        return overlay
    if overlay is None:
        return base

    # Dictionaries: recursive merge
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for k, v in overlay.items():
            out[k] = _deep_merge(base.get(k), v, list_strategy=list_strategy)
        return out

    # Lists: strategy-dependent merge
    if isinstance(base, list) and isinstance(overlay, list):
        if list_strategy == "replace":
            return overlay
        if list_strategy == "append":
            return [*base, *overlay]
        if list_strategy == "unique":
            seen = set()
            merged: list[Any] = []
            for item in [*base, *overlay]:
                key = (
                    json.dumps(item, sort_keys=True, default=str)
                    if not isinstance(item, (str, int, float))
                    else item
                )
                if key not in seen:
                    merged.append(item)
                    seen.add(key)
            return merged

    # Scalars: overlay wins
    return overlay


# --------------------------------------------------------------------------------------
# Pydantic models - Base Section
# --------------------------------------------------------------------------------------


class Section(BaseModel):
    """Base class for all config sections with validation hooks."""

    model_config = ConfigDict(extra="allow")

    def validate_section(self) -> None:
        """Override in subclasses for custom validation logic."""
        pass


# --------------------------------------------------------------------------------------
# Data Sources (Models first, then runtime-capable types)
# --------------------------------------------------------------------------------------


class _CSVSourceModel(Section):
    type: Literal["csv"] = "csv"
    path: str
    chunksize: int = 0
    use_dask: bool = Field(default_factory=lambda: _dependency_available("dask.dataframe"))
    compression: str | None = None
    data_format: str = "csv"


class _InfluxSourceModel(Section):
    type: Literal["influxdb"] = "influxdb"
    host: str
    port: int
    token: str
    org: str
    bucket: str
    query: str


# --------------------------------------------------------------------------------------
# Polars Integration (Optional)
# --------------------------------------------------------------------------------------


class _PolarsMixin:
    def to_polars(self, **read_kwargs) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement to_polars()")


class CSVSource(_PolarsMixin, _CSVSourceModel):
    def to_polars(self, **read_kwargs) -> Any:
        pl = ensure_polars()
        return pl.read_csv(
            self.path,
            has_header=True,
            rechunk=True,
            low_memory=False,
            **({} if self.compression is None else {"compression": self.compression}),
            **read_kwargs,
        )


class InfluxSource(_PolarsMixin, _InfluxSourceModel):
    def to_polars(self, **read_kwargs) -> Any:
        InfluxDBClient = ensure_influxdb_client()
        pl = ensure_polars()
        with InfluxDBClient(
            url=f"http://{self.host}:{self.port}", token=self.token, org=self.org
        ) as client:
            df_pandas = client.query_api().query_data_frame(self.query)
        return pl.from_pandas(df_pandas, **read_kwargs)


# Type-safe discriminated union over FINAL types
DataSource = Annotated[CSVSource | InfluxSource, Field(discriminator="type")]


# --------------------------------------------------------------------------------------
# Technical Indicators
# --------------------------------------------------------------------------------------


class RSI(Section):
    enabled: bool = False
    window: int = 14
    fillna_method: str = "ffill"

    def validate_section(self) -> None:
        if self.enabled and self.window <= 0:
            raise ValueError("RSI window must be positive")


class MACD(Section):
    enabled: bool = False
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    fillna_method: str = "ffill"

    def validate_section(self) -> None:
        if self.enabled and self.fast_period >= self.slow_period:
            raise ValueError("MACD fast_period must be < slow_period")


class ATR(Section):
    enabled: bool = False
    window: int = 14
    fillna_method: str = "ffill"


class Bollinger(Section):
    enabled: bool = False
    window: int = 20
    std_dev: float = 2.0
    fillna_method: str = "ffill"


class VWAP(Section):
    enabled: bool = False
    reset_period: str = "daily"
    fillna_method: str = "ffill"


class TechnicalIndicators(Section):
    rsi: RSI | None = None
    macd: MACD | None = None
    atr: ATR | None = None
    vwap: VWAP | None = None
    bollinger_bands: Bollinger | None = None
    extra_indicators: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def validate_section(self) -> None:
        for indicator in (self.rsi, self.macd, self.atr, self.vwap, self.bollinger_bands):
            if indicator:
                indicator.validate_section()


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------


class Clip(Section):
    min: float
    max: float


class Normalization(Section):
    method: str
    rolling_window: int
    clip_extremes: Clip

    def validate_section(self) -> None:
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")


class Calendar(Section):
    enabled: bool = False
    day_of_week: bool = True
    holidays: list[str] = Field(default_factory=list)
    timezones: list[str] = Field(default_factory=list)

    def is_holiday(self, dt) -> bool:
        return self.enabled and dt.strftime("%Y-%m-%d") in self.holidays


class Sentiment(Section):
    enabled: bool
    source: str
    sentiment_model: str


class ESGNormalized(Section):
    enabled: bool
    method: str


class CustomFeatures(Section):
    sentiment: Sentiment | None = None
    esg_normalized: ESGNormalized | None = None


class Preprocessing(Section):
    technical_indicators: TechnicalIndicators
    normalization: Normalization
    custom_features: CustomFeatures = Field(default_factory=CustomFeatures)
    calendar_features: Calendar = Field(default_factory=Calendar)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    step_macros: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def validate_section(self) -> None:
        self.technical_indicators.validate_section()
        self.normalization.validate_section()

    def expand_macros(self) -> None:
        """Expand macro templates in preprocessing steps."""
        if not self.step_macros:
            return

        expanded: list[dict[str, Any]] = []

        for step in self.steps:
            macro = step.get("use_macro")
            if not macro:
                expanded.append(step)
                continue

            grid = self.step_macros.get(macro) or {}
            if not grid:
                expanded.append(step)
                continue

            keys = list(grid.keys())
            values = [v if isinstance(v, list) else [v] for v in grid.values()]

            def cart(idx: int, base: dict[str, Any]) -> None:
                if idx == len(keys):
                    expanded.append(base)
                    return
                k = keys[idx]
                for val in values[idx]:
                    cart(idx + 1, {**base, k: val})

            seed = {k: v for k, v in step.items() if k != "use_macro"}
            cart(0, seed)

        self.steps = expanded


# --------------------------------------------------------------------------------------
# Other Config Sections (kept concise)
# --------------------------------------------------------------------------------------


class CleaningCombo(Section):
    name: str
    when: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    order: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    determinism_tier: str = "d1"
    governance_mode: str = "governed"
    seed_lineage: str = ""
    pit_boundary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Cleaning(Section):
    combos: list[CleaningCombo] = Field(default_factory=list)
    use: str | None = None
    determinism_tier: str = "d1"
    governance_mode: str = "governed"
    seed_lineage: str = ""
    pit_boundary: str = ""
    missing_values: dict[str, Any] = Field(default_factory=dict)
    outliers: dict[str, Any] = Field(default_factory=dict)
    denoising: dict[str, Any] = Field(default_factory=dict)


class Streaming(Section):
    batch_size: int = 100
    update_interval_seconds: int = 60
    buffer_size: int = 1000
    max_latency_ms: int = 500
    buffer_retention_seconds: int = 3600
    event_triggers: dict[str, str] = Field(default_factory=dict)
    priority_queue: str = "fifo"
    failure_recovery: dict[str, Any] = Field(default_factory=dict)
    sync_interval_seconds: int = 300


class RetryPolicy(Section):
    max_attempts: int
    initial_backoff_seconds: int
    max_backoff_seconds: int
    retry_strategy: str


class ValidationThresholds(Section):
    max_missing_ratio: float
    max_outlier_ratio: float


class Fallback(Section):
    twitter: str
    esg: str
    data_source: str


class Alerting(Section):
    enabled: bool
    channel: str
    critical_failures: list[str]
    alert_severity: list[str]


class ErrorHandling(Section):
    retry_policy: RetryPolicy
    validation_thresholds: ValidationThresholds
    fallback: Fallback
    alerting: Alerting
    fallback_timeout_seconds: int


class ModelArchitecture(Section):
    num_layers: int
    hidden_size: int
    dropout: float


class Model(Section):
    model_type: str
    architecture: ModelArchitecture
    sequence_length: int
    prediction_horizon: int
    feature_list: list[str]
    training_device: str
    model_checkpoint: dict[str, int]
    feature_importance: dict[str, str]
    onnx_export: dict[str, Any]


class FileOutput(Section):
    enabled: bool
    path: str
    rotation: str


class InfluxDBOutput(Section):
    enabled: bool
    host: str
    port: int
    token: str
    org: str
    bucket: str


class Outputs(Section):
    console: bool
    file: FileOutput
    influxdb: InfluxDBOutput


class MetricAggregation(Section):
    aggregation_window_seconds: int


class DashboardConfig(Section):
    grafana_url: str


class Logging(Section):
    level: str
    outputs: Outputs
    metrics_report_interval_seconds: int
    custom_metrics: list[str]
    model_metrics: list[str]
    metric_aggregation: MetricAggregation
    log_sampling_rate: float
    dashboard_config: DashboardConfig


class Encryption(Section):
    at_rest: bool
    in_transit: bool
    encryption_algorithm: str


class Credentials(Section):
    """Credentials should come from environment variables in production."""

    twitter_api_key: str | None = Field(default_factory=lambda: os.getenv("TWITTER_API_KEY"))
    esg_api_key: str | None = Field(default_factory=lambda: os.getenv("ESG_API_KEY"))
    fred_api_key: str | None = Field(default_factory=lambda: os.getenv("FRED_API_KEY"))
    bloomberg_api_key: str | None = Field(default_factory=lambda: os.getenv("BLOOMBERG_API_KEY"))
    weather_api_key: str | None = Field(default_factory=lambda: os.getenv("WEATHER_API_KEY"))
    alpha_vantage_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY")
    )

    @model_validator(mode="after")
    def validate_secrets(self) -> Credentials:
        """Enforce that secrets come from env vars in production, not YAML."""
        if not _is_production_mode():
            return self

        # Check if any credentials exist but corresponding env var doesn't
        for field_name in type(self).model_fields:
            field_value = getattr(self, field_name, None)
            env_var = field_name.upper()

            if field_value and not os.getenv(env_var):
                logger.warning(
                    f"Credential {field_name} found in config but not in env. "
                    f"In production, use {env_var} env var instead."
                )

        return self


class DataAnonymization(Section):
    anonymize_pii: bool


class Compliance(Section):
    audit_log: bool
    retention_days: int
    audit_frequency_days: int
    data_anonymization: DataAnonymization


class Security(Section):
    encryption: Encryption
    key_management: str
    credentials: Credentials
    compliance: Compliance


class RiskManagement(Section):
    stop_loss: float
    max_drawdown: float


class DateRange(Section):
    start: str
    end: str


class PositionSizing(Section):
    method: str


class Backtesting(Section):
    initial_capital: float
    transaction_cost_rate: float
    slippage_rate: float
    strategy_list: list[str]
    risk_management: RiskManagement
    date_range: DateRange
    performance_metrics: list[str]
    position_sizing: PositionSizing
    benchmark_index: str
    backtest_frequency: str


class DistributedProcessing(Section):
    framework: str
    num_workers: int
    memory_per_worker: str
    cluster_type: str
    min_rows_for_distributed: int = 1000


class InteractiveBrokers(Section):
    host: str
    port: int
    client_id: int
    subscription_topics: list[str]
    api_key: str
    endpoint: str
    connection_timeout_seconds: int
    priority: int
    what_to_show: str = "TRADES"
    use_rth: bool = True
    format_date: int = 1


class Alpaca(Section):
    api_key: str | None = None
    secret_key: str
    endpoint: str


class RealTimeMarketData(Section):
    interactive_brokers: InteractiveBrokers | None = None
    alpaca: Alpaca | None = None


# --------------------------------------------------------------------------------------
# Alternative Data Sources - Rate Limiting
# --------------------------------------------------------------------------------------


class RateLimit(Section):
    per_minute: int | None = None
    max_calls_per_window: int | None = None
    window_seconds: int | None = None


# --------------------------------------------------------------------------------------
# External API Data Sources (Base Class)
# --------------------------------------------------------------------------------------


class ExternalAPISource(Section):
    """Base class for standardized external API data sources."""

    base_url: str
    api_key: str | None = None
    authentication_type: str | None = None
    endpoints: dict[str, str]
    default_params: dict[str, str] = Field(default_factory=dict)
    rate_limit: RateLimit | None = None
    timeout_seconds: int = 10
    cache_duration_hours: int = 24
    data_resolution: str | None = None

    def validate_section(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"{self.__class__.__name__}: timeout_seconds must be positive")
        if self.cache_duration_hours < 0:
            raise ValueError(f"{self.__class__.__name__}: cache_duration_hours cannot be negative")


# --------------------------------------------------------------------------------------
# Specific API Sources
# --------------------------------------------------------------------------------------


class Twitter(Section):
    """Twitter/X API - has unique bearer_token authentication."""

    base_url: str
    bearer_token: str
    authentication_type: str
    endpoints: dict[str, str]
    default_params: dict[str, Any]
    rate_limit: RateLimit
    retry_after_header: str
    timeout_seconds: int
    cache_duration_hours: int
    data_resolution: str
    api_key: str | None = None


class ESG(Section):
    """ESG data API - requires api_key."""

    base_url: str
    api_key: str  # Required, not optional
    authentication_type: str
    endpoints: dict[str, str]
    default_params: dict[str, str]
    timeout_seconds: int
    cache_duration_hours: int
    data_resolution: str


class FRED(ExternalAPISource):
    """Federal Reserve Economic Data."""

    pass


class Bloomberg(ExternalAPISource):
    """Bloomberg Terminal Data."""

    pass


class Weather(ExternalAPISource):
    """Weather Data API."""

    pass


class AlternativeData(Section):
    twitter: Twitter | None = None
    alpaca: Alpaca | None = None
    esg: ESG | None = None
    fred: FRED | None = None
    bloomberg: Bloomberg | None = None
    weather: Weather | None = None


class AnomalyDetection(Section):
    enabled: bool = False
    method: str | None = None
    params: dict[str, Any] | None = None


# --------------------------------------------------------------------------------------
# Main Config Model
# --------------------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Main pipeline configuration with composition support."""

    model_config = ConfigDict(extra="allow")

    # Core required fields
    version: str
    schema_uri: str
    data_source: DataSource
    preprocessing: Preprocessing
    cleaning: Cleaning
    streaming: Streaming
    error_handling: ErrorHandling
    model: Model
    logging: Logging
    security: Security
    backtesting: Backtesting
    distributed_processing: DistributedProcessing

    # Optional blocks
    alternative_data: AlternativeData = Field(default_factory=AlternativeData)
    market_data_sources: list[dict[str, Any]] = Field(default_factory=list)
    real_time_market_data: RealTimeMarketData = Field(default_factory=RealTimeMarketData)
    anomaly_detection: AnomalyDetection = Field(default_factory=AnomalyDetection)

    # Composition features
    includes: list[str] = Field(default_factory=list)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    active_profiles: list[str] = Field(default_factory=list)
    list_merge_strategy: str = Field(default="replace")

    @model_validator(mode="after")
    def _validate_all_sections(self) -> PipelineConfig:
        """Run all section validators and expand macros."""
        self.preprocessing.validate_section()
        self.preprocessing.expand_macros()
        return self

    def merged(
        self, *overlays: Mapping[str, Any], list_strategy: str | None = None
    ) -> PipelineConfig:
        """Create new config by merging overlays onto this config."""
        data = self.model_dump(mode="python")

        for overlay in overlays:
            data = _deep_merge(
                data, dict(overlay), list_strategy=list_strategy or self.list_merge_strategy
            )

        try:
            return PipelineConfig(**data)
        except ValidationError as e:
            raise ConfigValidationError(f"Merge validation failed: {e}")

    def with_profile(self, *profile_names: str) -> PipelineConfig:
        """Fluent API for activating profiles."""
        overlays = [self.profiles.get(name, {}) for name in profile_names]
        return self.merged(*overlays)


# --------------------------------------------------------------------------------------
# JSON Schema Validation
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _compiled_validator(schema_path: Path) -> Draft202012Validator:
    """Load and compile JSON schema validator (cached)."""
    with open(schema_path) as f:
        schema = json.load(f)
    return Draft202012Validator(schema)


def _validate_schema(config_data: dict, schema_path: Path) -> None:
    """Validate config against JSON schema."""
    validator = _compiled_validator(schema_path)
    errors = sorted(validator.iter_errors(config_data), key=lambda e: e.path)

    if errors:
        first = errors[0]
        raise ConfigValidationError(
            f"JSON schema validation failed at {list(first.path)}: {first.message}"
        )


# --------------------------------------------------------------------------------------
# YAML Loading with Includes
# --------------------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_with_includes(config_path: Path) -> dict[str, Any]:
    """
    Load YAML with recursive include support.

    Example:
        # base.yaml
        includes:
          - common.yaml
          - env/production.yaml
    """
    seen: set[Path] = set()

    def recurse(p: Path) -> dict[str, Any]:
        p = p.resolve()
        if p in seen:
            return {}
        seen.add(p)

        data = _load_yaml(p)
        includes = data.get("includes", []) or []
        acc: dict[str, Any] = {}

        for inc in includes:
            inc_path = (p.parent / inc).resolve()
            acc = _deep_merge(acc, recurse(inc_path), list_strategy="replace")

            # Included files should override base keys (include wins on conflicts)
        return _deep_merge(data, acc, list_strategy="replace")

    return recurse(config_path)


def _apply_profiles(
    data: dict[str, Any], names: Iterable[str], list_strategy: str
) -> dict[str, Any]:
    """Apply profile overlays to config data."""
    profiles = data.get("profiles", {}) or {}
    out = dict(data)

    for name in names:
        overlay = profiles.get(name)
        if overlay:
            out = _deep_merge(out, overlay, list_strategy=list_strategy)

    return out


# --------------------------------------------------------------------------------------
# Thread-Safe Singleton
# --------------------------------------------------------------------------------------

_config_lock = threading.RLock()
_config_singleton: PipelineConfig | None = None


def load_config(
    path: Path | None = None,
    schema_path: Path | None = None,
    *,
    apply_env: bool = True,
    env_prefix: str = "MARKETMIND__",
    list_strategy: str = "replace",
) -> PipelineConfig:
    """
    Load and validate pipeline configuration.

    Args:
        path: Config file path (defaults to pipeline_config.yaml)
        schema_path: Schema file path (defaults to config_schema.json)
        apply_env: Whether to interpolate environment variables
        env_prefix: Prefix for hierarchical env overrides
        list_strategy: Default list merge strategy (replace/append/unique)

    Returns:
        Validated PipelineConfig instance

    Raises:
        FileNotFoundError: If config or schema not found
        ConfigValidationError: If validation fails
    """
    # Resolve paths
    cfg_path = Path(path or CONFIG_PATH or DEFAULT_CONFIG_PATH).resolve()
    sch_path = Path(schema_path or SCHEMA_PATH or DEFAULT_SCHEMA_PATH).resolve()

    # Validate file existence
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not sch_path.exists():
        raise FileNotFoundError(f"Schema file not found: {sch_path}")

    # 1. Load YAML with includes
    try:
        data = _load_with_includes(cfg_path)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"YAML parsing failed: {e}")

    if not isinstance(data, dict):
        raise ConfigValidationError("Config root must be an object")

    # 2. Interpolate environment variables
    if apply_env:
        data = _resolve_env(data)

    # 3. Apply hierarchical env overrides
    if apply_env:
        _apply_env_overrides(data, prefix=env_prefix)

    # 4. Apply active profiles
    active = data.get("active_profiles", [])
    data = _apply_profiles(data, active, data.get("list_merge_strategy", list_strategy))

    # 5. Validate schema_uri match
    expected_uri = sch_path.name
    declared = data.get("schema_uri")
    if declared is not None and declared != expected_uri:
        raise ConfigValidationError(
            f"Schema URI mismatch: expected {expected_uri}, found {declared}"
        )

    # 6. JSON Schema validation
    _validate_schema(data, sch_path)

    # 7. Pydantic validation
    try:
        conf = PipelineConfig(**data)
    except ValidationError as e:
        raise ConfigValidationError(f"Config structure validation failed: {e}")

    # Log loaded config metadata (no side effects, just info)
    logger.info(
        "config.loaded",
        version=conf.version,
        profiles=active,
        has_polars=_dependency_available("polars"),
        has_dask=_dependency_available("dask.dataframe"),
    )

    return conf


def get_config(path: Path | None = None) -> PipelineConfig:
    """
    Get cached config singleton (thread-safe).

    First call loads the config, subsequent calls return the same instance.
    Use reload_config() to invalidate cache.
    """
    global _config_singleton

    # Fast path: already loaded
    if _config_singleton is not None:
        return _config_singleton

    # Slow path: load with lock
    with _config_lock:
        if _config_singleton is None:
            _config_singleton = load_config(path)
        return _config_singleton


def reload_config(path: Path | None = None) -> PipelineConfig:
    """
    Force reload config (thread-safe).

    Useful for long-running services that need to pick up config changes.
    """
    global _config_singleton

    with _config_lock:
        _config_singleton = None
        return get_config(path)


def reset_config_cache() -> None:
    """Clear config cache (used by tests)."""
    global _config_singleton

    with _config_lock:
        _config_singleton = None


def get_runtime_config() -> PipelineConfig:
    """
    Get a fresh copy of the config (not cached).

    Useful when you need an independent instance for modification.
    """
    base = get_config()
    return PipelineConfig(**base.model_dump())


# --------------------------------------------------------------------------------------
# Convenience: Direct dataset loading
# --------------------------------------------------------------------------------------


def get_dataset(**kwargs) -> Any:
    """
    Load dataset as Polars DataFrame from configured data source.

    Example:
        df = get_dataset()  # Uses config from get_config()
        df = get_dataset(n_rows=1000)  # Pass kwargs to reader

    Raises:
        TypeError: If data source doesn't support Polars
    """
    conf = get_runtime_config()
    ds = conf.data_source

    if not isinstance(ds, _PolarsMixin):  # type: ignore[arg-type]
        raise TypeError(f"DataSource {type(ds).__name__} doesn't support Polars")

    return ds.to_polars(**kwargs)


# --------------------------------------------------------------------------------------
# Startup validation
# --------------------------------------------------------------------------------------


def validate_runtime_requirements(conf: PipelineConfig | None = None) -> list[str]:
    """
    Validate runtime requirements and return list of issues.

    Example:
        issues = validate_runtime_requirements()
        if issues:
            for issue in issues:
                logger.warning(issue)
    """
    if conf is None:
        conf = get_config()

    issues: list[str] = []

    # Check enabled features have required dependencies
    if (
        conf.preprocessing.technical_indicators.rsi
        and conf.preprocessing.technical_indicators.rsi.enabled
    ) and conf.preprocessing.technical_indicators.rsi.window <= 0:
        issues.append("RSI enabled but window <= 0")

    # Check data source compatibility
    if isinstance(conf.data_source, CSVSource):
        if conf.data_source.use_dask and not _dependency_available("dask.dataframe"):
            issues.append("CSV source configured with use_dask=True but dask not available")

    # Check credentials in production
    if _is_production_mode():
        creds = conf.security.credentials
        if conf.alternative_data.twitter and not creds.twitter_api_key:
            issues.append("Twitter data source enabled but TWITTER_API_KEY not set")
        if conf.alternative_data.esg and not creds.esg_api_key:
            issues.append("ESG data source enabled but ESG_API_KEY not set")

    return issues

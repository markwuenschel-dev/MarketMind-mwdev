"""W3-B pandas-ta-classic indicator provider and diagnostics."""

from pysrc.pipeline.stages.preprocessing.indicators.config import (
    IndicatorLibraryConfig,
    W3BPandasTAConfig,
)
from pysrc.pipeline.stages.preprocessing.indicators.diagnostics import (
    IndicatorDiagnosticsResult,
    build_indicator_diagnostics,
    lag_outcome_diagnostic_by_horizon,
    prune_redundant_indicators,
)
from pysrc.pipeline.stages.preprocessing.indicators.pandas_ta_classic_provider import (
    IndicatorProviderResult,
    compute_pandas_ta_classic_features,
)
from pysrc.pipeline.stages.preprocessing.indicators.registry import (
    IndicatorDefinition,
    indicator_config_payload,
    indicator_definitions,
    indicator_ids,
)
from pysrc.pipeline.stages.preprocessing.indicators.scaling import (
    IndicatorRobustScale,
    apply_robust_indicator_scales,
    fit_robust_indicator_scales,
    robust_scale_metadata_payload,
    sanitize_indicator_series,
)

__all__ = [
    "IndicatorDefinition",
    "IndicatorDiagnosticsResult",
    "IndicatorLibraryConfig",
    "IndicatorProviderResult",
    "IndicatorRobustScale",
    "W3BPandasTAConfig",
    "apply_robust_indicator_scales",
    "build_indicator_diagnostics",
    "compute_pandas_ta_classic_features",
    "fit_robust_indicator_scales",
    "indicator_config_payload",
    "indicator_definitions",
    "indicator_ids",
    "lag_outcome_diagnostic_by_horizon",
    "prune_redundant_indicators",
    "robust_scale_metadata_payload",
    "sanitize_indicator_series",
]

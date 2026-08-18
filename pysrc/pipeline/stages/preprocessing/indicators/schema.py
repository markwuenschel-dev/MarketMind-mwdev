"""Schema constants for the W3-B pandas-ta-classic indicator surface."""

from __future__ import annotations

from typing import Final, Literal

W3B_INDICATOR_SCHEMA_VERSION: Final[str] = "w3_b.pandas_ta.indicator_library.v1"
W3B_PROVIDER_NAME: Final[str] = "pandas-ta-classic"

IndicatorCategory = Literal["momentum", "trend", "volatility", "volume_liquidity", "risk_state"]
IndicatorClassification = Literal[
    "KEEP",
    "INVERT",
    "REGIME_ONLY",
    "LIQUIDITY_ONLY",
    "DROP",
    "TOO_UNSTABLE",
    "INCONCLUSIVE",
]

REQUIRED_PROVIDER_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "instrument",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
    "raw_close",
    "raw_volume",
)

W3B_INDICATOR_IDS: Final[tuple[str, ...]] = (
    "rsi_14",
    "roc_5",
    "roc_10",
    "roc_20",
    "stoch_k_14_3",
    "stoch_d_14_3",
    "williams_r_14",
    "cci_20",
    "macd_12_26_9",
    "macd_hist_12_26_9",
    "adx_14",
    "dmp_14",
    "dmn_14",
    "ema_distance_20",
    "ema_distance_50",
    "sma_cross_20_50",
    "atr_14",
    "natr_14",
    "bb_percent_b_20_2",
    "bb_bandwidth_20_2",
    "keltner_width_20",
    "donchian_position_20",
    "obv_change_20",
    "mfi_14",
    "volume_zscore_20",
    "volume_ratio_20",
    "volume_dollar_zscore_20",
    "dollar_volume_ratio_20",
    "drawdown_60",
    "realized_volatility_20",
    "range_ratio_10",
)

W3B_TA_CHILD_POLICY_IDS: Final[tuple[str, ...]] = (
    "ta_equal_weight_signal_allocator",
    "ta_orientation_weighted_allocator",
    "ta_regime_conditioned_allocator",
    "ta_liquidity_conditioned_allocator",
    "ta_xgb_ta",
    "ta_xgb_base_plus_ta",
    "ta_simple_ensemble_allocator",
)

W3B_CANONICAL_XGB_IDS: Final[tuple[str, str, str]] = ("xgb_base", "xgb_ta", "xgb_base_plus_ta")

W3B_REFERENCE_CHILD_POLICY_IDS: Final[tuple[str, ...]] = (
    "static_equal_weight_signal_allocator",
    "rolling_ic_weighted_signal_allocator",
    "regime_conditioned_signal_gate_allocator",
    "simple_ensemble_allocator",
    "xgb_base",
    "abstain_cash_policy",
)

W3B_ALLOWED_CLASSIFICATIONS: Final[tuple[str, ...]] = (
    "TA_IMPROVES_CHILD_POLICIES_AND_ROUTER",
    "TA_IMPROVES_CHILD_POLICIES_ONLY",
    "TA_IMPROVES_XGBOOST_ONLY",
    "TA_NO_MATERIAL_IMPROVEMENT",
    "TA_RESULT_CONCENTRATED",
    "INCONCLUSIVE",
)

W3C_SURFACE_D_AUDIT_SCHEMA_VERSION: Final[str] = "w3_c.surface_d_failure_audit.v1"
W3C_LIQUIDITY_CHILD_REPORT_SCHEMA_VERSION: Final[str] = "w3_c.liquidity_child.tradeoff_report.v2"
W3C_LIQUIDITY_CHILD_POLICY_REPORT_SCHEMA_VERSION: Final[str] = (
    "w3_c.liquidity_child.child_policy_report.v2"
)
W3C_CANONICAL_XGB_POLICY_ID: Final[str] = "ta_liquidity_penalized_xgb_ta"
W3C_D_RECOVERY_UTILITY_ANCHOR: Final[float] = 111.415
W3C_B_RETENTION_GUARDRAIL: Final[float] = 0.60
W3C_MIN_TEST_DATE_PARTICIPATION_RATE: Final[float] = 0.01

W3C_LIQUIDITY_CHILD_POLICY_IDS: Final[tuple[str, ...]] = (
    "ta_liquidity_penalized_equal_weight",
    "ta_liquidity_penalized_orientation_weighted",
    "ta_liquidity_penalized_xgb_ta",
)

W3C_CLASSIFICATIONS: Final[tuple[str, ...]] = (
    "W3C_LIQUIDITY_RECOVERY",
    "W3C_PARTIAL_RECOVERY",
    "W3C_OVER_CORRECTED",
    "W3C_NO_RECOVERY",
    "W3C_INCONCLUSIVE",
)

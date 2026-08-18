from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pysrc.strategies.stat_arb.config import PairsConfig


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _compute_gate_status(half_life_bars: float | None, signal_count: int) -> str:
    if half_life_bars is None or not (1.0 <= half_life_bars <= 252.0):
        return "FAIL"
    if signal_count == 0:
        return "WARN"
    return "PASS"


def build_execution_assumptions_payload(
    *,
    strategy: str,
    config: PairsConfig,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "strategy": strategy,
        "method": config.method,
        "hedge_estimator": config.hedge_estimator.value,
    }
    if extra:
        payload.update(extra)
    return payload


def build_stat_validity_payload(
    *,
    strategy: str,
    pair: tuple[str, str],
    config: PairsConfig,
    evaluation_window: str,
    half_life_bars: float | None,
    mean_spread_zscore: float | None,
    signal_count: int,
    pit_compliant: bool,
    validation_profile: str = "v1",
) -> dict[str, Any]:
    leg_a, leg_b = pair
    gate_status = _compute_gate_status(half_life_bars, signal_count)
    return {
        "schema_version": "1.0",
        "sharpe_ratio": 0.0,
        "dsr": {},
        "min_trl": {},
        "bootstrap_ci": {},
        "gate_result": gate_status,
        "strategy": strategy,
        "pair": {"leg_a": leg_a, "leg_b": leg_b},
        "evaluation_window": evaluation_window,
        "method": config.method,
        "hedge_estimator": config.hedge_estimator.value,
        "beta_window": config.beta_window,
        "half_life_window": config.half_life_window,
        "zscore_window": config.zscore_window,
        "entry_z": config.entry_z,
        "exit_z": config.exit_z,
        "max_hold_days": config.max_hold_days,
        "min_half_life": config.min_half_life,
        "max_half_life": config.max_half_life,
        "half_life_bars": _nan_to_none(half_life_bars),
        "mean_spread_zscore": _nan_to_none(mean_spread_zscore),
        "signal_count": int(signal_count),
        "pit_compliant": bool(pit_compliant),
        "validation_profile": validation_profile,
        "gate_status": gate_status,
    }

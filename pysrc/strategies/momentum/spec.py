"""Typed momentum strategy boundary (Programming Guidelines §3.3, §3.4).

Variant names and coerced parameter dict feed the canonical ``_PLAN_BUILDERS`` factory
table in ``MomentumStrategy``. Unknown keys are preserved for forward-compatible plans.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

MOMENTUM_VARIANTS: Final[frozenset[str]] = frozenset(
    {
        "xsec",
        "tsmom",
        "dual",
        "industry",
        "residual_ols",
        "residual_kalman",
        "ensemble",
        "ml",
    }
)


def build_momentum_params(variant: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a merged params dict with validated ``variant`` and normalized scalars.

    Raises:
        ValueError: unknown variant or invalid numeric configuration.
    """
    if variant not in MOMENTUM_VARIANTS:
        raise ValueError(
            f"Unsupported momentum variant {variant!r}. Expected one of {sorted(MOMENTUM_VARIANTS)}."
        )
    merged: dict[str, Any] = dict(params or ())
    merged["variant"] = variant

    if "target_vol" in merged:
        tv = float(merged["target_vol"])
        if tv <= 0.0:
            raise ValueError("target_vol must be positive")
        merged["target_vol"] = tv
    if "max_leverage" in merged:
        ml = float(merged["max_leverage"])
        if ml <= 0.0:
            raise ValueError("max_leverage must be positive")
        merged["max_leverage"] = ml

    for int_key in (
        "lookback_window",
        "skip_window",
        "industry_window",
        "dual_window",
        "residual_window",
        "rank_window",
    ):
        if int_key in merged:
            merged[int_key] = int(merged[int_key])

    return merged

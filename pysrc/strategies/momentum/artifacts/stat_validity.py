from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Submodule aliases so tests can monkeypatch pbo / pbo_bridge without stale
# from-import bindings.
from pysrc.backtesting.validation.statistical import pbo as _pbo
from pysrc.backtesting.validation.statistical import pbo_bridge as _pbo_bridge
from pysrc.backtesting.validation.statistical.report import run_validity_report
from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.artifacts.signal_card import RunMeta
from pysrc.strategies.momentum.validation.production_v1 import PRODUCTION_V1_PROFILE


def build_stat_validity_payload(
    *,
    alpha_ir: AlphaIR,
    run_meta: RunMeta,
    returns: list[float],
    n_trials: int,
    pbo_path_pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_pairs = _pbo_bridge.build_pbo_path_pairs(pbo_path_pairs) if pbo_path_pairs else None
    pbo_result: Mapping[str, Any] | None = None
    if normalized_pairs is not None:
        pbo_result = _pbo.compute_pbo(normalized_pairs, mode=_pbo_bridge.CANONICAL_PBO_MODE)

    payload = run_validity_report(
        returns,
        n_trials=n_trials,
        pbo_result=pbo_result,
    )
    payload["schema_version"] = "v1"
    payload["strategy"] = run_meta.strategy
    payload["run_id"] = run_meta.run_id
    payload["variant"] = alpha_ir.variant
    payload["pit_compliant"] = alpha_ir.pit_provenance is not None
    payload["validation_profile"] = PRODUCTION_V1_PROFILE.to_dict()
    payload["alpha_ir"] = {
        "information_coefficient": alpha_ir.information_coefficient,
        "diagnostics": dict(alpha_ir.diagnostics),
    }
    return payload

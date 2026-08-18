# py/backtesting/validation/statistical/report.py
"""
Assembles stat_validity_report.json from DSR, minTRL, bootstrap CI,
and validator-supplied PBO output. Wires into gate.py via the public
run_validity_report() function.

Output schema (Appendix H, v1):
{
    "schema_version": "v1",
    "sharpe_ratio": 1.23,
    "dsr": {"value": 0.87, "p_value": 0.02, "n_trials": 5},
    "min_trl": {"years_needed": 2.1, "years_available": 5.0},
    "bootstrap_ci": {
        "lower_95": 0.41, "upper_95": 2.05,
        "lower_99": 0.12, "upper_99": 2.34,
        "n_resamples": 10000
    },
    "pbo": {"value": 0.22, "gate_result": "PASS"},
    "gate_result": "PASS"   # PASS | WARN | FAIL
}
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from typing import Any

from pysrc.backtesting.validation.statistical.dsr import (
    DSRError,
    compute_bootstrap_ci,
    compute_dsr,
    compute_min_trl,
)
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

# Gate precedence: FAIL > WARN > PASS
_GATE_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _aggregate_gate(*gates: str) -> str:
    """Return the worst gate result across all components."""
    return max(gates, key=lambda g: _GATE_RANK.get(g, 0))


def _default_pbo_result() -> dict[str, Any]:
    """Emit a neutral PBO warning when no CPCV score surface is available."""
    return {
        "value": 0.50,
        "threshold": 0.50,
        "warn_threshold": 0.40,
        "gate_result": "WARN",
        "method": "unavailable",
        "score_basis": "net_sharpe",
        "n_trials": 0,
        "n_paths": 0,
    }


def _coerce_pbo_result(pbo_result: Mapping[str, Any] | None) -> dict[str, Any]:
    if pbo_result is None:
        return _default_pbo_result()
    return dict(pbo_result)


def run_validity_report(
    returns: Any,
    *,
    n_trials: int = 1,
    periods_per_year: int = 252,
    benchmark_sr: float = 0.0,
    n_resamples: int = 10_000,
    block_size: int | None = None,
    random_state: int = 42,
    output_path: str | pathlib.Path | None = None,
    pbo_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the full statistical validity suite and return the report dict.

    Runs DSR, minTRL, and block-bootstrap CI in sequence. Aggregates
    gate results - FAIL > WARN > PASS - so any single FAIL produces
    a FAIL at the top level.

    Args:
        returns:          Return series (daily or per-period).
        n_trials:         Number of strategy variants tested. Pass the actual
                          count for honest DSR adjustment. Defaults to 1
                          (no multi-testing correction, with a logged warning).
        periods_per_year: 252 daily / 52 weekly / 12 monthly.
        benchmark_sr:     Minimum acceptable SR for DSR and minTRL.
        n_resamples:      Bootstrap replications (Appendix H specifies 10,000).
        block_size:       Bootstrap block size. None = cube-root rule.
        random_state:     RNG seed for bootstrap reproducibility.
        output_path:      If provided, write JSON to this path.
        pbo_result:       Validator-supplied PBO payload assembled from CPCV
                          evaluations. When omitted, emit a neutral WARN payload
                          so the v1 report contract remains intact.

    Returns:
        Report dict conforming to Appendix H v1 schema.
    """
    # --- DSR ---
    try:
        dsr_result = compute_dsr(
            returns,
            n_trials=n_trials,
            periods_per_year=periods_per_year,
            benchmark_sr=benchmark_sr,
        )
    except DSRError as e:  # numerical or data pathology - degrade to WARN
        LOG.warning("dsr_failed", error=str(e))
        dsr_result = {
            "sharpe_ratio": 0.0,
            "dsr": 0.0,
            "p_value": 1.0,
            "n_trials": n_trials,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "gate_result": "WARN",
        }

    # --- minTRL ---
    try:
        trl_result = compute_min_trl(
            returns,
            periods_per_year=periods_per_year,
            benchmark_sr=benchmark_sr,
        )
    except DSRError as e:
        LOG.warning("min_trl_failed", error=str(e))
        trl_result = {
            "observed_sr": dsr_result.get("sharpe_ratio", 0.0),
            "years_needed": None,
            "years_available": 0.0,
            "target_confidence": 0.95,
            "gate_result": "WARN",
        }

    # --- Bootstrap CI ---
    try:
        ci_result = compute_bootstrap_ci(
            returns,
            n_resamples=n_resamples,
            block_size=block_size,
            periods_per_year=periods_per_year,
            random_state=random_state,
        )
    except DSRError as e:
        LOG.warning("bootstrap_ci_failed", error=str(e))
        ci_result = {
            "lower_95": None,
            "upper_95": None,
            "lower_99": None,
            "upper_99": None,
            "n_resamples": n_resamples,
            "block_size": block_size,
            "gate_result": "WARN",
        }

    pbo_payload = _coerce_pbo_result(pbo_result)

    # --- Assemble report (Appendix H v1 schema) ---
    report: dict[str, Any] = {
        "schema_version": "v1",
        "sharpe_ratio": dsr_result["sharpe_ratio"],
        "dsr": {
            "value": dsr_result["dsr"],
            "p_value": dsr_result["p_value"],
            "n_trials": dsr_result["n_trials"],
            "skewness": dsr_result["skewness"],
            "excess_kurtosis": dsr_result["excess_kurtosis"],
            "gate_result": dsr_result["gate_result"],
        },
        "min_trl": {
            "years_needed": trl_result["years_needed"],
            "years_available": trl_result["years_available"],
            "target_confidence": trl_result["target_confidence"],
            "gate_result": trl_result["gate_result"],
        },
        "bootstrap_ci": {
            "lower_95": ci_result.get("lower_95"),
            "upper_95": ci_result.get("upper_95"),
            "lower_99": ci_result.get("lower_99"),
            "upper_99": ci_result.get("upper_99"),
            "n_resamples": ci_result["n_resamples"],
            "block_size": ci_result["block_size"],
            "gate_result": ci_result["gate_result"],
        },
        "pbo": pbo_payload,
    }
    report["gate_result"] = _aggregate_gate(
        report["dsr"]["gate_result"],
        report["min_trl"]["gate_result"],
        report["bootstrap_ci"]["gate_result"],
        str(report["pbo"].get("gate_result", "PASS")),
    )

    LOG.info(
        "validity_report_complete",
        sharpe_ratio=report["sharpe_ratio"],
        dsr_p_value=report["dsr"]["p_value"],
        years_available=report["min_trl"]["years_available"],
        years_needed=report["min_trl"]["years_needed"],
        ci_lower_95=report["bootstrap_ci"]["lower_95"],
        pbo_value=report["pbo"]["value"],
        gate_result=report["gate_result"],
    )

    # --- Optional file write ---
    if output_path is not None:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
        LOG.info("validity_report_written", path=str(path))

    return report

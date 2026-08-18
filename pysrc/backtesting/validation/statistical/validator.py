from __future__ import annotations

from typing import Any

from pysrc.backtesting.contracts.registry import register_validator
from pysrc.backtesting.contracts.types import ValidationReport, ValidationStatus
from pysrc.backtesting.validation.statistical.pbo import compute_pbo
from pysrc.backtesting.validation.statistical.pbo_bridge import (
    CANONICAL_PBO_MODE,
    build_pbo_path_pairs,
)


class StatisticalValidator:
    def validate(self, result, ctx: dict[str, Any], store) -> ValidationReport:
        returns = ctx.get("returns")
        try:
            from pysrc.backtesting.validation.statistical.report import run_validity_report
        except ModuleNotFoundError:
            run_validity_report = None

        pbo_path_pairs = self._resolve_pbo_path_pairs(ctx)
        pbo_result = self._build_pbo_result(pbo_path_pairs)
        n_trials = self._resolve_n_trials(ctx, pbo_path_pairs)

        if returns is None or run_validity_report is None:
            report = {
                "schema_version": "v1",
                "sharpe_ratio": float(result.metrics.get("sharpe_ratio", 0.0)),
                "dsr": {
                    "value": 0.0,
                    "p_value": 1.0,
                    "n_trials": n_trials,
                    "skewness": 0.0,
                    "excess_kurtosis": 0.0,
                    "gate_result": "WARN",
                },
                "min_trl": {
                    "years_needed": 0.0,
                    "years_available": 0.0,
                    "target_confidence": 0.95,
                    "gate_result": "WARN",
                },
                "bootstrap_ci": {
                    "lower_95": 0.0,
                    "upper_95": 0.0,
                    "lower_99": 0.0,
                    "upper_99": 0.0,
                    "n_resamples": 0,
                    "block_size": None,
                    "gate_result": "WARN",
                },
                "gate_result": self._aggregate_gate_result(pbo_result.get("gate_result", "WARN")),
            }
        else:
            report = run_validity_report(
                returns,
                n_trials=n_trials,
                pbo_result=pbo_result,
            )
        ref = store.put_json("stat_validity_report.json", report)
        status = ValidationStatus(report["gate_result"])
        return ValidationReport(
            status=status,
            reason_code=f"STAT_{status.value}",
            message=f"Statistical validator emitted {status.value}",
            artifacts={"stat_validity_report.json": ref},
        )

    @staticmethod
    def _resolve_pbo_path_pairs(ctx: dict[str, Any]) -> list[dict[str, Any]] | None:
        if "pbo_path_pairs" in ctx:
            return build_pbo_path_pairs(ctx["pbo_path_pairs"])
        if "cpcv_evaluations" in ctx:
            return build_pbo_path_pairs(ctx["cpcv_evaluations"])
        return None

    @staticmethod
    def _build_pbo_result(pbo_path_pairs: list[dict[str, Any]] | None) -> dict[str, Any]:
        if pbo_path_pairs is None:
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
        return compute_pbo(
            pbo_path_pairs,
            mode=CANONICAL_PBO_MODE,
        )

    @staticmethod
    def _resolve_n_trials(ctx: dict[str, Any], pbo_path_pairs: list[dict[str, Any]] | None) -> int:
        if pbo_path_pairs:
            return int(len(pbo_path_pairs[0]["in_sample_scores"]))
        return int(ctx.get("n_trials", 1))

    @staticmethod
    def _aggregate_gate_result(pbo_gate_result: str) -> str:
        if pbo_gate_result == "FAIL":
            return ValidationStatus.FAIL.value
        return ValidationStatus.WARN.value


register_validator("statistical.v1", lambda: StatisticalValidator())

from __future__ import annotations

from typing import Any

from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.api import PlanSpec

from ._base_plan_step import PlanStep
from ._normalize import (
    canonical_op,
    normalize_common_keys,
    normalize_input,
)
from ._provenance import build_meta
from .aliases import TECHNICAL as A

logger = get_logger(__name__)


class TechnicalFeaturesStep(PlanStep):
    STEP_NAME = "TechnicalFeaturesStep"
    STEP_VERSION = "2.1.0"

    def _build_spec(self) -> PlanSpec:
        ops = self._build_ops_from_cfg(self.cfg)
        meta = build_meta(self.STEP_NAME, self.STEP_VERSION, self.cfg)
        return PlanSpec(
            ops=ops,
            target=self.cfg.get("target"),
            sequence=self.cfg.get("sequence"),
            scaling=self.cfg.get("scaling"),
            meta=meta,
        )

    def _build_ops_from_cfg(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if cfg.get("ops"):
            return [self._normalize_op_dict(o) for o in cfg["ops"]]
        return self._ops_from_legacy_indicators(cfg)

    def _normalize_op_dict(self, op: dict[str, Any]) -> dict[str, Any]:
        o = dict(op)
        name = o.get("op") or o.get("indicator") or o.get("name")
        if not name:
            raise ValueError("Each technical op requires 'op' (or 'indicator'/'name').")
        o["op"] = canonical_op(name, A)

        o = normalize_common_keys(o)
        o = normalize_input(o)

        # Shorthands
        if "window" not in o and "w" in o:
            o["window"] = o.pop("w")
        if "k" not in o and "std_dev" in o:
            o["k"] = o.pop("std_dev")

        # Bollinger mean
        if o["op"] == "technical.bollinger_bands" and "bb_type" in o and "mean" not in o:
            mean = str(o.pop("bb_type")).lower()
            o["mean"] = "ema" if mean == "ema" else "sma"

        return o

    def _ops_from_legacy_indicators(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        by = cfg.get("group_key") or cfg.get("by")
        nan_policy = cfg.get("nan_policy")
        min_periods = cfg.get("min_periods")

        rsi = cfg.get("rsi") or {}
        if rsi.get("enabled"):
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "technical.rsi",
                        "input": rsi.get("col", "close"),
                        "window": int(rsi.get("window", 14)),
                        "by": by,
                        "min_periods": min_periods,
                        "nan_policy": nan_policy,
                    }
                )
            )

        macd = cfg.get("macd") or {}
        if macd.get("enabled"):
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "technical.macd",
                        "input": macd.get("col", "close"),
                        "fast_period": int(macd.get("fast_period", 12)),
                        "slow_period": int(macd.get("slow_period", 26)),
                        "signal_period": int(macd.get("signal_period", 9)),
                        "by": by,
                        "nan_policy": nan_policy,
                    }
                )
            )

        atr = cfg.get("atr") or {}
        if atr.get("enabled"):
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "technical.atr",
                        "high": atr.get("high", "high"),
                        "low": atr.get("low", "low"),
                        "close": atr.get("close", "close"),
                        "window": int(atr.get("window", 14)),
                        "min_periods": min_periods,
                        "by": by,
                        "nan_policy": nan_policy,
                    }
                )
            )

        vwap = cfg.get("vwap") or {}
        if vwap.get("enabled"):
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "technical.vwap",
                        "high": vwap.get("high", "high"),
                        "low": vwap.get("low", "low"),
                        "close": vwap.get("close", "close"),
                        "volume": vwap.get("volume", "volume"),
                        "reset_period": vwap.get("reset_period", "daily"),
                        "by": by,
                        "nan_policy": nan_policy,
                    }
                )
            )

        bb = cfg.get("bollinger_bands") or {}
        if bb.get("enabled"):
            mean = str(cfg.get("bb_type", bb.get("bb_type", "sma"))).lower()
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "technical.bollinger_bands",
                        "input": bb.get("col", "close"),
                        "window": int(bb.get("window", 20)),
                        "k": float(bb.get("std_dev", 2.0)),
                        "mean": "ema" if mean == "ema" else "sma",
                        "min_periods": min_periods,
                        "by": by,
                        "nan_policy": nan_policy,
                    }
                )
            )

        if not ops:
            logger.warning("TechnicalFeaturesStep: no indicators configured; no-op.")
        return ops

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

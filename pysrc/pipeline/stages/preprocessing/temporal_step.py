from __future__ import annotations

from typing import Any

from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.api import PlanSpec

from ._base_plan_step import PlanStep
from ._normalize import (
    canonical_op,
    normalize_bucket,
    normalize_common_keys,
    normalize_lag,
)
from ._provenance import build_meta
from .aliases import TEMPORAL as A

logger = get_logger(__name__)


class TemporalStep(PlanStep):
    STEP_NAME = "TemporalStep"
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
        return self._ops_from_legacy(cfg)

    def _normalize_op_dict(self, op: dict[str, Any]) -> dict[str, Any]:
        o = normalize_common_keys(dict(op))
        name = o.get("op") or o.get("name")
        if not name:
            raise ValueError("Temporal op requires 'op' or 'name'.")
        o["op"] = canonical_op(name, A)

        if o["op"] == "time.lag":
            o = normalize_lag(o)
        elif o["op"] == "time.bucket":
            o = normalize_bucket(o)

        return o

    def _ops_from_legacy(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        ts_col = cfg.get("timestamp_col") or cfg.get("ts_col") or "timestamp"
        by = cfg.get("group_key") or cfg.get("by")

        lags = cfg.get("lags")
        if lags:
            base_col = cfg.get("base_col", "close")
            fill = cfg.get("fill_strategy", "none")
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "time.lag",
                        "column": base_col,
                        "lags": lags,
                        "fill_strategy": fill,
                        "by": by,
                        "ts_col": ts_col,
                        "prefix": cfg.get("lag_prefix", f"{base_col}_lag_"),
                    }
                )
            )

        if cfg.get("session_flags"):
            start_h, end_h = cfg.get("trading_hours", (9, 16))
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "time.session_flag",
                        "ts_col": ts_col,
                        "start_hour": int(start_h),
                        "end_hour": int(end_h),
                        "label": cfg.get("session_label", "is_trading_session"),
                        "by": by,
                    }
                )
            )

        feats = cfg.get("calendar_features")
        if feats:
            ops.append(
                self._normalize_op_dict(
                    {"op": "time.calendar", "ts_col": ts_col, "features": feats, "by": by}
                )
            )

        buckets = cfg.get("time_buckets")
        if buckets:
            if isinstance(buckets, dict):
                for unit, size in buckets.items():
                    ops.append(
                        self._normalize_op_dict(
                            {
                                "op": "time.bucket",
                                "ts_col": ts_col,
                                "unit": unit,
                                "size": int(size),
                                "label": f"{unit}_bucket",
                                "by": by,
                            }
                        )
                    )
            else:
                for b in buckets:
                    unit = b.get("unit") or b.get("granularity")
                    size = b.get("size") or b.get("every")
                    ops.append(
                        self._normalize_op_dict(
                            {
                                "op": "time.bucket",
                                "ts_col": ts_col,
                                "unit": unit,
                                "size": int(size),
                                "label": b.get("label") or f"{unit}_bucket",
                                "by": by,
                            }
                        )
                    )

        if not ops:
            logger.warning("TemporalStep: no temporal options provided; no-op.")
        return ops

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

from __future__ import annotations

from typing import Any

from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.api import PlanSpec

from ._base_plan_step import PlanStep
from ._normalize import canonical_op, normalize_clip, normalize_common_keys
from ._provenance import build_meta
from .aliases import SCALING as A

logger = get_logger(__name__)


class ScalingStep(PlanStep):
    STEP_NAME = "ScalingStep"
    STEP_VERSION = "2.1.0"

    def _build_spec(self) -> PlanSpec:
        ops = self._build_ops_from_cfg(self.cfg)
        meta = build_meta(self.STEP_NAME, self.STEP_VERSION, self.cfg)
        return PlanSpec(
            ops=ops,
            target=self.cfg.get("target"),
            sequence=self.cfg.get("sequence"),
            meta=meta,
        )

    def _build_ops_from_cfg(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if cfg.get("ops"):
            return [self._normalize_op_dict(o) for o in cfg["ops"]]
        return self._ops_from_legacy(cfg)

    def _normalize_op_dict(self, op: dict[str, Any]) -> dict[str, Any]:
        o = normalize_common_keys(dict(op))
        name = o.get("op") or o.get("method")
        if not name:
            raise ValueError("Scaling op requires 'op' (or 'method').")
        o["op"] = canonical_op(name, A)
        if o["op"] in ("transform.clip",):
            o = normalize_clip(o)
        return o

    def _ops_from_legacy(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if not cfg.get("enabled", True):
            logger.info("ScalingStep: enabled=false; no ops.")
            return []
        method = (cfg.get("method") or "zscore").lower()
        op_id = canonical_op(method, A)

        base: dict[str, Any] = {"op": op_id}
        if "columns" in cfg:
            base["columns"] = cfg["columns"]
        if "cols" in cfg and "columns" not in base:
            base["columns"] = cfg["cols"]
        if "group_key" in cfg:
            base["by"] = cfg["group_key"]
        if "by" in cfg and "by" not in base:
            base["by"] = cfg["by"]
        if "nan_policy" in cfg:
            base["nan_policy"] = cfg["nan_policy"]

        if op_id == "normalize.zscore":
            if "with_mean" in cfg:
                base["with_mean"] = bool(cfg["with_mean"])
            if "with_std" in cfg:
                base["with_std"] = bool(cfg["with_std"])
        elif op_id == "normalize.minmax":
            if "feature_range" in cfg:
                base["feature_range"] = cfg["feature_range"]
        elif op_id == "normalize.robust":
            if "quantiles" in cfg:
                base["quantiles"] = cfg["quantiles"]
            if "iqr_factor" in cfg:
                base["iqr_factor"] = cfg["iqr_factor"]
        elif op_id == "transform.clip":
            if "clip_min" in cfg:
                base["lower"] = cfg["clip_min"]
            if "clip_max" in cfg:
                base["upper"] = cfg["clip_max"]
        elif op_id == "transform.winsorize":
            if "p_low" in cfg:
                base["p_low"] = cfg["p_low"]
            if "p_high" in cfg:
                base["p_high"] = cfg["p_high"]

        return [self._normalize_op_dict(base)]

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

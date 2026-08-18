from __future__ import annotations

from typing import Any

from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.api import PlanSpec

from ._base_plan_step import PlanStep
from ._normalize import canonical_op, normalize_common_keys
from ._provenance import build_meta
from .aliases import SEQUENCE as A

logger = get_logger(__name__)


class SequenceStep(PlanStep):
    STEP_NAME = "SequenceStep"
    STEP_VERSION = "2.1.0"

    def _build_spec(self) -> PlanSpec:
        ops = self._build_ops_from_cfg(self.cfg)
        meta = build_meta(self.STEP_NAME, self.STEP_VERSION, self.cfg)
        sequence_spec = self._build_sequence_spec_from_cfg(self.cfg)
        target = self.cfg.get("target") or self._infer_target(self.cfg)
        return PlanSpec(ops=ops, sequence=sequence_spec, target=target, meta=meta)

    def _build_ops_from_cfg(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if not cfg.get("ops"):
            return []
        return [self._normalize_op_dict(o) for o in cfg["ops"]]

    def _build_sequence_spec_from_cfg(self, cfg: dict[str, Any]) -> dict[str, Any] | None:
        seq_len = int(cfg.get("sequence_length", cfg.get("length", 0) or 0))
        horizon = int(cfg.get("horizon", 1))
        stride = int(cfg.get("stride", 1))
        feats = cfg.get("feature_cols") or cfg.get("features")
        by = cfg.get("group_key") or cfg.get("by")
        ts_col = cfg.get("timestamp_col") or cfg.get("ts_col") or "timestamp"
        adaptive = bool(cfg.get("adaptive", False))
        drop_incomplete = bool(cfg.get("drop_incomplete", True))
        target_col = self._infer_target_col(cfg)

        if not (seq_len or cfg.get("ops")):
            return None

        return {
            "length": seq_len or 60,
            "horizon": horizon,
            "stride": stride,
            "features": feats,
            "target": target_col,
            "by": by,
            "ts_col": ts_col,
            "adaptive": adaptive,
            "drop_incomplete": drop_incomplete,
        }

    def _normalize_op_dict(self, op: dict[str, Any]) -> dict[str, Any]:
        o = normalize_common_keys(dict(op))
        name = o.get("op") or o.get("name")
        if not name:
            raise ValueError("Sequence op requires 'op' or 'name'.")
        o["op"] = canonical_op(name, A)

        if "length" not in o and "sequence_length" in o:
            o["length"] = int(o.pop("sequence_length"))
        if "stride" in o:
            o["stride"] = int(o["stride"])
        if "horizon" in o:
            o["horizon"] = int(o["horizon"])
        if "target" not in o and "target_col" in o:
            o["target"] = o.pop("target_col")
        return o

    def _infer_target(self, cfg: dict[str, Any]) -> dict[str, Any] | None:
        col = self._infer_target_col(cfg)
        return {"column": col} if col else None

    def _infer_target_col(self, cfg: dict[str, Any]) -> str | None:
        return cfg.get("target_col") or (cfg.get("target") or {}).get("column")

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

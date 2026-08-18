from __future__ import annotations

from typing import Any

from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.api import PlanSpec

from ._base_plan_step import PlanStep
from ._normalize import (
    assign_device_hint,
    canonical_op,
    normalize_clip,
    normalize_common_keys,
)
from ._provenance import build_meta
from .aliases import SENTIMENT as A

logger = get_logger(__name__)


class SentimentESGStep(PlanStep):
    STEP_NAME = "SentimentESGStep"
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
        if cfg.get("processors"):
            return self._ops_from_legacy_processors(cfg)
        return self._ops_from_legacy_custom_features(cfg)

    def _normalize_op_dict(self, op: dict[str, Any]) -> dict[str, Any]:
        o = normalize_common_keys(dict(op))
        name = o.get("op") or o.get("name")
        if not name:
            raise ValueError("Sentiment/ESG op requires 'op' or 'name'.")
        o["op"] = canonical_op(name, A)
        o = assign_device_hint(o)
        if o["op"] == "transform.clip":
            o = normalize_clip(o)
        return o

    def _ops_from_legacy_processors(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        text_col = cfg.get("text_col", "text")
        use_gpu = bool(cfg.get("use_gpu", False))
        gpu_id = cfg.get("gpu_id", 0)
        post_clip = cfg.get("post_clip_sentiment", True)
        lower, upper = tuple(cfg.get("clip_bounds", (-1.0, 1.0)))

        if cfg.get("clean_text", True):
            ops.append(self._normalize_op_dict({"op": "text.clean_ascii", "text_col": text_col}))

        for item in cfg.get("processors", []):
            name = item.get("name")
            params = dict(item.get("params", {}))
            if "device" not in params:
                params["device"] = "gpu" if use_gpu else "cpu"
            if params["device"] == "gpu" and "gpu_id" not in params:
                params["gpu_id"] = gpu_id
            if "text_col" not in params:
                params["text_col"] = text_col

            ops.append(self._normalize_op_dict({"op": name, **params}))

            if canonical_op(name, A) in ("nlp.sentiment.finbert", "nlp.sentiment.hf") and post_clip:
                ops.append(
                    self._normalize_op_dict(
                        {
                            "op": "transform.clip",
                            "columns": ["sentiment"],
                            "lower": float(lower),
                            "upper": float(upper),
                        }
                    )
                )

        return ops

    def _ops_from_legacy_custom_features(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        cf = cfg.get("custom_features", {}) or {}
        esg_cols = cfg.get("esg_cols", ["environmental", "social", "governance"])
        text_col = cfg.get("text_col", "text")

        sent = cf.get("sentiment") or {}
        if sent.get("enabled"):
            col = sent.get("column", "sentiment")
            if sent.get("create_if_missing", True):
                ops.append(
                    self._normalize_op_dict(
                        {
                            "op": "column.create",
                            "name": col,
                            "value": float(sent.get("default", 0.0)),
                            "if_missing": True,
                        }
                    )
                )
            lo, hi = (-1.0, 1.0)
            if isinstance(sent.get("clip"), (list, tuple)) and len(sent["clip"]) == 2:
                lo, hi = float(sent["clip"][0]), float(sent["clip"][1])
            ops.append(
                self._normalize_op_dict(
                    {"op": "transform.clip", "columns": [col], "lower": lo, "upper": hi}
                )
            )

        esg_norm = cf.get("esg_normalized") or {}
        if esg_norm.get("enabled"):
            if esg_norm.get("aggregate_from_cols", True):
                ops.append(
                    self._normalize_op_dict(
                        {
                            "op": "esg.normalize",
                            "esg_cols": esg_cols,
                            "aggregate": esg_norm.get("aggregate", "mean"),
                            "output": esg_norm.get("output", "esg_score"),
                        }
                    )
                )
            method = (esg_norm.get("method") or "minmax").lower()
            score_col = esg_norm.get("column", "esg_score")
            if method == "minmax":
                ops.append(
                    self._normalize_op_dict(
                        {
                            "op": "normalize.minmax",
                            "columns": [score_col],
                            "feature_range": esg_norm.get("feature_range", [0.0, 1.0]),
                        }
                    )
                )
            elif method in ("zscore", "standard"):
                ops.append(
                    self._normalize_op_dict(
                        {
                            "op": "normalize.zscore",
                            "columns": [score_col],
                            "with_mean": True,
                            "with_std": True,
                        }
                    )
                )
            else:
                logger.warning(f"Unknown ESG method '{method}'; skipping scale op.")

        if cfg.get("infer_sentiment"):
            params = {
                "text_col": text_col,
                "model_name": cfg.get("model_name", "ProsusAI/finbert"),
                "batch_size": cfg.get("batch_size", 32),
                "max_length": cfg.get("max_length", 512),
                "device": "gpu" if cfg.get("use_gpu", False) else "cpu",
                "gpu_id": cfg.get("gpu_id", 0),
            }
            ops.append(self._normalize_op_dict({"op": "nlp.sentiment.finbert", **params}))
            if cfg.get("post_clip_sentiment", True):
                ops.append(
                    self._normalize_op_dict(
                        {
                            "op": "transform.clip",
                            "columns": ["sentiment"],
                            "lower": -1.0,
                            "upper": 1.0,
                        }
                    )
                )

        if not ops:
            logger.warning("SentimentESGStep: no processors/custom_features/ops; no-op.")
        return ops

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

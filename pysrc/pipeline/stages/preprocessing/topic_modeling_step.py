# py/pipeline/stages/preprocessing/topic_modeling_step.py
from __future__ import annotations

from typing import Any

from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.api import PlanSpec

from ._base_plan_step import PlanStep
from ._normalize import (
    assign_device_hint,
    canonical_op,
    normalize_common_keys,
)
from ._provenance import build_meta
from .aliases import TOPIC as A

logger = get_logger(__name__)


class TopicModelingStep(PlanStep):
    STEP_NAME = "TopicModelingStep"
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

    # ------------------------
    # Config → ops translation
    # ------------------------
    def _build_ops_from_cfg(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if cfg.get("ops"):
            return [self._normalize_op_dict(o, cfg) for o in cfg["ops"]]
        return self._ops_from_legacy(cfg)

    def _normalize_op_dict(
        self, op: dict[str, Any], root_cfg: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        o = normalize_common_keys(dict(op))
        name = o.get("op") or o.get("name")
        if not name:
            raise ValueError("Topic modeling op requires 'op' or 'name'.")
        o["op"] = canonical_op(name, A)

        # Accept legacy 'column' for text
        if "text_col" not in o and "column" in o:
            o["text_col"] = o.pop("column")

        # Device hint (optional)
        o = assign_device_hint(o)

        # Embeddings pipeline_config normalization
        if "embeddings" in o and isinstance(o["embeddings"], dict):
            emb = dict(o["embeddings"])
            # unify common legacy keys
            if "use_precomputed_embeddings" in emb and "use_precomputed" not in emb:
                emb["use_precomputed"] = bool(emb.pop("use_precomputed_embeddings"))
            if "prefix" not in emb and "embedding_prefix" in emb:
                emb["prefix"] = emb.pop("embedding_prefix")
            o["embeddings"] = emb

        return o

    def _ops_from_legacy(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if cfg.get("enabled") is False:
            logger.info("TopicModelingStep: enabled=false; no-op.")
            return []

        ops: list[dict[str, Any]] = []

        # Main topic op (BERTopic)
        op: dict[str, Any] = {
            "op": "nlp.topic.bertopic",
            "text_col": cfg.get("column", "text"),
            "model_params": cfg.get("model_params") or {},
        }

        # Device hint (optional)
        if "device" in cfg:
            op["device"] = "gpu" if str(cfg["device"]).lower() == "gpu" else "cpu"
        if "gpu_id" in cfg:
            op["gpu_id"] = int(cfg["gpu_id"])

        # Embeddings usage (legacy)
        use_pre = bool(cfg.get("use_precomputed_embeddings", True))
        prefix = cfg.get("embedding_prefix", "emb_")
        op["embeddings"] = {"use_precomputed": use_pre, "prefix": prefix}

        ops.append(self._normalize_op_dict(op, cfg))

        # Probability filter as a separate op (lets backends implement efficiently)
        min_prob = float(cfg.get("min_prob", 0.0))
        if min_prob > 0.0:
            ops.append(
                self._normalize_op_dict(
                    {
                        "op": "topic.filter_prob",
                        "min_prob": min_prob,
                        "null_topic": cfg.get("null_topic", -1),
                    },
                    cfg,
                )
            )

        return ops

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

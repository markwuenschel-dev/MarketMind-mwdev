# py/pipeline/stages/preprocessing/text_embedding_step.py
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
from .aliases import EMBEDDING as A

logger = get_logger(__name__)


class TextEmbeddingStep(PlanStep):
    STEP_NAME = "TextEmbeddingStep"
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
            return [self._normalize_op_dict(o) for o in cfg["ops"]]
        return self._ops_from_legacy(cfg)

    def _normalize_op_dict(self, op: dict[str, Any]) -> dict[str, Any]:
        o = normalize_common_keys(dict(op))
        name = o.get("op") or o.get("name")
        if not name:
            raise ValueError("Embedding op requires 'op' or 'name'.")
        o["op"] = canonical_op(name, A)

        # Model name normalization
        if "model_name" not in o and "model" in o:
            o["model_name"] = o.pop("model")

        # Precision/dtype normalization
        if "precision" not in o and "fp16" in o:
            o["precision"] = "fp16" if o.pop("fp16") else "fp32"

        # Device hint from legacy use_gpu/device
        o = assign_device_hint(o)  # maps use_gpu -> device if present
        if "device" not in o and "device_type" in o:
            o["device"] = o.pop("device_type")

        # Output mode normalization
        if "output" not in o:
            # accept legacy fields directly on the op
            mode = o.pop("output_mode", None)
            id_col = o.pop("id_column", None)
            if mode:
                out = {"mode": str(mode)}
                if id_col:
                    out["id_column"] = id_col
                o["output"] = out

        # text column naming
        if "text_col" not in o:
            # normalize_common_keys may have copied from "text" already
            if "column" in o:
                o["text_col"] = o.pop("column")

        return o

    def _ops_from_legacy(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        if cfg.get("enabled") is False:
            logger.info("TextEmbeddingStep: enabled=false; no-op.")
            return []

        # Canonical op id (Sentence-Transformers default)
        op_id = canonical_op("nlp.embed.sentence_transformers", A)

        # Build op from legacy keys
        op: dict[str, Any] = {
            "op": op_id,
            "model_name": cfg.get("model") or "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": int(cfg.get("batch_size", 32)),
            "normalize": bool(cfg.get("normalize", False)),
            "text_col": cfg.get("column", "text"),
        }

        # precision from fp16
        if "fp16" in cfg:
            op["precision"] = "fp16" if cfg["fp16"] else "fp32"

        # device hint
        device = cfg.get("device")
        if device:
            # map "cuda" -> "gpu"; keep "cpu" as-is
            op["device"] = "gpu" if str(device).lower() in ("cuda", "gpu") else "cpu"
        if "gpu_id" in cfg:
            op["gpu_id"] = int(cfg["gpu_id"])

        # output mode
        mode = cfg.get("output_mode")
        if mode:
            out = {"mode": str(mode)}
            if str(mode).lower() == "sidecar":
                id_col = cfg.get("id_column")
                if not id_col:
                    logger.warning(
                        "TextEmbeddingStep: output_mode='sidecar' but no id_column provided."
                    )
                else:
                    out["id_column"] = id_col
            op["output"] = out

        return [self._normalize_op_dict(op)]

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

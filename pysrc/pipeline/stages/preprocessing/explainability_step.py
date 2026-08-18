# py/pipeline/stages/preprocessing/explainability_step.py
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
from .aliases import EXPLAIN as A

logger = get_logger(__name__)


class ExplainabilityStep(PlanStep):
    STEP_NAME = "ExplainabilityStep"
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
            raise ValueError("Explainability op requires 'op' or 'name'.")
        o["op"] = canonical_op(name, A)

        # device hint (optional)
        o = assign_device_hint(o)

        # model binding: prefer per-op model_ref, else inherit from root_cfg
        if "model_ref" not in o and root_cfg:
            if "model_ref" in root_cfg:
                o["model_ref"] = root_cfg["model_ref"]
            elif "model" in root_cfg and isinstance(root_cfg["model"], str):
                # legacy: allow model as string id
                o["model_ref"] = root_cfg["model"]

        # background normalization
        if "background" in o:
            o["background"] = self._normalize_background(o["background"])
        else:
            # map legacy flat keys if present on the op
            if "background_size" in o:
                o["background"] = {"strategy": "random", "k": int(o.pop("background_size"))}
            if "kmeans_k" in o:
                o["background"] = {"strategy": "kmeans", "k": int(o.pop("kmeans_k"))}

        # batch & runtime knobs pass-through (backends decide defaults)
        # keep: batch_size, max_batch_rows, oom_retry, precision, etc., if provided.
        return o

    def _normalize_background(self, bg: Any) -> dict[str, Any]:
        # Canonical background format: {"strategy": "random"|"kmeans", "k": int}
        # Accepts legacy: {"size":100} or {"kmeans_k":50} or just 100.
        if bg is None:
            return {"strategy": "random", "k": 50}
        if isinstance(bg, int):
            return {"strategy": "random", "k": int(bg)}
        if isinstance(bg, dict):
            out = dict(bg)
            # unify keys
            if "k" not in out and "size" in out:
                out["k"] = int(out.pop("size"))
            if "strategy" not in out and "kmeans_k" in out:
                out["strategy"] = "kmeans"
                out["k"] = int(out.pop("kmeans_k"))
            # defaults
            out.setdefault("strategy", "random")
            out.setdefault("k", 50)
            return out
        # fallback
        return {"strategy": "random", "k": 50}

    def _ops_from_legacy(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        model_ref = cfg.get("model_ref") or cfg.get(
            "model"
        )  # string id; executor resolves actual model
        device = cfg.get("device")  # optional hint

        for item in cfg.get("explainers", []):
            name = item.get("name")
            params = dict(item.get("params", {}))

            # normalize to canonical op id
            op_id = canonical_op(name, A)

            # background: accept background_size / kmeans_k on params
            bg = None
            if "background" in params:
                bg = self._normalize_background(params.pop("background"))
            elif "background_size" in params:
                bg = {"strategy": "random", "k": int(params.pop("background_size"))}
            elif "kmeans_k" in params:
                bg = {"strategy": "kmeans", "k": int(params.pop("kmeans_k"))}

            op: dict[str, Any] = {"op": op_id, **params}

            # inherit model_ref and device if not set
            if model_ref and "model_ref" not in op:
                op["model_ref"] = model_ref
            if device and "device" not in op:
                op["device"] = device
            if bg:
                op["background"] = bg

            ops.append(self._normalize_op_dict(op, cfg))

        if not ops:
            logger.warning("ExplainabilityStep: no explainers/ops configured; no-op.")
        return ops

    @staticmethod
    def forbid_backend_imports() -> bool:
        return True

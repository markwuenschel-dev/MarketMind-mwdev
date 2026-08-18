from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pysrc.preprocessor.contracts.plan import CanonicalOp, MaterializationSpec, PreprocessingPlan


def normalize_preprocessing_plan(raw: Mapping[str, Any]) -> PreprocessingPlan:
    ops = tuple(
        CanonicalOp(
            name=str(op["name"]),
            params=dict(op.get("params", {})),
            provides=tuple(op.get("provides", ())),
            requires=tuple(op.get("requires", ())),
        )
        for op in raw.get("ops", ())
    )
    materialization_raw = raw.get("materialization", {})
    materialization = MaterializationSpec(
        format=str(materialization_raw.get("format", "polars")),
        schema_signature=str(materialization_raw.get("schema_signature", "")),
        partial_allowed=bool(materialization_raw.get("partial_allowed", False)),
    )
    return PreprocessingPlan(
        version=str(raw.get("version", "1.0")),
        ops=ops,
        group_by=tuple(raw.get("group_by", ())),
        materialization=materialization,
        metadata=dict(raw.get("metadata", {})),
    )

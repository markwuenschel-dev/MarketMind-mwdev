"""SearchSpaceFactory: build SearchSpaceSpec from config dicts."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.core.specs.search_space_spec import DimensionSpec, SearchSpaceSpec


def build_dimension_spec(raw: dict[str, Any]) -> DimensionSpec:
    return DimensionSpec(
        name=raw["name"],
        kind=raw["kind"],
        low=raw.get("low"),
        high=raw.get("high"),
        choices=tuple(raw["choices"]) if raw.get("choices") else None,
        prior=raw.get("prior", "uniform"),
    )


def build_search_space_spec(raw: dict[str, Any], spec_hash: str) -> SearchSpaceSpec:
    return SearchSpaceSpec(
        name=raw["name"],
        version=raw.get("version", "1.0.0"),
        model_type=raw["model_type"],
        spec_hash=spec_hash,
        dimensions=tuple(build_dimension_spec(d) for d in raw.get("dimensions", [])),
        fixed=dict(raw.get("fixed", {})),
    )


__all__ = ["build_dimension_spec", "build_search_space_spec"]

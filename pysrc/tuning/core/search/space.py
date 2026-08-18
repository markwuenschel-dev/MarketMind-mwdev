"""SearchSpace: runtime representation of a parameter search space."""

from __future__ import annotations

from dataclasses import dataclass

from pysrc.tuning.core.specs.search_space_spec import DimensionSpec, SearchSpaceSpec


@dataclass(frozen=True)
class SearchSpace:
    """Runtime search space built from a SearchSpaceSpec."""

    spec_hash: str
    model_type: str
    dimensions: tuple[DimensionSpec, ...]
    fixed: dict[str, object]

    @classmethod
    def from_spec(cls, spec: SearchSpaceSpec) -> SearchSpace:
        return cls(
            spec_hash=spec.spec_hash,
            model_type=spec.model_type,
            dimensions=spec.dimensions,
            fixed=dict(spec.fixed),
        )

    @property
    def dim_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)


__all__ = ["SearchSpace"]

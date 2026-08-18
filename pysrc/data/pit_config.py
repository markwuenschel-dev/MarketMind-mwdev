"""
Point-in-time (PIT) configuration: per-field policy (TTL, FillPolicy, MissingPolicy).

Phase I-A operates at daily granularity: TTL is expressed in integer days and staleness
is computed in days. Sub-day TTLs and quote-second resolution are out of scope for this phase.

Lookup order: exact field match -> namespace/glob match (e.g. price.*, fred.*) -> __default__.
resolve_field_config() returns a full config object per field, not just TTL.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum


class FillPolicy(StrEnum):
    """Whether to allow backward lookup from the snapshot anchor for gap-closing."""

    FORWARD = "forward"  # walk backward through visible rows for latest non-null
    REJECT = "reject"  # only the anchor row may satisfy that field; no backward walk


class MissingPolicy(StrEnum):
    """Behavior when a field is stale or missing after resolution."""

    WARN = "warn"  # resolve to NaN
    FAIL = "fail"  # raise StalenessError


@dataclass(frozen=True)
class ResolvedFieldConfig:
    """Per-field PIT policy: TTL (days, Phase I-A daily granularity) and fill/missing behavior."""

    ttl_days: int  # Phase I-A: daily granularity only; sub-day TTL out of scope
    fill_policy: FillPolicy
    missing_policy: MissingPolicy


def _default_config() -> ResolvedFieldConfig:
    return ResolvedFieldConfig(
        ttl_days=30,
        fill_policy=FillPolicy.FORWARD,
        missing_policy=MissingPolicy.WARN,
    )


@dataclass
class FieldTTLConfig:
    """
    Per-field and per-namespace PIT config. Lookup: exact field -> namespace glob -> __default__.

    - field_configs: exact field name -> ResolvedFieldConfig
    - namespace_configs: glob pattern (e.g. "price.*", "fred.*") -> ResolvedFieldConfig
    - default_config: used when no exact or namespace match
    """

    default_config: ResolvedFieldConfig = field(default_factory=_default_config)
    field_configs: dict[str, ResolvedFieldConfig] = field(default_factory=dict)
    namespace_configs: dict[str, ResolvedFieldConfig] = field(default_factory=dict)

    def resolve_field_config(self, field_name: str) -> ResolvedFieldConfig:
        """
        Resolve full config for a field. Lookup order:
        1. Exact field name in field_configs
        2. Namespace/glob: if field_name contains ".", try "namespace.*" in namespace_configs;
           also try any glob in namespace_configs that matches field_name (e.g. price.* matches price.close)
        3. __default__ from default_config
        """
        # Exact match
        if field_name in self.field_configs:
            return self.field_configs[field_name]
        # Namespace/glob: "price.close" -> try "price.*"
        if "." in field_name:
            namespace = field_name.split(".", 1)[0]
            glob_key = f"{namespace}.*"
            if glob_key in self.namespace_configs:
                return self.namespace_configs[glob_key]
        # Any glob matching field_name (e.g. fred.* matches fred.cpi)
        for pattern, cfg in self.namespace_configs.items():
            if "*" in pattern and fnmatch.fnmatch(field_name, pattern):
                return cfg
        return self.default_config


def resolve_field_config(field_name: str, config: FieldTTLConfig) -> ResolvedFieldConfig:
    """
    Return the resolved PIT config for the given field using config lookup order.

    Order: exact field match -> namespace/glob match -> __default__.
    """
    return config.resolve_field_config(field_name)

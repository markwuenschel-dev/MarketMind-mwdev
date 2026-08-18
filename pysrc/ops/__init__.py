from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_TO_MODULE: dict[str, str] = {
    "mm_logkit": "pysrc.ops.mm_logkit",
    "BoundLogger": "pysrc.ops.mm_logkit",
    "configure_logger": "pysrc.ops.mm_logkit",
    "JSONFormatter": "pysrc.ops.mm_logkit",
    "PersistentCache": "pysrc.ops.caching",
    "EnhancedCacheManager": "pysrc.ops.caching",
    "enhanced_cache": "pysrc.ops.caching",
    "hash_dataframe_deterministic": "pysrc.ops.caching",
    "hash_config": "pysrc.ops.caching",
    "versioned_key": "pysrc.ops.caching",
    "HashAlgorithm": "pysrc.ops.caching",
    "MultiTierClient": "pysrc.ops.multi_tier_cache",
    "multi_tier_cache": "pysrc.ops.multi_tier_cache",
    "SPAN_ATTR_UNKNOWN": "pysrc.ops.telemetry",
    "SPAN_BUNDLE_PROMOTE": "pysrc.ops.telemetry",
    "SPAN_DATAVIEW_AS_OF": "pysrc.ops.telemetry",
    "SPAN_GATE_EVALUATE": "pysrc.ops.telemetry",
    "SPAN_OP_EXECUTE": "pysrc.ops.telemetry",
    "tracer": "pysrc.ops.telemetry",
    "instrument": "pysrc.ops.observability",
    "init_observability": "pysrc.ops.observability",
    "get_metrics": "pysrc.ops.observability",
    "get_tracing": "pysrc.ops.observability",
    "get_logging": "pysrc.ops.observability",
    "get_logger": "pysrc.ops.observability",
    "MetricsManager": "pysrc.ops.observability",
    "TracingManager": "pysrc.ops.observability",
    "LoggingManager": "pysrc.ops.observability",
    "set_tenant": "pysrc.ops.observability",
    "get_tenant": "pysrc.ops.observability",
    "set_strategy": "pysrc.ops.observability",
    "get_strategy": "pysrc.ops.observability",
    "FastAPIMiddleware": "pysrc.ops.observability",
    "KafkaInstrumentor": "pysrc.ops.observability",
    "register_cache_hit_rate_gauges": "pysrc.ops.observability",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'pysrc.ops' has no attribute {name!r}")
    module = import_module(module_name)
    if name == "mm_logkit":
        return module
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)

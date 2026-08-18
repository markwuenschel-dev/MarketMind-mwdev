"""
Canonical OTel telemetry surface for MarketMind.

Uses the default no-op tracer when no exporter is configured.
Exporter configuration is operator responsibility at runtime.
Do not import opentelemetry-sdk in production code.
"""

from __future__ import annotations

import importlib
from typing import Any

try:
    trace: Any = importlib.import_module("opentelemetry.trace")
except ModuleNotFoundError:

    class _NoOpSpan:
        def __enter__(self) -> _NoOpSpan:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def set_attribute(self, name: str, value: object) -> None:
            return None

    class _NoOpTracer:
        def start_as_current_span(self, name: str) -> _NoOpSpan:
            _ = name
            return _NoOpSpan()

    class _NoOpTraceModule:
        def get_tracer(self, name: str) -> _NoOpTracer:
            _ = name
            return _NoOpTracer()

    trace = _NoOpTraceModule()

# Canonical span name constants — imported by tests and callers
SPAN_DATAVIEW_AS_OF = "pysrc.dataview.as_of"
SPAN_OP_EXECUTE = "pysrc.op.execute"
SPAN_GATE_EVALUATE = "pysrc.gate.evaluate"
SPAN_BUNDLE_PROMOTE = "pysrc.bundle.promote"

# Sentinel for unknown attribute values (stable contract for downstream / F-8)
SPAN_ATTR_UNKNOWN = "unknown"

tracer = trace.get_tracer("marketmind")

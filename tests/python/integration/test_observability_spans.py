"""
Stub: OTel span wiring for F-2 / GATE-I-F-02 criterion 4.
Full contract tests are F-8 / OI-28.
"""

from __future__ import annotations

import pytest


@pytest.mark.observability_stub
def test_otel_import() -> None:
    """opentelemetry-api is importable; default tracer is no-op."""
    from opentelemetry import trace

    t = trace.get_tracer("marketmind")
    assert t is not None


@pytest.mark.observability_stub
def test_span_names_are_canonical() -> None:
    """Canonical span name constants are defined and match spec."""
    from pysrc.ops.telemetry import (
        SPAN_BUNDLE_PROMOTE,
        SPAN_DATAVIEW_AS_OF,
        SPAN_GATE_EVALUATE,
        SPAN_OP_EXECUTE,
    )

    assert SPAN_DATAVIEW_AS_OF == "pysrc.dataview.as_of"
    assert SPAN_OP_EXECUTE == "pysrc.op.execute"
    assert SPAN_GATE_EVALUATE == "pysrc.gate.evaluate"
    assert SPAN_BUNDLE_PROMOTE == "pysrc.bundle.promote"


@pytest.mark.observability_stub
def test_span_attr_unknown_constant() -> None:
    from pysrc.ops.telemetry import SPAN_ATTR_UNKNOWN

    assert SPAN_ATTR_UNKNOWN == "unknown"

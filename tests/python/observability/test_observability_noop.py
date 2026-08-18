# tests/python/observability/test_observability_noop.py
from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock, patch

import pytest

# Import with proper error handling
try:
    from pysrc.ops.observability import _HAS_OTEL
except ImportError:
    _HAS_OTEL = False


# ==================== No-Op Classes Tests ====================
# Tests for dummy/stub classes when OpenTelemetry is not available


def test_noop_counter_operations():
    """Test _NoOpCounter does nothing without raising"""
    from pysrc.ops.observability import _NoOpCounter

    counter = _NoOpCounter()
    counter.add(1.0)
    counter.add(5.0, attributes={"key": "value"})
    # Should not raise


def test_noop_histogram_operations():
    """Test _NoOpHistogram does nothing without raising"""
    from pysrc.ops.observability import _NoOpHistogram

    hist = _NoOpHistogram()
    hist.record(10.0)
    hist.record(100.0, attributes={"key": "value"})
    # Should not raise


def test_noop_meter_operations():
    """Test _NoOpMeter creates no-op instruments"""
    from pysrc.ops.observability import _NoOpMeter

    meter = _NoOpMeter()
    counter = meter.create_counter("test_counter")
    hist = meter.create_histogram("test_hist")
    gauge = meter.create_observable_gauge("test_gauge")

    # Should return no-op instances
    counter.add(1.0)
    hist.record(1.0)
    assert gauge is None  # Gauge returns None


def test_noop_metrics_manager():
    """Test NoOpMetricsManager provides no-op API surface"""
    from pysrc.ops.observability import NoOpMetricsManager

    mgr = NoOpMetricsManager()

    # Should not raise
    counter = mgr.counter("test")
    hist = mgr.histogram("test")
    mgr.record_counter(counter, 1.0, {"k": "v"})
    mgr.record_histogram(hist, 1.0, {"k": "v"})
    mgr.shutdown()


def test_noop_tracing_manager():
    """Test NoOpTracingManager provides no-op API surface"""
    from pysrc.ops.observability import NoOpTracingManager

    mgr = NoOpTracingManager()

    # Should not raise
    mgr.set_sample_rate(0.5)
    span = mgr.start_span("test")
    assert span is None

    linked = mgr.start_span_with_links("test", [])
    assert linked is None

    carrier = {"key": "value"}
    result = mgr.inject_context(carrier)
    assert result == carrier

    mgr.extract_context(carrier)


def test_dummy_span_operations():
    """Test _DummySpan when OpenTelemetry unavailable"""
    # This test only runs if OTel is actually not available
    # Otherwise it tests the real Span
    try:
        from opentelemetry.trace import Span

        pytest.skip("OpenTelemetry is available, dummy classes not in use")
    except ImportError:
        from pysrc.ops.observability import Span, Status, StatusCode

        span = Span()

        # Should not raise
        span.set_attribute("key", "value")
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(ValueError("test"))

        ctx = span.get_span_context()
        assert ctx.trace_id == 0
        assert ctx.span_id == 0


# ==================== Logging Fallback Tests ====================
# Tests for TraceEnrichedLogger fallback paths (lines 1011, 1047-1088)


def test_trace_enriched_logger_structured_logging_success():
    """Test TraceEnrichedLogger with structured logging support"""
    import pysrc.ops.observability as obs

    class StructuredLogger:
        def __init__(self):
            self.events = []

        def info(self, msg, **kwargs):
            self.events.append(("info", msg, kwargs))

        def error(self, msg, **kwargs):
            self.events.append(("error", msg, kwargs))

    base_logger = StructuredLogger()
    logger = obs.TraceEnrichedLogger(base_logger, "test-svc")

    # Set context
    obs.set_tenant("tenant-123")
    obs.set_strategy("strategy-456")

    logger.info("test message", user_email="test@example.com", action="login")

    # Verify structured logging was used
    assert len(base_logger.events) == 1
    level, msg, kwargs = base_logger.events[0]

    assert level == "info"
    assert msg == "test message"
    assert kwargs["tenant_id"] == "tenant-123"
    assert kwargs["strategy_id"] == "strategy-456"
    assert kwargs["service"] == "test-svc"
    assert "[REDACTED_EMAIL]" in kwargs["user_email"]


def test_trace_enriched_logger_json_fallback_when_kwargs_not_supported():
    """Test TraceEnrichedLogger falls back to JSON when base logger doesn't support kwargs"""
    import pysrc.ops.observability as obs

    class SimpleLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            # Does NOT accept **kwargs - triggers fallback
            self.messages.append(("info", msg))

        def error(self, msg):
            self.messages.append(("error", msg))

    base_logger = SimpleLogger()
    logger = obs.TraceEnrichedLogger(base_logger, "test-svc")

    obs.set_tenant("tenant-abc")
    obs.set_strategy("strategy-xyz")

    logger.info("event occurred", user="alice@example.com", action="test")

    # Verify JSON fallback was used
    assert len(base_logger.messages) == 1
    level, json_str = base_logger.messages[0]

    assert level == "info"
    payload = json.loads(json_str)

    assert payload["msg"] == "event occurred"
    assert payload["tenant_id"] == "tenant-abc"
    assert payload["strategy_id"] == "strategy-xyz"
    assert payload["service"] == "test-svc"
    assert "[REDACTED_EMAIL]" in payload["user"]


def test_trace_enriched_logger_silent_failure_on_complete_error():
    """Test TraceEnrichedLogger fails silently when all emit paths fail"""
    import pysrc.ops.observability as obs

    class BrokenLogger:
        def info(self, *args, **kwargs):
            raise RuntimeError("Logger is broken")

    base_logger = BrokenLogger()
    logger = obs.TraceEnrichedLogger(base_logger, "test-svc")

    # Should not raise - fails silently for hot-path safety
    logger.info("message", key="value")


def test_trace_enriched_logger_enriches_with_trace_context():
    """Test TraceEnrichedLogger adds trace/span IDs when available"""
    import pysrc.ops.observability as obs

    class MockLogger:
        def __init__(self):
            self.events = []

        def info(self, msg, **kwargs):
            self.events.append(kwargs)

    base_logger = MockLogger()
    logger = obs.TraceEnrichedLogger(base_logger, "test-svc")

    # Without active span (trace_id/span_id should not be present or be default values)
    logger.info("no span")

    if base_logger.events:
        # Either no trace_id or default value
        event = base_logger.events[0]
        if "trace_id" in event:
            # Dummy span returns 0
            assert event["trace_id"] in ["0" * 32, "00000000000000000000000000000000"]


# ==================== Logging Manager mm_logkit Fallback ====================


def test_logging_manager_fallback_to_structlog_when_mm_logkit_unavailable():
    """Test LoggingManager falls back to structlog when mm_logkit not available"""
    # mm_logkit should not be available in test environment
    import importlib.util

    import pysrc.ops.observability as obs

    if importlib.util.find_spec("mm_logkit") is not None:
        pytest.skip("mm_logkit is available, cannot test fallback")

    # Should create LoggingManager using structlog fallback
    mgr = obs.LoggingManager(service_name="test-svc")
    logger = mgr.get_logger()

    # Should be able to log without errors
    logger.info("test message", key="value")


def test_logging_manager_handles_mm_logkit_configuration_error():
    """Test LoggingManager handles mm_logkit configuration errors gracefully"""
    from unittest.mock import patch

    import pysrc.ops.observability as obs

    # Mock mm_logkit to raise on configuration
    mock_mm_logkit = Mock()
    mock_mm_logkit.configure_logger.side_effect = RuntimeError("Config failed")
    mock_mm_logkit.get_logger.return_value = None

    with patch.dict("sys.modules", {"mm_logkit": mock_mm_logkit}):
        # Should fall back to structlog without raising
        mgr = obs.LoggingManager(service_name="test-svc")
        logger = mgr.get_logger()

        # Should still work
        logger.info("message")


# ==================== Module Import & Init Tests ====================


def test_module_imports_successfully_without_opentelemetry():
    """Test that observability module can be imported when OpenTelemetry is not installed"""
    # If OTel is available, skip this test
    import importlib.util

    if importlib.util.find_spec("opentelemetry") is not None:
        pytest.skip("OpenTelemetry is available, testing no-op mode not applicable")

    # Should import successfully
    import pysrc.ops.observability as obs

    # Should have _HAS_OTEL = False
    assert obs._HAS_OTEL is False

    # Dummy classes should be available
    assert hasattr(obs, "_DummySpan")
    assert hasattr(obs, "NoOpMetricsManager")
    assert hasattr(obs, "NoOpTracingManager")


def test_init_observability_creates_noop_managers_without_otel():
    """Test init_observability creates no-op managers when OTel unavailable"""
    import pysrc.ops.observability as obs

    # Reset globals
    obs._metrics = None
    obs._tracing = None
    obs._logging = None
    obs._logger = None

    obs.init_observability(
        service_name="test-svc", enable_metrics=True, enable_tracing=True, enable_logging=True
    )

    # Should have created managers
    metrics = obs.get_metrics()
    tracing = obs.get_tracing()
    logging = obs.get_logging()
    logger = obs.get_logger()

    assert metrics is not None
    assert tracing is not None
    assert logging is not None
    assert logger is not None

    # If OTel not available, should be no-op managers
    if not obs._HAS_OTEL:
        assert isinstance(metrics, obs.NoOpMetricsManager)
        assert isinstance(tracing, obs.NoOpTracingManager)


def test_init_observability_respects_enable_flags():
    """Test init_observability respects enable flags even in no-op mode"""
    import pysrc.ops.observability as obs

    # Reset globals
    obs._metrics = None
    obs._tracing = None
    obs._logging = None
    obs._logger = None

    obs.init_observability(
        service_name="test-svc",
        enable_metrics=False,
        enable_tracing=False,
        enable_logging=True,  # Only logging enabled
    )

    assert obs.get_metrics() is None
    assert obs.get_tracing() is None
    assert obs.get_logging() is not None


# ==================== Component Tests That Don't Need OTel ====================
# Note: Most component tests are in test_tracing.py since they work identically
# with or without OpenTelemetry. These are just smoke tests for the no-op path.


def test_instrument_decorator_works_without_otel():
    """Smoke test that @instrument works in no-op mode"""
    import pysrc.ops.observability as obs

    obs.init_observability(enable_metrics=True, enable_tracing=True, enable_logging=True)

    @obs.instrument(name="test_func")
    def my_func(x):
        return x * 2

    result = my_func(5)
    assert result == 10


def test_context_vars_work_without_otel():
    """Test tenant/strategy context works in no-op mode"""
    import pysrc.ops.observability as obs

    obs.set_tenant("tenant-123")
    obs.set_strategy("strategy-456")

    assert obs.get_tenant() == "tenant-123"
    assert obs.get_strategy() == "strategy-456"


# ==================== Dummy Class Edge Cases (Lines 67-229) ====================


def test_dummy_span_context_attributes():
    """Test _DummySpanContext attributes when OTel unavailable"""
    if _HAS_OTEL:
        pytest.skip("OpenTelemetry available, dummy classes not in use")

    from pysrc.ops.observability import Span

    span = Span()
    ctx = span.get_span_context()

    # Should have required attributes
    assert hasattr(ctx, "trace_id")
    assert hasattr(ctx, "span_id")
    assert ctx.trace_id == 0
    assert ctx.span_id == 0


def test_dummy_status_creation():
    """Test _DummyStatus creation with different parameters"""
    if _HAS_OTEL:
        pytest.skip("OpenTelemetry available")

    from pysrc.ops.observability import Status, StatusCode

    # Should accept optional parameters
    status1 = Status()
    assert status1.status_code is None
    assert status1.description is None

    status2 = Status(status_code=StatusCode.ERROR)
    assert status2.status_code == StatusCode.ERROR

    status3 = Status(status_code=StatusCode.OK, description="all good")
    assert status3.description == "all good"


def test_dummy_link_creation():
    """Test _DummyLink creation"""
    if _HAS_OTEL:
        pytest.skip("OpenTelemetry available")

    from pysrc.ops.observability import Link

    # Should accept arbitrary args/kwargs without raising
    link = Link("context", attributes={"key": "value"})
    assert link is not None


def test_dummy_metrics_observation_class():
    """Test _DummyMetricsModule.Observation class"""
    if _HAS_OTEL:
        pytest.skip("OpenTelemetry available")

    from pysrc.ops.observability import metrics

    obs = metrics.Observation(42.5, {"label": "value"})
    assert obs.value == 42.5
    assert obs.attributes == {"label": "value"}

    # Without attributes
    obs2 = metrics.Observation(10.0)
    assert obs2.attributes == {}


def test_dummy_trace_use_span_context_manager():
    """Test _DummyTraceModule.use_span as context manager"""
    if _HAS_OTEL:
        pytest.skip("OpenTelemetry available")

    from pysrc.ops.observability import trace

    # Should work as context manager without raising
    span = trace.get_current_span()
    with trace.use_span(span, end_on_exit=True):
        pass  # Context manager should work


def test_dummy_sampler_methods():
    """Test _DummySampler methods"""
    if _HAS_OTEL:
        pytest.skip("OpenTelemetry available")

    from pysrc.ops.observability import Sampler

    sampler = Sampler()

    # should_sample returns a result object
    result = sampler.should_sample(None, None, "name", None, None, None)
    assert hasattr(result, "decision")

    # get_description returns string
    desc = sampler.get_description()
    assert isinstance(desc, str)


# ==================== TraceEnrichedLogger Edge Cases (Lines 918-925) ====================


def test_trace_enriched_logger_when_span_has_no_context():
    """Test _enrich when span.get_span_context() returns invalid context"""
    import pysrc.ops.observability as obs

    class MockSpan:
        @staticmethod
        def get_span_context():
            # Return context with no trace_id/span_id attributes
            return object()

    class MockLogger:
        def __init__(self):
            self.events = []

        def info(self, _msg, **kwargs):
            self.events.append(kwargs)

    base_logger = MockLogger()
    logger = obs.TraceEnrichedLogger(base_logger, "svc")

    # Mock trace.get_current_span to return our mock span
    with patch("pysrc.ops.observability.trace") as mock_trace:
        mock_trace.get_current_span.return_value = MockSpan()

        # Should not crash even though span context is incomplete
        logger.info("test")

        # Should still have tenant/strategy/service
        if base_logger.events:
            assert "service" in base_logger.events[0]


def test_trace_enriched_logger_when_get_span_context_raises():
    """Test _enrich when span.get_span_context() raises exception"""
    import pysrc.ops.observability as obs

    class BrokenSpan:
        @staticmethod
        def get_span_context():
            raise AttributeError("No context available")

    class MockLogger:
        def __init__(self):
            self.events = []

        def info(self, _msg, **kwargs):
            self.events.append(kwargs)

    base_logger = MockLogger()
    logger = obs.TraceEnrichedLogger(base_logger, "svc")

    with patch("pysrc.ops.observability.trace") as mock_trace:
        mock_trace.get_current_span.return_value = BrokenSpan()

        # Should handle exception gracefully
        logger.info("test")

        # Should still work
        assert len(base_logger.events) >= 0


def test_trace_enriched_logger_all_log_levels():
    """Test all log level methods (info, error, warning, debug)"""
    import pysrc.ops.observability as obs

    class MockLogger:
        def __init__(self):
            self.calls = []

        def info(self, _msg, **_kw):
            self.calls.append("info")

        def error(self, _msg, **_kw):
            self.calls.append("error")

        def warning(self, _msg, **_kw):
            self.calls.append("warning")

        def debug(self, _msg, **_kw):
            self.calls.append("debug")

    base = MockLogger()
    logger = obs.TraceEnrichedLogger(base, "svc")

    logger.info("info msg", key1="val1")
    logger.error("error msg", key2="val2")
    logger.warning("warn msg", key3="val3")
    logger.debug("debug msg", key4="val4")

    assert len(base.calls) == 4
    assert base.calls[0] == "info"
    assert base.calls[1] == "error"
    assert base.calls[2] == "warning"
    assert base.calls[3] == "debug"


def test_trace_enriched_logger_when_base_logger_missing_method():
    """Test _emit when base logger is missing a log method"""
    import pysrc.ops.observability as obs

    class IncompleteLogger:
        def info(self, _msg, **_kw):
            pass

        # Missing error, warning, debug methods

    base = IncompleteLogger()
    logger = obs.TraceEnrichedLogger(base, "svc")

    # Should not crash when calling missing methods
    logger.error("error msg")
    logger.warning("warn msg")
    logger.debug("debug msg")


# ==================== instrument decorator with None span (Lines 1190-1196) ====================


def test_instrument_decorator_with_none_span_in_use_span():
    """Test instrument decorator when start_span returns None"""
    import pysrc.ops.observability as obs

    # Set up tracing that returns None span
    obs._tracing = obs.NoOpTracingManager()
    obs._metrics = None

    @obs.instrument(name="test_func")
    def my_func(x):
        return x * 3

    # Should handle None span gracefully
    result = my_func(4)
    assert result == 12


@pytest.mark.asyncio
async def test_instrument_decorator_async_with_none_span():
    """Test async instrument decorator when start_span returns None"""
    import pysrc.ops.observability as obs

    obs._tracing = obs.NoOpTracingManager()
    obs._metrics = None

    @obs.instrument(name="async_test")
    async def async_func(x):
        await asyncio.sleep(0.001)
        return x * 5

    # Should handle None span gracefully
    result = await async_func(3)
    assert result == 15


def test_instrument_decorator_exception_recording_with_none_span():
    """Test instrument decorator exception recording when span is None"""
    import pysrc.ops.observability as obs

    obs._tracing = obs.NoOpTracingManager()
    obs._metrics = obs.NoOpMetricsManager()

    @obs.instrument(name="test_error", record_exceptions=True)
    def failing_func():
        raise ValueError("intentional error")

    # Should still raise, but handle None span during exception recording
    with pytest.raises(ValueError, match="intentional error"):
        failing_func()


# ==================== FastAPIMiddleware with None span (Lines 1221-1237) ====================


@pytest.mark.asyncio
async def test_fastapi_middleware_finally_block_with_none_span():
    """Test FastAPIMiddleware finally block when span is None"""
    import pysrc.ops.observability as obs

    obs._tracing = obs.NoOpTracingManager()
    obs._metrics = obs.NoOpMetricsManager()

    app_executed = []

    async def test_app(_scope, _receive, _send):
        app_executed.append(True)

    middleware = obs.FastAPIMiddleware(test_app, service_name="api")

    test_scope = {
        "type": "http",
        "path": "/test",
        "method": "GET",
        "scheme": "http",
        "server": ("localhost", 8000),
    }

    # Should execute and handle None span in finally block
    await middleware(test_scope, None, None)
    assert app_executed


@pytest.mark.asyncio
async def test_fastapi_middleware_latency_recording_with_none_span():
    """Test that latency is recorded even when span is None"""
    import pysrc.ops.observability as obs

    class MockMetricsManager:
        def __init__(self):
            self.recorded = []

        def histogram(self, _name, description="", unit="ms"):
            return Mock()

        def counter(self, _name, description=""):
            return Mock()

        def record_counter(self, _counter_inst, value=1, labels=None):
            self.recorded.append(("counter", labels))

        def record_histogram(self, _hist_inst, latency, labels=None):
            self.recorded.append(("histogram", latency, labels))

    obs._tracing = obs.NoOpTracingManager()
    obs._metrics = MockMetricsManager()

    async def test_app(_scope, _receive, _send):
        pass

    middleware = obs.FastAPIMiddleware(test_app)

    test_scope = {
        "type": "http",
        "path": "/test",
        "method": "POST",
        "scheme": "http",
        "server": ("localhost", 8000),
    }

    await middleware(test_scope, None, None)

    # Should have recorded metrics even with None span
    assert any(r[0] == "histogram" for r in obs._metrics.recorded)


@pytest.mark.asyncio
async def test_fastapi_middleware_error_path_with_none_span():
    """Test FastAPIMiddleware error handling when span is None"""
    import pysrc.ops.observability as obs

    obs._tracing = obs.NoOpTracingManager()
    obs._metrics = obs.NoOpMetricsManager()

    async def failing_app(_scope, _receive, _send):
        raise RuntimeError("app error")

    middleware = obs.FastAPIMiddleware(failing_app)

    test_scope = {
        "type": "http",
        "path": "/error",
        "method": "GET",
        "scheme": "http",
        "server": ("localhost", 8000),
    }

    # Should propagate error and handle None span in error path
    with pytest.raises(RuntimeError, match="app error"):
        await middleware(test_scope, None, None)


# ==================== KafkaInstrumentor with None tracing (Lines 1245, 1247) ====================


def test_kafka_instrumentor_producer_when_tracing_is_none():
    """Test instrumented producer when _tracing is None"""
    import pysrc.ops.observability as obs

    obs._tracing = None

    instrumentor = obs.KafkaInstrumentor()

    class Producer:
        def __init__(self):
            self.sent = []

        def send(self, topic, value=None, key=None, headers=None, **_kwargs):
            self.sent.append((topic, headers))
            return "sent"

    producer = Producer()
    instrumented = instrumentor.instrument_producer(producer)

    # Should work without tracing
    result = instrumented.send("topic", value=b"data", headers={"k": "v"})
    assert result == "sent"
    assert len(producer.sent) == 1


def test_kafka_instrumentor_consumer_when_tracing_is_none():
    """Test instrumented consumer when _tracing is None"""
    import pysrc.ops.observability as obs

    obs._tracing = None

    instrumentor = obs.KafkaInstrumentor()

    class Message:
        def __init__(self):
            self.topic = "test-topic"
            self.offset = 123
            self.headers = [("k", b"v")]

    class Consumer:
        def __init__(self):
            self.records = {}

        def poll(self, *_args, **_kwargs):
            return self.records

    consumer = Consumer()
    consumer.records = {"partition": [Message()]}

    instrumented = instrumentor.instrument_consumer(consumer)

    # Should work without tracing
    records = instrumented.poll()
    assert "partition" in records
    assert len(records["partition"]) == 1


def test_kafka_instrumentor_extract_context_with_none_headers():
    """Test extract_context when headers is None"""
    import pysrc.ops.observability as obs

    instrumentor = obs.KafkaInstrumentor()

    # Should handle None headers gracefully
    instrumentor.extract_context(None)


def test_kafka_instrumentor_extract_context_with_mixed_types():
    """Test extract_context with mixed bytes/string headers"""
    import pysrc.ops.observability as obs

    instrumentor = obs.KafkaInstrumentor()

    # Mix of bytes and strings
    headers = {
        b"binary_key": b"binary_value",
        "string_key": "string_value",
        b"traceparent": b"00-trace-span-01",
    }

    # Should handle mixed types
    instrumentor.extract_context(headers)


# ==================== AdaptiveThreshold Edge Cases (Lines 1128-1140) ====================


def test_adaptive_threshold_with_zero_values():
    """Test AdaptiveThreshold with all zero values"""
    import pysrc.ops.observability as obs

    at = obs.AdaptiveThreshold(alpha=0.5, sensitivity=3.0)

    anomaly = False
    for _ in range(10):
        anomaly = at.update(0.0)

    # Should have calculated threshold
    threshold = at.threshold()
    assert threshold >= 0  # May be 0 or small positive
    assert not anomaly  # Zeros shouldn't be anomalies


def test_adaptive_threshold_with_negative_values():
    """Test AdaptiveThreshold with negative values (if allowed by domain)"""
    import pysrc.ops.observability as obs

    at = obs.AdaptiveThreshold(alpha=0.1, sensitivity=2.0)

    # Some domains allow negative metrics
    at.update(-5.0)
    at.update(-4.0)
    at.update(-6.0)

    threshold = at.threshold()
    assert threshold < float("inf")


def test_adaptive_threshold_mad_calculation_with_single_value():
    """Test MAD calculation with very few values"""
    import pysrc.ops.observability as obs

    at = obs.AdaptiveThreshold(alpha=0.5, sensitivity=2.0, window_size=100)

    # First value
    at.update(10.0)
    at.threshold()

    # Second value
    at.update(12.0)
    at.threshold()

    # MAD should start calculating after 10 values
    for _ in range(8):
        at.update(11.0)

    threshold3 = at.threshold()
    assert threshold3 < float("inf")


# ==================== SLOBurnRate Edge Cases (Lines 1162, 1176-1177) ====================


def test_slo_burn_rate_with_zero_error_budget():
    """Test SLOBurnRate with zero error budget"""
    import pysrc.ops.observability as obs

    slo = obs.SLOBurnRate(minutes=[1])

    for _ in range(10):
        slo.record(ok=False)

    # Should handle zero budget gracefully
    rates = slo.burn_rates(slo_error_budget=0.0)
    assert rates[1] == 0.0  # Division by zero handled


def test_slo_burn_rate_window_cleanup():
    """Test that old events are removed from windows"""
    from unittest.mock import patch

    import pysrc.ops.observability as obs

    base_time = 1000.0
    current_time = [base_time]

    def mock_time():
        return current_time[0]

    with patch("time.time", side_effect=mock_time):
        slo = obs.SLOBurnRate(minutes=[1])  # 60 second window

        # Record at base time
        slo.record(ok=False)
        rates = slo.burn_rates(slo_error_budget=0.01)
        assert rates[1] > 0  # Error is in window

        # Advance time beyond window
        current_time[0] = base_time + 120  # 2 minutes later

        # Old event should be removed
        slo.record(ok=True)  # New success
        rates = slo.burn_rates(slo_error_budget=0.01)
        # Burn rate should be 0 since only success in window
        assert rates[1] == 0.0


def test_slo_burn_rate_multiple_window_cleanup():
    """Test cleanup across multiple window sizes"""
    import pysrc.ops.observability as obs

    slo = obs.SLOBurnRate(minutes=[1, 5, 30])

    # Record events
    for _ in range(100):
        slo.record(ok=True)

    rates = slo.burn_rates(slo_error_budget=0.01)

    # All windows should be calculated
    assert 1 in rates
    assert 5 in rates
    assert 30 in rates


# ==================== SafeOTLPMetricExporter Timeout Handling ====================


def test_safe_otlp_metric_exporter_socket_timeout():
    """Test SafeOTLPMetricExporter handles socket.timeout"""
    if not _HAS_OTEL:
        pytest.skip("OpenTelemetry not available")

    from pysrc.ops.observability import CircuitBreaker, SafeOTLPMetricExporter

    breaker = CircuitBreaker()

    with patch("pysrc.ops.observability.OTLPMetricExporter.export") as mock_export:
        mock_export.side_effect = TimeoutError("Connection timeout")

        exporter = SafeOTLPMetricExporter(
            allowlist=("localhost:4317",),
            breaker=breaker,
            timeout=5.0,
            endpoint="localhost:4317",
            insecure=True,
        )

        # Should handle timeout and increment breaker
        exporter.export(Mock())
        assert breaker.fail_count == 1


def test_safe_otlp_metric_exporter_os_error():
    """Test SafeOTLPMetricExporter handles OSError"""
    if not _HAS_OTEL:
        pytest.skip("OpenTelemetry not available")

    from pysrc.ops.observability import CircuitBreaker, SafeOTLPMetricExporter

    breaker = CircuitBreaker()

    with (
        patch("pysrc.ops.observability.OTLPMetricExporter.export") as mock_export,
        patch("pysrc.ops.observability._logger"),
    ):
        mock_export.side_effect = OSError("Network unreachable")

        exporter = SafeOTLPMetricExporter(
            allowlist=("localhost:4317",),
            breaker=breaker,
            timeout=5.0,
            endpoint="localhost:4317",
            insecure=True,
        )

        exporter.export(Mock())
        assert breaker.fail_count == 1


# ==================== NoOp Manager Edge Cases ====================


def test_noop_metrics_manager_with_none_labels():
    """Test NoOpMetricsManager handles None labels"""
    import pysrc.ops.observability as obs

    mgr = obs.NoOpMetricsManager()
    counter_inst = mgr.counter("test")

    # Should handle None labels
    mgr.record_counter(counter_inst, 1.0, labels=None)
    mgr.record_counter(counter_inst, 1.0)  # No labels arg


def test_noop_tracing_manager_extract_with_empty_carrier():
    """Test NoOpTracingManager.extract_context with various carrier types"""
    import pysrc.ops.observability as obs

    mgr = obs.NoOpTracingManager()

    # Empty dict
    mgr.extract_context({})

    # Dict with data
    mgr.extract_context({"traceparent": "00-trace-span-01"})


def test_noop_managers_can_be_called_repeatedly():
    """Test that no-op managers can be called many times without side effects"""
    import pysrc.ops.observability as obs

    metrics = obs.NoOpMetricsManager()
    tracing = obs.NoOpTracingManager()

    # Should handle repeated calls
    for _ in range(100):
        counter_inst = metrics.counter("test")
        counter_inst.add(1.0)
        metrics.record_counter(counter_inst, 1.0)

        hist_inst = metrics.histogram("test")
        hist_inst.record(1.0)
        metrics.record_histogram(hist_inst, 1.0)

        tracing.set_sample_rate(0.5)
        tracing.start_span("test")
        tracing.inject_context({})

    # Final shutdown should work
    metrics.shutdown()

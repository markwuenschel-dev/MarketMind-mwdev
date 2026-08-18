# tests/python/observability/test_tracing.py
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

import pysrc.ops.observability as obs
from pysrc.ops.observability import (
    _HAS_OTEL,
    AdaptiveRatioSampler,
    AdaptiveThreshold,
    BoundedEventQueue,
    CardinalityLimiter,
    CircuitBreaker,
    FastAPIMiddleware,
    KafkaInstrumentor,
    MetricConfig,
    MetricsManager,
    NoOpMetricsManager,
    NoOpTracingManager,
    PiiRedactionSpanProcessor,
    PIIRedactor,
    SafeOTLPSpanExporter,
    SLOBurnRate,
    TracingConfig,
    TracingManager,
    get_logging,
    get_metrics,
    get_tracing,
    init_observability,
    instrument,
)
from tests.python.conftest import allocate_port
from tests.python.infra.compat_layer import patched_attr
from tests.python.infra.matrix import matrix

# ================== Circuit Breaker Tests ==================


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(fail_threshold=3, reset_after_sec=1)
    assert not cb.is_open()

    cb.on_failure()
    cb.on_failure()
    assert not cb.is_open()

    cb.on_failure()
    assert cb.is_open()


def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker(fail_threshold=2, reset_after_sec=1)
    cb.on_failure()
    cb.on_success()
    assert not cb.is_open()
    assert cb.fail_count == 0


def test_circuit_breaker_half_open_after_timeout():
    cb = CircuitBreaker(fail_threshold=2, reset_after_sec=0.05)
    cb.on_failure()
    cb.on_failure()
    assert cb.is_open()

    time.sleep(0.1)
    assert not cb.is_open()
    assert cb.fail_count == 1


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_circuit_breaker_thread_safe(workers):
    # Use regular parametrize for single-parameter tests
    cb = CircuitBreaker(fail_threshold=10, reset_after_sec=1)

    def worker():
        for _ in range(5):
            cb.on_failure()
            time.sleep(0.001)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker) for _ in range(workers)]
        for f in futures:
            f.result()

    assert cb.fail_count == workers * 5
    assert cb.is_open()


# ================== Cardinality Limiter Tests ==================


def test_cardinality_limiter_allows_within_limit():
    limiter = CardinalityLimiter(max_keys_per_label=3)

    assert limiter.sanitize("env", "prod") == "prod"
    assert limiter.sanitize("env", "staging") == "staging"
    assert limiter.sanitize("env", "dev") == "dev"
    assert limiter.overflow_count() == 0


def test_cardinality_limiter_hashes_overflow():
    limiter = CardinalityLimiter(max_keys_per_label=2)

    limiter.sanitize("env", "prod")
    limiter.sanitize("env", "staging")

    result = limiter.sanitize("env", "dev")
    assert result.startswith("h_")
    assert limiter.overflow_count() == 1


def test_cardinality_limiter_stats():
    limiter = CardinalityLimiter(max_keys_per_label=5)

    for i in range(3):
        limiter.sanitize("method", f"GET_{i}")
    for i in range(4):
        limiter.sanitize("path", f"/api/v{i}")

    stats = limiter.stats()
    assert stats["labels"]["method"] == 3
    assert stats["labels"]["path"] == 4
    assert stats["overflow"] == 0


@given(
    label_key=st.text(
        min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))
    ),
    values=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=50),
)
@settings(deadline=None, max_examples=100)
@seed(12345)
def test_cardinality_limiter_property_no_crash(label_key, values):
    limiter = CardinalityLimiter(max_keys_per_label=10)

    for val in values:
        result = limiter.sanitize(label_key, val)
        assert isinstance(result, str)
        assert len(result) > 0


# ================== PII Redactor Tests ==================


def test_pii_redactor_email():
    redactor = PIIRedactor()
    text = "Contact us at support@example.com for help"
    result = redactor.redact_text(text)
    assert "support@example.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_pii_redactor_ssn():
    redactor = PIIRedactor()
    text = "SSN: 123-45-6789"
    result = redactor.redact_text(text)
    assert "123-45-6789" not in result
    assert "[REDACTED_SSN]" in result


def test_pii_redactor_phone():
    redactor = PIIRedactor()
    text = "Call 555-123-4567"
    result = redactor.redact_text(text)
    assert "555-123-4567" not in result
    assert "[REDACTED_PHONE]" in result


def test_pii_redactor_credit_card():
    redactor = PIIRedactor()
    text = "Card: 4532-1234-5678-9010"
    result = redactor.redact_text(text)
    assert "4532-1234-5678-9010" not in result
    assert "[REDACTED_CREDIT_CARD]" in result


def test_pii_redactor_api_key():
    redactor = PIIRedactor()
    text = 'api_key="secret123"'
    result = redactor.redact_text(text)
    assert "secret123" not in result
    assert "[REDACTED_API_KEY]" in result


def test_pii_redactor_dict_fast_keys():
    redactor = PIIRedactor(fast_keys_allowlist={"tenant_id", "service"})
    data = {"tenant_id": "tenant-123", "email": "user@example.com", "service": "api"}
    result = redactor.redact_dict(data)
    assert result["tenant_id"] == "tenant-123"
    assert result["service"] == "api"
    assert "[REDACTED_EMAIL]" in result["email"]


def test_pii_redactor_dict_nested():
    redactor = PIIRedactor()
    data = {"user": {"email": "test@example.com", "phone": "555-1234"}}
    result = redactor.redact_dict(data)
    assert "[REDACTED_EMAIL]" in result["user"]["email"]


def test_pii_redactor_dict_list():
    redactor = PIIRedactor()
    data = {"emails": ["a@b.com", "c@d.com"]}
    result = redactor.redact_dict(data)
    assert all("[REDACTED_EMAIL]" in e for e in result["emails"])


@given(text=st.text(min_size=0, max_size=200), should_contain_email=st.booleans())
@settings(deadline=None, max_examples=100)
@seed(12345)
def test_pii_redactor_property_no_crash(text, should_contain_email):
    if should_contain_email:
        text = f"{text} test@example.com"

    redactor = PIIRedactor()
    result = redactor.redact_text(text)
    assert isinstance(result, str)


# ================== Adaptive Threshold Tests ==================


def test_adaptive_threshold_initial_infinity():
    at = AdaptiveThreshold()
    assert at.threshold() == float("inf")


def test_adaptive_threshold_learns_baseline():
    at = AdaptiveThreshold(alpha=0.5, sensitivity=2.0)

    at.update(10.0)
    at.update(12.0)
    at.update(11.0)

    threshold = at.threshold()
    assert threshold > 0
    assert threshold < float("inf")


def test_adaptive_threshold_detects_anomaly():
    at = AdaptiveThreshold(alpha=0.1, sensitivity=3.0, window_size=20)

    for _ in range(15):
        at.update(10.0)

    anomaly = at.update(100.0)
    assert anomaly


def test_adaptive_threshold_no_anomaly_in_range():
    at = AdaptiveThreshold(alpha=0.1, sensitivity=3.0)

    for _ in range(10):
        anomaly = at.update(10.0)

    assert not anomaly


@given(values=st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=10, max_size=100))
@settings(deadline=None, max_examples=50)
@seed(12345)
def test_adaptive_threshold_property_converges(values):
    at = AdaptiveThreshold()

    for v in values:
        at.update(v)

    threshold = at.threshold()
    assert 0 < threshold < float("inf")


# ================== SLO Burn Rate Tests ==================


def test_slo_burn_rate_tracks_success():
    slo = SLOBurnRate(minutes=[1])

    for _ in range(100):
        slo.record(ok=True)

    rates = slo.burn_rates(slo_error_budget=0.01)
    assert rates[1] == 0.0


def test_slo_burn_rate_tracks_errors():
    slo = SLOBurnRate(minutes=[1])

    for _ in range(90):
        slo.record(ok=True)
    for _ in range(10):
        slo.record(ok=False)

    rates = slo.burn_rates(slo_error_budget=0.01)
    assert rates[1] > 0


def test_slo_burn_rate_multiple_windows():
    slo = SLOBurnRate(minutes=[1, 5, 30])

    for _ in range(100):
        slo.record(ok=True)

    rates = slo.burn_rates(slo_error_budget=0.01)
    assert 1 in rates
    assert 5 in rates
    assert 30 in rates


def test_slo_burn_rate_no_events():
    slo = SLOBurnRate(minutes=[1])
    rates = slo.burn_rates(slo_error_budget=0.01)
    assert rates[1] == 0.0


# ================== Bounded Event Queue Tests ==================


def test_bounded_event_queue_put_get():
    queue = BoundedEventQueue(maxsize=10)

    ev = obs._Event("counter", Mock(), 1.0, {})
    assert queue.put_nowait(ev)

    batch = queue.get_batch(1)
    assert len(batch) == 1
    assert batch[0].kind == "counter"


def test_bounded_event_queue_drops_overflow():
    queue = BoundedEventQueue(maxsize=3)

    for i in range(3):
        ev = obs._Event("counter", Mock(), float(i), {})
        assert queue.put_nowait(ev)

    ev = obs._Event("counter", Mock(), 999.0, {})
    assert not queue.put_nowait(ev)
    assert queue.dropped() == 1


def test_bounded_event_queue_size():
    queue = BoundedEventQueue(maxsize=10)

    for i in range(5):
        queue.put_nowait(obs._Event("counter", Mock(), float(i), {}))

    assert queue.size() == 5


def test_bounded_event_queue_concurrent_put():
    queue = BoundedEventQueue(maxsize=1000)

    def worker(n):
        for i in range(10):
            ev = obs._Event("counter", Mock(), float(i), {"worker": str(n)})
            queue.put_nowait(ev)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker, i) for i in range(4)]
        for f in futures:
            f.result()

    assert queue.size() <= 40


# ================== Context Tests ==================


def test_tenant_context():
    obs.set_tenant("tenant-abc")
    assert obs.get_tenant() == "tenant-abc"

    obs.set_tenant("tenant-xyz")
    assert obs.get_tenant() == "tenant-xyz"


def test_strategy_context():
    obs.set_strategy("strategy-momentum")
    assert obs.get_strategy() == "strategy-momentum"

    obs.set_strategy("strategy-arbitrage")
    assert obs.get_strategy() == "strategy-arbitrage"


# ================== Adaptive Ratio Sampler Tests ==================


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_adaptive_ratio_sampler_clamps_rate():
    sampler = AdaptiveRatioSampler(initial_rate=0.5)
    assert sampler._rate == 0.5

    sampler.set_rate(1.5)
    assert sampler._rate == 1.0

    sampler.set_rate(-0.5)
    assert sampler._rate == 0.0


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_adaptive_ratio_sampler_description():
    sampler = AdaptiveRatioSampler(initial_rate=0.75)
    desc = sampler.get_description()
    assert "0.75" in desc or "0.7500" in desc


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_adaptive_ratio_sampler_thread_safe():
    sampler = AdaptiveRatioSampler(initial_rate=0.5)

    def worker():
        for i in range(10):
            sampler.set_rate(float(i) / 10.0)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker) for _ in range(4)]
        for f in futures:
            f.result()

    assert 0.0 <= sampler._rate <= 1.0


# ================== PII Redaction Span Processor Tests ==================


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_pii_redaction_span_processor_on_start():
    processor = PiiRedactionSpanProcessor()
    span = Mock()

    processor.on_start(span, None)


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_pii_redaction_span_processor_on_end_redacts():
    processor = PiiRedactionSpanProcessor()
    span = Mock()
    span.attributes = {"email": "test@example.com", "user_id": "123"}

    processor.on_end(span)

    calls = span.set_attribute.call_args_list
    assert any("[REDACTED_EMAIL]" in str(call) for call in calls)


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_pii_redaction_span_processor_shutdown():
    processor = PiiRedactionSpanProcessor()
    processor.shutdown()


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_pii_redaction_span_processor_force_flush():
    processor = PiiRedactionSpanProcessor()
    result = processor.force_flush(timeout_millis=1000)
    assert result is True


# ================== Safe OTLP Exporters Tests ==================


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_safe_otlp_span_exporter_allowlist_validation():
    breaker = CircuitBreaker()
    allowlist = ("localhost:4317",)

    with pytest.raises(ValueError, match="not in allow-list"):
        SafeOTLPSpanExporter(
            allowlist=allowlist,
            breaker=breaker,
            timeout=5.0,
            endpoint="malicious.com:4317",
            insecure=True,
        )


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_safe_otlp_span_exporter_circuit_breaker_open():
    breaker = CircuitBreaker(fail_threshold=1, reset_after_sec=10)
    breaker.on_failure()

    exporter = SafeOTLPSpanExporter(
        allowlist=("localhost:4317",),
        breaker=breaker,
        timeout=5.0,
        endpoint="localhost:4317",
        insecure=True,
    )

    assert breaker.is_open()
    exporter.export([])


# ================== Metrics Manager Tests ==================


def test_metrics_manager_initialization():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(service_name="test-svc", config=config)

    assert mgr.service_name == "test-svc"
    assert mgr.meter is not None


def test_metrics_manager_counter_creation():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(config=config)

    counter = mgr.counter("test_counter", "Test counter")
    assert counter is not None

    counter2 = mgr.counter("test_counter", "Test counter")
    assert counter is counter2


def test_metrics_manager_histogram_creation():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(config=config)

    hist = mgr.histogram("test_histogram", "Test histogram")
    assert hist is not None


def test_metrics_manager_record_counter_no_buffer():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(config=config)

    counter = mgr.counter("test_counter")
    mgr.record_counter(counter, 5.0, {"label": "value"})


def test_metrics_manager_record_histogram_no_buffer():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(config=config)

    hist = mgr.histogram("test_histogram")
    mgr.record_histogram(hist, 100.0, {"label": "value"})


def test_metrics_manager_buffered_emit():
    config = MetricConfig(
        prometheus_port=0, otlp_endpoint=None, buffered_emit=True, queue_max_events=100
    )
    mgr = MetricsManager(config=config)

    counter = mgr.counter("test_counter")
    mgr.record_counter(counter, 1.0)

    time.sleep(0.05)
    mgr.shutdown()


def test_metrics_manager_shutdown():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(config=config)
    mgr.shutdown()


# ================== Tracing Manager Tests ==================


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_tracing_manager_initialization():
    config = TracingConfig(otlp_endpoint=None, sample_rate=0.5)
    mgr = TracingManager(service_name="test-svc", config=config)

    assert mgr.service_name == "test-svc"
    assert mgr.tracer is not None


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_tracing_manager_set_sample_rate():
    config = TracingConfig(otlp_endpoint=None, sample_rate=0.5)
    mgr = TracingManager(config=config)

    mgr.set_sample_rate(0.8)
    assert mgr._adaptive_sampler._rate == 0.8


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_tracing_manager_start_span():
    config = TracingConfig(otlp_endpoint=None)
    mgr = TracingManager(config=config)

    obs.set_tenant("test-tenant")
    obs.set_strategy("test-strategy")

    span = mgr.start_span("test-span", attributes={"key": "value"})
    assert span is not None


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_tracing_manager_start_span_with_links():
    config = TracingConfig(otlp_endpoint=None)
    mgr = TracingManager(config=config)

    span1 = mgr.start_span("span1")
    span2 = mgr.start_span_with_links("span2", [span1])
    assert span2 is not None


@pytest.mark.skipif(not _HAS_OTEL, reason="OpenTelemetry not available")
def test_tracing_manager_inject_extract_context():
    config = TracingConfig(otlp_endpoint=None)
    mgr = TracingManager(config=config)

    carrier = {}
    mgr.inject_context(carrier)

    mgr.extract_context(carrier)


# ================== No-Op Managers Tests ==================


def test_noop_metrics_manager():
    mgr = NoOpMetricsManager()

    counter = mgr.counter("test")
    counter.add(1.0)

    hist = mgr.histogram("test")
    hist.record(1.0)

    mgr.record_counter(counter, 1.0)
    mgr.record_histogram(hist, 1.0)
    mgr.shutdown()


def test_noop_tracing_manager():
    mgr = NoOpTracingManager()

    mgr.set_sample_rate(0.5)
    mgr.start_span("test")
    mgr.start_span_with_links("test", [])

    carrier = {"key": "value"}
    result = mgr.inject_context(carrier)
    assert result == carrier

    mgr.extract_context(carrier)


# ================== Init Observability Pairwise Tests ==================


# Use @matrix for true multi-parameter pairwise testing
@matrix(enable_metrics=[True, False], enable_tracing=[True, False], enable_logging=[True, False])
def test_init_observability_pairwise(enable_metrics, enable_tracing, enable_logging):
    original_metrics = obs._metrics
    original_tracing = obs._tracing
    original_logging = obs._logging

    try:
        obs._metrics = None
        obs._tracing = None
        obs._logging = None
        obs._logger = None

        init_observability(
            service_name="test-svc",
            enable_metrics=enable_metrics,
            enable_tracing=enable_tracing,
            enable_logging=enable_logging,
        )

        if enable_metrics:
            assert get_metrics() is not None

        if enable_tracing and _HAS_OTEL:
            assert get_tracing() is not None

        if enable_logging:
            assert get_logging() is not None
    finally:
        obs._metrics = original_metrics
        obs._tracing = original_tracing
        obs._logging = original_logging


def test_init_observability_with_configs():
    obs._metrics = None
    obs._tracing = None
    obs._logging = None

    metric_cfg = MetricConfig(prometheus_port=0, otlp_endpoint=None)
    trace_cfg = TracingConfig(otlp_endpoint=None, sample_rate=0.1)

    init_observability(
        service_name="test",
        metrics_config=metric_cfg,
        tracing_config=trace_cfg,
        enable_metrics=True,
        enable_tracing=True,
        enable_logging=True,
    )

    assert get_metrics() is not None
    assert get_logging() is not None


# ================== Instrument Decorator Tests ==================


def test_instrument_sync_function(monkeypatch):
    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):

        @instrument(name="test_func")
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10


def test_instrument_sync_function_with_metrics():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)

    with patched_attr(obs, "_metrics", MetricsManager(config=config)):

        @instrument(name="test_func", labels={"env": "test"})
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10


def test_instrument_sync_function_error():
    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):

        @instrument(name="test_func", record_exceptions=True)
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()


@pytest.mark.asyncio
async def test_instrument_async_function():
    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):

        @instrument(name="async_test")
        async def async_func(x):
            await asyncio.sleep(0.001)
            return x * 3

        result = await async_func(4)
        assert result == 12


@pytest.mark.asyncio
async def test_instrument_async_function_error():
    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):

        @instrument(name="async_test", record_exceptions=True)
        async def async_func():
            await asyncio.sleep(0.001)
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await async_func()


# ================== FastAPI Middleware Tests ==================


@pytest.mark.asyncio
async def test_fastapi_middleware_http_request():
    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        middleware = FastAPIMiddleware(app, service_name="test-api")

        scope = {
            "type": "http",
            "path": "/test",
            "method": "GET",
            "scheme": "http",
            "server": ("localhost", 8000),
        }

        await middleware(scope, None, None)
        assert app_called


@pytest.mark.asyncio
async def test_fastapi_middleware_with_port_allocation():
    # Demonstrate proper use of allocate_port()
    port = allocate_port()

    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):

        async def app(scope, receive, send):
            pass

        middleware = FastAPIMiddleware(app)

        scope = {
            "type": "http",
            "path": "/test",
            "method": "GET",
            "scheme": "http",
            "server": ("localhost", port),
        }

        await middleware(scope, None, None)


@pytest.mark.asyncio
async def test_fastapi_middleware_non_http():
    app_called = False

    async def app(scope, receive, send):
        nonlocal app_called
        app_called = True

    middleware = FastAPIMiddleware(app)

    scope = {"type": "websocket"}

    await middleware(scope, None, None)
    assert app_called


@pytest.mark.asyncio
async def test_fastapi_middleware_error_handling():
    with patched_attr(obs, "_metrics", None), patched_attr(obs, "_tracing", None):

        async def app(scope, receive, send):
            raise ValueError("test error")

        middleware = FastAPIMiddleware(app)

        scope = {
            "type": "http",
            "path": "/error",
            "method": "POST",
            "scheme": "http",
            "server": ("localhost", 8000),
        }

        with pytest.raises(ValueError, match="test error"):
            await middleware(scope, None, None)


# ================== Kafka Instrumentor Tests ==================


def test_kafka_instrumentor_inject_context():
    instrumentor = KafkaInstrumentor()

    headers = {}
    result = instrumentor.inject_context(headers)

    assert isinstance(result, dict)


def test_kafka_instrumentor_extract_context():
    instrumentor = KafkaInstrumentor()

    headers = {b"traceparent": b"00-trace-span-01"}
    instrumentor.extract_context(headers)


def test_kafka_instrumentor_producer():
    with patched_attr(obs, "_tracing", None):
        instrumentor = KafkaInstrumentor()

        producer = Mock()
        original_send = Mock(return_value=None)
        producer.send = original_send  # Save reference to original mock

        instrumented = instrumentor.instrument_producer(producer)
        instrumented.send("test-topic", value=b"data", headers={})

        # Assert the original mock (captured in closure) was called
        original_send.assert_called_once()


def test_kafka_instrumentor_consumer():
    with patched_attr(obs, "_tracing", None):
        instrumentor = KafkaInstrumentor()

        consumer = Mock()
        original_poll = Mock(return_value={})
        consumer.poll = original_poll  # Save reference to original mock

        instrumented = instrumentor.instrument_consumer(consumer)
        result = instrumented.poll()

        assert result == {}
        original_poll.assert_called_once()


# ================== Property-Based Tests ==================


@given(
    fail_threshold=st.integers(min_value=1, max_value=100),
    reset_after_sec=st.integers(min_value=1, max_value=60),
)
@settings(deadline=None, max_examples=50)
@seed(12345)
def test_circuit_breaker_property_invariants(fail_threshold, reset_after_sec):
    cb = CircuitBreaker(fail_threshold=fail_threshold, reset_after_sec=reset_after_sec)

    for _ in range(fail_threshold - 1):
        cb.on_failure()

    assert not cb.is_open()

    cb.on_failure()
    assert cb.is_open()

    cb.on_success()
    assert not cb.is_open()


@given(
    max_keys=st.integers(min_value=1, max_value=100),
    num_values=st.integers(min_value=1, max_value=200),
)
@settings(deadline=None, max_examples=50)
@seed(12345)
def test_cardinality_limiter_property_bounded(max_keys, num_values):
    limiter = CardinalityLimiter(max_keys_per_label=max_keys)

    unique_results = set()
    for i in range(num_values):
        result = limiter.sanitize("test", f"value_{i}")
        unique_results.add(result)

    stats = limiter.stats()
    assert stats["labels"]["test"] <= max_keys + 1


@given(
    values=st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=50),
    alpha=st.floats(min_value=0.01, max_value=0.99),
)
@settings(deadline=None, max_examples=50)
@seed(12345)
def test_adaptive_threshold_property_monotonic_learning(values, alpha):
    at = AdaptiveThreshold(alpha=alpha, sensitivity=2.0)

    thresholds = []
    for v in values:
        at.update(v)
        thresholds.append(at.threshold())

    assert all(t < float("inf") for t in thresholds[-5:])


# ================== Concurrency Invariant Tests ==================


def test_circuit_breaker_concurrent_invariant_no_race():
    cb = CircuitBreaker(fail_threshold=50, reset_after_sec=1)
    results = []

    def worker():
        for _ in range(10):
            cb.on_failure()
            results.append(cb.fail_count)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker) for _ in range(8)]
        for f in futures:
            f.result()

    assert cb.fail_count == 80
    assert max(results) == 80


def test_bounded_queue_concurrent_invariant_no_loss():
    queue = BoundedEventQueue(maxsize=10000)

    def worker(worker_id):
        for i in range(100):
            ev = obs._Event("counter", Mock(), float(i), {"worker": str(worker_id)})
            queue.put_nowait(ev)

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(worker, i) for i in range(10)]
        for f in futures:
            f.result()

    total_size = queue.size() + queue.dropped()
    assert total_size == 1000


def test_cardinality_limiter_concurrent_invariant_bounded():
    limiter = CardinalityLimiter(max_keys_per_label=50)

    def worker(worker_id):
        for i in range(20):
            limiter.sanitize("env", f"worker_{worker_id}_val_{i}")

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(worker, i) for i in range(10)]
        for f in futures:
            f.result()

    stats = limiter.stats()
    assert stats["labels"]["env"] <= 50


# ================== Error Condition Tests ==================


def test_safe_otlp_span_exporter_raises_exporter_egress_error():
    breaker = CircuitBreaker()

    with pytest.raises(ValueError, match="not in allow-list"):
        SafeOTLPSpanExporter(
            allowlist=("localhost:4317",),
            breaker=breaker,
            timeout=1.0,
            endpoint="invalid:9999",
            insecure=True,
        )


def test_metrics_manager_handles_invalid_labels():
    config = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    mgr = MetricsManager(config=config)

    counter = mgr.counter("test")
    mgr.record_counter(counter, 1.0, labels=None)


def test_pii_redactor_handles_malformed_dict():
    redactor = PIIRedactor()

    data = {"key": None, "nested": {"inner": 123}}
    result = redactor.redact_dict(data)

    assert "key" in result
    assert "nested" in result


# ================== Integration Tests ==================


def test_end_to_end_observability_stack():
    obs._metrics = None
    obs._tracing = None
    obs._logging = None
    obs._logger = None

    init_observability(
        service_name="integration-test",
        enable_metrics=True,
        enable_tracing=True,
        enable_logging=True,
    )

    assert get_metrics() is not None
    assert get_logging() is not None

    obs.set_tenant("test-tenant")
    obs.set_strategy("test-strategy")

    assert obs.get_tenant() == "test-tenant"
    assert obs.get_strategy() == "test-strategy"


def test_metrics_and_tracing_integration():
    metric_cfg = MetricConfig(prometheus_port=0, otlp_endpoint=None, buffered_emit=False)
    trace_cfg = TracingConfig(otlp_endpoint=None, sample_rate=1.0)

    with patched_attr(
        obs, "_metrics", MetricsManager(config=metric_cfg) if _HAS_OTEL else NoOpMetricsManager()
    ), patched_attr(
        obs, "_tracing", TracingManager(config=trace_cfg) if _HAS_OTEL else NoOpTracingManager()
    ):

        @instrument(name="integrated_func", measure_latency=True)
        def test_func(x):
            return x * 2

        result = test_func(10)
        assert result == 20

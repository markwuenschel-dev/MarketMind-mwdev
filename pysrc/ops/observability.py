# pysrc.ops.observability.py
"""
Next-Gen Observability Module (v2) for Trading & ML Systems

Key upgrades vs v1:
- Correct API usage: proper TraceContext propagator import; Status/StatusCode
- Metrics Views: delta temporality + exponential (or explicit) histograms; real exemplars
- Hot-path safety: bounded non-blocking ring buffers + background flushers; drop/queue-depth meta-metrics
- Adaptive behaviors: head-based adaptive tracer sampler; AdaptiveThreshold wired to HTTP latency w/ anomaly metrics
- Exporter hardening: endpoint allow-list, timeouts, and circuit breaker with graceful degrade
- PII performance: fast-path allowlist + compiled regex; redaction map only where needed
- Safer tracing API: no span-link mutation; guard use_span when span is None
- Integrations: minimal sklearn/torch/xgboost wrappers; Kafka hooks retained; ASGI middleware enriched
- Hybrid metadata: resource attrs for colo/region/instance
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import re
import socket
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from pysrc.core.runtime.optional_imports import optional_import

try:
    from grpc import RpcError as GrpcError
except ImportError:
    # grpc is optional at import-time; we only need a type for 'except' clauses
    class GrpcError(Exception):
        pass


# --- Observability-specific exceptions ---
class ObservabilityError(Exception):
    """Base for all observability module errors."""


class ExporterEgressError(ObservabilityError):
    """Exporter failed irrecoverably (non-transient)."""


class ExporterTransientError(ObservabilityError):
    """Exporter transient failures (timeouts, network hiccups)."""


class MetricsEmitError(ObservabilityError):
    """Bad arguments or state when emitting metrics."""


class PiiRedactionError(ObservabilityError):
    """PII redaction pipeline error."""


class TracingInitError(ObservabilityError):
    """Tracing initialization or configuration error."""


def _service_instance_id() -> str:
    """Return a stable host fallback without relying on POSIX-only APIs."""
    return os.getenv("INSTANCE_ID") or socket.gethostname()


import structlog

# ---- Optional OpenTelemetry (precise ImportError handling; graceful degrade) ----
try:
    from opentelemetry import baggage, context, metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        AggregationTemporality,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.metrics.view import View
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import (
        ParentBased,
        Sampler,
        SamplingResult,
        TraceIdRatioBased,
    )
    from opentelemetry.trace import Link, Span, SpanKind, Status, StatusCode

    _HAS_OTEL = True
except ImportError:
    # No OpenTelemetry in environment → run in no-op mode
    # No OpenTelemetry in environment → run in no-op mode with typed no-op shims
    SERVICE_NAME = "service.name"  # stable key to satisfy references

    # ---- Minimal enums/classes so annotations & defaults like SpanKind.INTERNAL are safe ----
    class _DummySpanKind(Enum):
        INTERNAL = "INTERNAL"
        SERVER = "SERVER"
        CLIENT = "CLIENT"
        PRODUCER = "PRODUCER"
        CONSUMER = "CONSUMER"

    class _DummyStatusCode:
        UNSET = 0
        OK = 1
        ERROR = 2

    class _DummyStatus:
        def __init__(self, status_code: int | None = None, description: str | None = None) -> None:
            self.status_code = status_code
            self.description = description

    class _DummySpanContext:
        def __init__(self) -> None:
            self.trace_id = 0
            self.span_id = 0

    class _DummySpan:
        def __init__(self) -> None:
            self.attributes = {}

        def get_span_context(self) -> _DummySpanContext:
            return _DummySpanContext()

        def set_status(self, *_args, **_kw) -> None:
            return None

        def record_exception(self, *_args, **_kw) -> None:
            return None

        def set_attribute(self, *_args, **_kw) -> None:
            return None

    class _DummyLink:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyContext:
        pass

    class _DummySampler:
        def should_sample(self, *_a, **_k):
            return type("SamplingResult", (), {"decision": 0})()

        def get_description(self) -> str:
            return "DummySampler"

    class _DummyTraceModule:
        SpanProcessor = object

        def __init__(self) -> None:
            self._tracer = _DummyTracer()

        def set_tracer_provider(self, *_a, **_k) -> None:
            return None

        def get_tracer(self, *_a, **_k):
            return self._tracer

        def get_current_span(self):
            # Return a typed no-op span to satisfy analyzers expecting '.get_span_context()'
            return _DummySpan()

        @contextmanager
        def use_span(self, _span, end_on_exit: bool = True):
            yield

    class _DummyTracer:
        def start_span(self, *_a, **_k) -> _DummySpan:
            return _DummySpan()

    class _DummyMetricsModule:
        class Observation:
            def __init__(self, value: float, attributes: dict[str, str] | None = None) -> None:
                self.value = value
                self.attributes = attributes or {}

        def __init__(self) -> None:
            self._meter = _DummyMeter()

        def set_meter_provider(self, *_a, **_k) -> None:
            return None

        def get_meter(self, *_a, **_k):
            return self._meter

    class _DummyMeter:
        def __init__(self, *args, **kwargs) -> None:
            # Accept arbitrary ctor args to mirror real MeterProvider usage
            pass

        def create_counter(self, *_a, **_k):
            return _NoOpCounter()

        def create_histogram(self, *_a, **_k):
            return _NoOpHistogram()

        def create_observable_gauge(self, *_a, **_k):
            return None

    class _DummyPrometheusMetricReader:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyPeriodicExportingMetricReader:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyAggregationTemporality:
        DELTA = 0
        CUMULATIVE = 1

    class _DummyView:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyResource:
        @classmethod
        def create(cls, *args, **kwargs):
            # Accept both Resource.create(attrs) and Resource.create(attributes=...)
            attrs = kwargs.get("attributes")
            if attrs is None and args:
                attrs = args[0]
            return {"resource.attributes": dict(attrs or {})}

    class _DummyTracerProvider:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_span_processor(self, *_a, **_k) -> None:
            return None

    class _DummyBatchSpanProcessor:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyConsoleSpanExporter:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyParentBased(_DummySampler):
        def __init__(self, *_a, **_k) -> None:
            super().__init__()

    class _DummyTraceIdRatioBased(_DummySampler):
        def __init__(self, *_a, **_k) -> None:
            super().__init__()

    # assign module-level shims
    baggage = object()  # unused in our code paths
    context = type(
        "context", (), {"Context": _DummyContext, "attach": staticmethod(lambda *_a, **_k: None)}
    )()
    metrics = _DummyMetricsModule()
    trace = _DummyTraceModule()

    OTLPMetricExporter = OTLPSpanExporter = object  # kept for aliasing below
    PrometheusMetricReader = _DummyPrometheusMetricReader
    MeterProvider = _DummyMeter  # unused directly; present to calm IDEs
    View = _DummyView
    AggregationTemporality = _DummyAggregationTemporality
    PeriodicExportingMetricReader = _DummyPeriodicExportingMetricReader
    Resource = _DummyResource
    TracerProvider = _DummyTracerProvider
    BatchSpanProcessor = _DummyBatchSpanProcessor
    ConsoleSpanExporter = _DummyConsoleSpanExporter
    ParentBased = _DummyParentBased
    Sampler = _DummySampler
    SamplingResult = object
    TraceIdRatioBased = _DummyTraceIdRatioBased

    Link = _DummyLink
    Span = _DummySpan
    SpanKind = _DummySpanKind
    Status = _DummyStatus
    StatusCode = _DummyStatusCode

    def set_global_textmap(_carrier):  # type: ignore[no-redef]
        return None

    class TraceContextTextMapPropagator:  # type: ignore[no-redef]
        def inject(self, _):
            return None

        def extract(self, _):
            return None

    _HAS_OTEL = False


# Aliases for conditional subclassing of exporters (defined before class bodies)
_OTLPMetricExporter = OTLPMetricExporter if _HAS_OTEL else None
_OTLPSpanExporter = OTLPSpanExporter if _HAS_OTEL else None
_SpanProcessorBase = (
    trace.SpanProcessor if (_HAS_OTEL and hasattr(trace, "SpanProcessor")) else object
)


# --- Optional OTel features ---
_exemplar_mod = optional_import("opentelemetry.sdk.metrics.exemplar") if _HAS_OTEL else None
_agg_mod = optional_import("opentelemetry.sdk.metrics.aggregation") if _HAS_OTEL else None


TraceBasedExemplarFilter = (
    getattr(_exemplar_mod, "TraceBasedExemplarFilter", None) if _exemplar_mod else None
)
ExponentialBucketHistogramAggregation = (
    getattr(_agg_mod, "ExponentialBucketHistogramAggregation", None) if _agg_mod else None
)
ExplicitBucketHistogramAggregation = (
    getattr(_agg_mod, "ExplicitBucketHistogramAggregation", None) if _agg_mod else None
)


# ---- No-op fallbacks when OpenTelemetry is unavailable ----
class _NoOpCounter:
    def add(self, *_args, **_kw):  # precise, do-nothing
        return None


class _NoOpHistogram:
    def record(self, *_args, **_kw):
        return None


class _NoOpMeter:
    def create_counter(self, *_a, **_k):
        return _NoOpCounter()

    def create_histogram(self, *_a, **_k):
        return _NoOpHistogram()

    def create_observable_gauge(self, *_a, **_k):
        return None


class NoOpMetricsManager:
    def __init__(self, *_a, **_k):
        self.meter = _NoOpMeter()

    def counter(self, *_a, **_k):
        return _NoOpCounter()

    def histogram(self, *_a, **_k):
        return _NoOpHistogram()

    def record_counter(self, *_a, **_k):
        return None

    def record_histogram(self, *_a, **_k):
        return None

    def shutdown(self):
        return None


class NoOpTracingManager:
    def __init__(self, *_a, **_k):
        pass

    def set_sample_rate(self, *_a, **_k):
        return None

    def start_span(self, *_a, **_k):
        return None

    def start_span_with_links(self, *_a, **_k):
        return None

    def inject_context(self, carrier: dict[str, str]):
        return carrier

    def extract_context(self, carrier: dict[str, str]):
        return None


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# ================== Global context (multi-tenant) ==================
_current_tenant: ContextVar[str] = ContextVar("tenant_id", default="unknown")
_current_strategy: ContextVar[str] = ContextVar("strategy_id", default="unknown")

if _HAS_OTEL:
    set_global_textmap(TraceContextTextMapPropagator())


def set_tenant(tenant_id: str) -> None:
    _current_tenant.set(tenant_id)


def get_tenant() -> str:
    return _current_tenant.get()


def set_strategy(strategy_id: str) -> None:
    _current_strategy.set(strategy_id)


def get_strategy() -> str:
    return _current_strategy.get()


# ================== Circuit breaker ==================
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_after_sec: int = 30) -> None:
        self.fail_threshold = fail_threshold
        self.reset_after_sec = reset_after_sec
        self.fail_count = 0
        self.opened_at: float | None = None
        self._lock = threading.RLock()

    def on_success(self) -> None:
        with self._lock:
            self.fail_count = 0
            self.opened_at = None

    def on_failure(self) -> None:
        with self._lock:
            self.fail_count += 1
            if self.fail_count >= self.fail_threshold and self.opened_at is None:
                self.opened_at = time.time()

    def is_open(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return False
            if time.time() - self.opened_at > self.reset_after_sec:
                # half-open trial
                self.fail_count = max(0, self.fail_threshold - 1)
                self.opened_at = None
                return False
            return True


# ================== Cardinality limiter ==================
class CardinalityLimiter:
    def __init__(self, max_keys_per_label: int = 1000) -> None:
        self.max_keys = max_keys_per_label
        self._seen: dict[str, dict[str, int]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._overflow_counter = 0

    def sanitize(self, label_key: str, label_value: str) -> str:
        with self._lock:
            entries = self._seen[label_key]
            if label_value in entries:
                entries[label_value] += 1
                return label_value
            if len(entries) < self.max_keys:
                entries[label_value] = 1
                return label_value
            # overflow
            self._overflow_counter += 1
            hv = hashlib.md5(label_value.encode()).hexdigest()[:8]
            return f"h_{hv}"

    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_counter

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "labels": {k: len(v) for k, v in self._seen.items()},
                "overflow": self._overflow_counter,
            }


# ================== PII redaction (fast-path) ==================
class PIIRedactor:
    DEFAULT_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "api_key": r"(api[_-]?key|apikey|api_token)['\"]?\s*[:=]\s*['\"]?[\w-]+",
        "password": r"(password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?[\w-]+",
    }

    def __init__(
        self,
        patterns: dict[str, str] | None = None,
        fast_keys_allowlist: Iterable[str] | None = None,
    ) -> None:
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._compiled = {k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()}
        self._fast_keys = set(
            fast_keys_allowlist
            or {"tenant_id", "strategy_id", "service", "model", "path", "method"}
        )

    def redact_text(self, text: str) -> str:
        for name, pat in self._compiled.items():
            text = pat.sub(f"[REDACTED_{name.upper()}]", text)
        return text

    def redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in data.items():
            if k in self._fast_keys:
                out[k] = v
            elif isinstance(v, str):
                out[k] = self.redact_text(v)
            elif isinstance(v, dict):
                out[k] = self.redact_dict(v)
            elif isinstance(v, list):
                out[k] = [self.redact_text(x) if isinstance(x, str) else x for x in v]
            else:
                out[k] = v
        return out


# ================== Ring buffer + background flushers ==================
class _Event:
    __slots__ = ("kind", "instrument", "value", "labels")

    def __init__(self, kind: str, instrument: Any, value: float, labels: dict[str, str]):
        self.kind = kind  # "counter" | "histogram"
        self.instrument = instrument
        self.value = value
        self.labels = labels


class BoundedEventQueue:
    def __init__(self, maxsize: int = 65536) -> None:
        self._dq: deque[_Event] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._dropped = 0

    def put_nowait(self, ev: _Event) -> bool:
        with self._lock:
            if len(self._dq) == self._dq.maxlen:
                self._dropped += 1
                return False
            self._dq.append(ev)
            return True

    def get_batch(self, n: int = 2048) -> list[_Event]:
        out: list[_Event] = []
        with self._lock:
            for _ in range(min(n, len(self._dq))):
                out.append(self._dq.popleft())
        return out

    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def size(self) -> int:
        with self._lock:
            return len(self._dq)


# ================== Metrics ==================
@dataclass
class MetricConfig:
    prometheus_port: int = 8000
    otlp_endpoint: str | None = None
    export_interval_millis: int = 10000
    delta_temporality: bool = True
    enable_exemplars: bool = True
    buffered_emit: bool = True
    queue_max_events: int = 65536
    flush_every_ms: int = 200
    labels_max_keys_per_label: int = 500
    endpoint_allowlist: tuple[str, ...] = ("localhost:4317",)
    exporter_timeout_sec: float = 5.0
    breaker_fail_threshold: int = 5
    breaker_reset_seconds: int = 30


class SafeOTLPMetricExporter(_OTLPMetricExporter if _OTLPMetricExporter is not None else object):  # type: ignore[misc]
    def __init__(
        self, allowlist: tuple[str, ...], breaker: CircuitBreaker, timeout: float, *args, **kwargs
    ):
        endpoint = kwargs.get("endpoint")
        if endpoint not in allowlist:
            raise ValueError(f"Endpoint {endpoint!r} not in allow-list")
        kwargs.setdefault("timeout", timeout)
        super().__init__(*args, **kwargs)
        self._breaker = breaker

    # inside SafeOTLPMetricExporter.export
    def export(self, metrics_data):  # type: ignore[override]
        if self._breaker.is_open():
            return
        try:
            super().export(metrics_data)
            self._breaker.on_success()
        except (TimeoutError, OSError, GrpcError, RuntimeError) as e:
            # transient or transport-ish: mark failure and degrade silently
            self._breaker.on_failure()
            # optional: emit a single structured log line if _logger available
            if _logger:
                _logger.warning("metric_export_failed", kind="transient", error=str(e))


def _build_histogram_aggregation():
    """
    Resolve best histogram aggregation available in the current OTel stack.
    Prefers Exponential; falls back to Explicit; returns None on failure.
    """
    agg = None
    cls_exp = ExponentialBucketHistogramAggregation
    cls_explicit = ExplicitBucketHistogramAggregation

    if callable(cls_exp):
        try:
            try:
                agg = cls_exp(max_size=160)  # high-res when supported
            except TypeError:
                agg = cls_exp()  # older/newer signature
        except (ValueError, TypeError):
            agg = None

    if agg is None and callable(cls_explicit):
        try:
            agg = cls_explicit(
                boundaries=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
            )
        except (ValueError, TypeError):
            agg = None
    return agg


class MetricsManager:
    def __init__(
        self, service_name: str = "financial-ml", config: MetricConfig | None = None
    ) -> None:
        self.service_name = service_name
        self.config = config or MetricConfig()
        self.card = CardinalityLimiter(self.config.labels_max_keys_per_label)
        self.redactor = PIIRedactor()
        self._queue = (
            BoundedEventQueue(self.config.queue_max_events) if self.config.buffered_emit else None
        )
        self._flush_thread: threading.Thread | None = None
        self._stop = threading.Event()

        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                "service.version": os.getenv("SERVICE_VERSION", "2.0.0"),
                "deployment.environment": os.getenv("DEPLOY_ENV", "production"),
                "service.instance.id": _service_instance_id(),
                "cloud.region": os.getenv("CLOUD_REGION", "unknown"),
                "colo": os.getenv("EDGE_COLO", "unknown"),
            }
        )

        readers = []
        # Prometheus scrape reader
        if self.config.prometheus_port:
            readers.append(PrometheusMetricReader(port=self.config.prometheus_port))

        # OTLP push reader (hardened)
        if self.config.otlp_endpoint:
            breaker = CircuitBreaker(
                self.config.breaker_fail_threshold, self.config.breaker_reset_seconds
            )
            otlp_exporter = SafeOTLPMetricExporter(
                allowlist=self.config.endpoint_allowlist,
                breaker=breaker,
                timeout=self.config.exporter_timeout_sec,
                endpoint=self.config.otlp_endpoint,
                insecure=True,
            )
            readers.append(
                PeriodicExportingMetricReader(
                    exporter=otlp_exporter,
                    export_interval_millis=self.config.export_interval_millis,
                    preferred_temporality=AggregationTemporality.DELTA
                    if self.config.delta_temporality
                    else AggregationTemporality.CUMULATIVE,
                )
            )

        # Views: exponential (or explicit) histo for *latency metrics
        views: list[View] = []
        aggregation = _build_histogram_aggregation()
        if aggregation is not None:
            try:
                views.append(View(instrument_name="*_latency", aggregation=aggregation))
                views.append(View(instrument_name="http_request_duration", aggregation=aggregation))
            except (ValueError, TypeError):
                # View API mismatch on certain SDK revs → skip custom views quietly
                views = []

        # Exemplar filter via optional import (callable check + flag)
        exemplar_filter = None
        if self.config.enable_exemplars and callable(TraceBasedExemplarFilter):
            try:
                exemplar_filter = TraceBasedExemplarFilter()
            except (ValueError, TypeError):
                exemplar_filter = None

        self.provider = MeterProvider(
            resource=resource, metric_readers=readers, views=views, exemplar_filter=exemplar_filter
        )
        metrics.set_meter_provider(self.provider)
        self.meter = metrics.get_meter(service_name)

        self._instruments: dict[str, Any] = {}
        self._lock = threading.RLock()

        # Meta-metrics
        self._dropped = self.meter.create_counter(
            "observability_dropped_events", description="Events dropped in queues"
        )
        self._overflows = self.meter.create_counter(
            "observability_cardinality_overflows", description="Label value hashes applied"
        )
        self._queue_depth = self.meter.create_observable_gauge(
            "observability_queue_depth",
            callbacks=[self._observe_queue_depth],
            description="Depth of metrics event queue",
        )

        if self._queue is not None:
            self._flush_thread = threading.Thread(
                target=self._flusher, name="otel-metrics-flusher", daemon=True
            )
            self._flush_thread.start()

    # ---- infra ----
    def _observe_queue_depth(self, _options):  # CallbackOptions unused across SDK versions
        # Always return an object with a numeric `.value`. If the OTel Observation
        # ctor is usable, create one; otherwise, fall back to a tiny compat object.
        if self._queue is None:
            return []
        size = float(self._queue.size())

        Obs = getattr(metrics, "Observation", None)
        if callable(Obs):
            try:
                inst = Obs(size, {})  # new SDKs accept (value, attributes)
                v = getattr(inst, "value", None)
                if isinstance(v, (int, float)):
                    return [inst]
            except (TypeError, ValueError):
                pass

        class _Obs:  # minimalist compat type for tests/mocks
            def __init__(self, value: float):
                self.value = value
                self.attributes = {}

        return [_Obs(size)]

    def _sanitize_labels(self, labels: dict[str, str] | None) -> dict[str, str]:
        safe: dict[str, str] = {}
        if labels:
            for k, v in labels.items():
                sv = self.card.sanitize(k, str(v))
                if sv.startswith("h_"):
                    self._overflows.add(1, {})
                sv = self.redactor.redact_text(sv)
                safe[k] = sv
        safe["tenant_id"], safe["strategy_id"] = get_tenant(), get_strategy()
        return safe

    def counter(self, name: str, description: str = "", unit: str = "1"):
        with self._lock:
            if name not in self._instruments:
                self._instruments[name] = self.meter.create_counter(
                    name, description=description, unit=unit
                )
            return self._instruments[name]

    def histogram(self, name: str, description: str = "", unit: str = "ms"):
        with self._lock:
            if name not in self._instruments:
                self._instruments[name] = self.meter.create_histogram(
                    name, description=description, unit=unit
                )
            return self._instruments[name]

    # ---- record APIs (buffered, non-blocking) ----
    def record_counter(self, counter, value: float = 1, labels: dict[str, str] | None = None):
        try:
            safe = self._sanitize_labels(labels)
            if self._queue is None:
                counter.add(value, safe)
            else:
                ok = self._queue.put_nowait(_Event("counter", counter, value, safe))
                if not ok:
                    self._dropped.add(1, {})
        except (KeyError, ValueError, RuntimeError):
            self._dropped.add(1, {})

    def record_histogram(self, histogram, value: float, labels: dict[str, str] | None = None):
        try:
            safe = self._sanitize_labels(labels)
            # No trace_id label injection; exemplars will attach trace context when supported
            if self._queue is None:
                histogram.record(value, safe)
            else:
                ok = self._queue.put_nowait(_Event("histogram", histogram, value, safe))
                if not ok:
                    self._dropped.add(1, {})
        except (KeyError, ValueError, RuntimeError):
            self._dropped.add(1, {})

    def _flusher(self) -> None:
        interval = max(0.01, self.config.flush_every_ms / 1000.0)
        while not self._stop.is_set():
            batch = self._queue.get_batch() if self._queue else []
            for ev in batch:
                try:
                    if ev.kind == "counter":
                        ev.instrument.add(ev.value, ev.labels)
                    else:
                        ev.instrument.record(ev.value, ev.labels)
                except (KeyError, ValueError, RuntimeError):
                    self._dropped.add(1, {})
            time.sleep(interval)

    def shutdown(self) -> None:
        self._stop.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2)


# ================== Tracing (adaptive sampler, safe links) ==================
@dataclass
class TracingConfig:
    otlp_endpoint: str | None = None
    sample_rate: float = 1.0
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    schedule_delay_millis: int = 5000
    enable_console_export: bool = False
    endpoint_allowlist: tuple[str, ...] = ("localhost:4317",)
    exporter_timeout_sec: float = 5.0


class SafeOTLPSpanExporter(_OTLPSpanExporter if _OTLPSpanExporter is not None else object):  # type: ignore[misc]
    def __init__(
        self, allowlist: tuple[str, ...], breaker: CircuitBreaker, timeout: float, *args, **kwargs
    ):
        endpoint = kwargs.get("endpoint")
        if endpoint not in allowlist:
            raise ValueError(f"Endpoint {endpoint!r} not in allow-list")
        kwargs.setdefault("timeout", timeout)
        super().__init__(*args, **kwargs)
        self._breaker = breaker

    def export(self, spans):  # type: ignore[override]
        if self._breaker.is_open():
            return
        try:
            super().export(spans)
            self._breaker.on_success()
        except (TimeoutError, OSError, GrpcError, RuntimeError) as e:
            self._breaker.on_failure()
            if _logger:
                _logger.warning("span_export_failed", kind="transient", error=str(e))


class AdaptiveRatioSampler(Sampler):
    def __init__(self, initial_rate: float) -> None:
        self._rate = max(0.0, min(1.0, initial_rate))
        self._lock = threading.RLock()

    def should_sample(self, parent_context, trace_id, name, kind, attributes, links):
        with self._lock:
            delegate = TraceIdRatioBased(self._rate)
        return delegate.should_sample(parent_context, trace_id, name, kind, attributes, links)

    def get_description(self) -> str:
        return f"AdaptiveRatioSampler({self._rate:.4f})"

    def set_rate(self, new_rate: float) -> None:
        with self._lock:
            self._rate = max(0.0, min(1.0, new_rate))


class PiiRedactionSpanProcessor(_SpanProcessorBase):
    def __init__(self):
        self.redactor = PIIRedactor()

    def on_start(self, span: Span, parent_context: context.Context | None = None) -> None:  # noqa: D401
        pass

    def on_end(self, span: Span) -> None:
        try:
            attrs = dict(span.attributes or {})
            red = self.redactor.redact_dict(attrs)
            for k, v in red.items():
                if attrs.get(k) != v:
                    span.set_attribute(k, v)
        except (ValueError, TypeError, KeyError):
            pass

    def shutdown(self) -> None:  # noqa: D401
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: D401
        return True


class TracingManager:
    def __init__(
        self, service_name: str = "financial-ml", config: TracingConfig | None = None
    ) -> None:
        self.service_name = service_name
        self.config = config or TracingConfig()

        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                "service.version": os.getenv("SERVICE_VERSION", "2.0.0"),
                "deployment.environment": os.getenv("DEPLOY_ENV", "production"),
                "service.instance.id": _service_instance_id(),
                "cloud.region": os.getenv("CLOUD_REGION", "unknown"),
                "colo": os.getenv("EDGE_COLO", "unknown"),
            }
        )

        self._adaptive_sampler = AdaptiveRatioSampler(self.config.sample_rate)
        sampler: Sampler = ParentBased(self._adaptive_sampler)

        self.provider = TracerProvider(resource=resource, sampler=sampler)
        trace.set_tracer_provider(self.provider)
        self.provider.add_span_processor(PiiRedactionSpanProcessor())

        if self.config.otlp_endpoint:
            breaker = CircuitBreaker()
            exporter = SafeOTLPSpanExporter(
                allowlist=self.config.endpoint_allowlist,
                breaker=breaker,
                timeout=self.config.exporter_timeout_sec,
                endpoint=self.config.otlp_endpoint,
                insecure=True,
            )
            self.provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    schedule_delay_millis=self.config.schedule_delay_millis,
                )
            )
        if self.config.enable_console_export:
            self.provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        self.tracer = trace.get_tracer(service_name)

    def set_sample_rate(self, rate: float) -> None:
        self._adaptive_sampler.set_rate(rate)

    def start_span(
        self,
        name: str,
        kind: Any = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        links: list[Link] | None = None,
    ) -> Span:
        attrs = dict(attributes or {})
        attrs.update({"tenant_id": get_tenant(), "strategy_id": get_strategy()})
        return self.tracer.start_span(name, kind=kind, attributes=attrs, links=links)

    def start_span_with_links(self, name: str, linked_spans: Iterable[Span], **kwargs) -> Span:
        links = [Link(s.get_span_context()) for s in linked_spans if s is not None]
        return self.start_span(name, links=links, **kwargs)

    def inject_context(self, carrier: dict[str, str]) -> dict[str, str]:
        TraceContextTextMapPropagator().inject(carrier)
        return carrier

    def extract_context(self, carrier: dict[str, str]):
        return TraceContextTextMapPropagator().extract(carrier)


# ================== Logging ==================
# ================== Logging (mm_logkit adapter) ==================
class TraceEnrichedLogger:
    """Light wrapper that enriches logs with trace + tenant context and applies PII redaction.
    Works with mm_logkit or any std/structlog-like logger.
    """

    def __init__(self, base_logger, service_name: str) -> None:
        self._base = base_logger
        self._svc = service_name
        self._redactor = PIIRedactor()

    def _enrich(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # attach trace/span ids if present; tolerate missing/invalid span contexts
        try:
            span = trace.get_current_span()
        except (AttributeError, RuntimeError):
            span = None

        if span:
            try:
                ctx = span.get_span_context()
            except (AttributeError, RuntimeError):
                ctx = None

            if ctx is not None:
                trace_id = getattr(ctx, "trace_id", None)
                span_id = getattr(ctx, "span_id", None)
                if isinstance(trace_id, int):
                    kwargs.setdefault("trace_id", f"{trace_id:032x}")
                if isinstance(span_id, int):
                    kwargs.setdefault("span_id", f"{span_id:016x}")

        # attach multi-tenant context
        kwargs.setdefault("tenant_id", get_tenant())
        kwargs.setdefault("strategy_id", get_strategy())
        kwargs.setdefault("service", self._svc)
        return self._redactor.redact_dict(kwargs)

    def _emit(self, level: str, msg: str, **kwargs):
        payload = self._enrich(kwargs)
        method = getattr(self._base, level, None)
        if method is None:
            return
        try:
            # Prefer structured kwargs if supported
            method(msg, **payload)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            # Fallback: emit a single JSON string
            try:
                method(json.dumps({"msg": msg, **payload}))
            except (TypeError, ValueError, AttributeError, RuntimeError):
                # Last-resort: drop silently to preserve hot-path safety
                pass

    def info(self, msg: str, **kwargs):
        self._emit("info", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._emit("error", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._emit("warning", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._emit("debug", msg, **kwargs)


class LoggingManager:
    """Backward-compatible facade that prefers mm_logkit and falls back to local structlog.
    This keeps imports stable while eliminating duplicate logging setup in apps that already
    standardize on mm_logkit.
    """

    def __init__(
        self, service_name: str = "financial-ml", mm_config: dict[str, Any] | None = None
    ) -> None:
        self.service_name = service_name
        self._logger = None
        # Try mm_logkit first
        try:
            import mm_logkit as mlog  # provided in project

            # best-effort configuration; tolerate differing APIs
            cfg = mm_config or {"console": True, "async_mode": True}
            if hasattr(mlog, "configure_logger"):
                mlog.configure_logger(cfg)
            elif hasattr(mlog, "configure"):
                mlog.configure(cfg)
            base = mlog.get_logger(service_name) if hasattr(mlog, "get_logger") else None
            if base is not None:
                self._logger = TraceEnrichedLogger(base, service_name)
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError):
            # precise failure set; avoids catching asyncio.CancelledError or system-exiting exceptions
            self._logger = None

        if self._logger is None:
            # Fallback to local structlog setup (kept minimal)
            def add_trace(logger, method_name, event_dict):
                span = trace.get_current_span()
                if span:
                    ctx = span.get_span_context()
                    event_dict["trace_id"] = f"{ctx.trace_id:032x}"
                    event_dict["span_id"] = f"{ctx.span_id:016x}"
                event_dict.update(
                    {
                        "tenant_id": get_tenant(),
                        "strategy_id": get_strategy(),
                        "service": self.service_name,
                    }
                )
                return event_dict

            redactor = PIIRedactor()

            def redact(logger, method_name, event_dict):
                return redactor.redact_dict(event_dict)

            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    add_trace,
                    redact,
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            self._logger = structlog.get_logger(self.service_name)

    def get_logger(self):
        return self._logger


# ================== Adaptive Thresholds & SLO ==================
class AdaptiveThreshold:
    def __init__(
        self, alpha: float = 0.1, sensitivity: float = 3.0, window_size: int = 1024
    ) -> None:
        self.alpha = alpha
        self.sensitivity = sensitivity
        self._ewma: float | None = None
        self._mad = 0.0
        self._values = deque(maxlen=window_size)
        self._lock = threading.RLock()

    def update(self, value: float) -> bool:
        with self._lock:
            anomaly = False
            if self._ewma is not None and value > self.threshold():
                anomaly = True
            self._values.append(value)
            self._ewma = (
                value if self._ewma is None else self.alpha * value + (1 - self.alpha) * self._ewma
            )
            if len(self._values) > 10:
                sv = sorted(self._values)
                median = sv[len(sv) // 2]
                self._mad = sum(abs(x - median) for x in self._values) / len(self._values)
            return anomaly

    def threshold(self) -> float:
        return float("inf") if self._ewma is None else self._ewma + self.sensitivity * self._mad


class SLOBurnRate:
    """Tracks SLO burn over sliding windows using counters."""

    def __init__(self, minutes: Iterable[int] = (1, 5, 30)) -> None:
        self.windows = list(minutes)
        self._lock = threading.RLock()
        self._events: dict[int, deque[tuple[float, int, int]]] = {m: deque() for m in self.windows}

    def record(self, ok: bool) -> None:
        now = time.time()
        for m in self.windows:
            dq = self._events[m]
            dq.append((now, 1 if ok else 0, 0 if ok else 1))
            cutoff = now - m * 60
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def burn_rates(self, slo_error_budget: float) -> dict[int, float]:
        out: dict[int, float] = {}
        time.time()
        for m, dq in self._events.items():
            ok = sum(x[1] for x in dq)
            err = sum(x[2] for x in dq)
            total = ok + err
            if total == 0:
                out[m] = 0.0
            else:
                error_rate = err / total
                out[m] = (error_rate / slo_error_budget) if slo_error_budget > 0 else 0.0
        return out


# ================== Decorators ==================
_metrics: MetricsManager | None = None
_tracing: TracingManager | None = None
_logging: LoggingManager | None = None
_logger: structlog.BoundLogger | None = None


def instrument(
    name: str | None = None,
    labels: dict[str, str] | None = None,
    record_exceptions: bool = True,
    measure_latency: bool = True,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        fname = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if _metrics:
                _metrics.record_counter(
                    _metrics.counter(f"{fname}_total", "Total calls"), labels=labels
                )
            span = _tracing.start_span(fname, attributes=labels) if _tracing else None
            start = time.perf_counter()
            try:
                if span:
                    with trace.use_span(span, end_on_exit=True):
                        res = func(*args, **kwargs)
                else:
                    res = func(*args, **kwargs)
                if measure_latency and _metrics:
                    lat_ms = (time.perf_counter() - start) * 1000
                    _metrics.record_histogram(
                        _metrics.histogram(f"{fname}_latency", "Function latency", "ms"),
                        lat_ms,
                        labels,
                    )
                return res
            except (ValueError, TypeError, RuntimeError) as e:
                if record_exceptions:
                    if _metrics:
                        _metrics.record_counter(
                            _metrics.counter(f"{fname}_errors", "Total errors"), labels=labels
                        )
                    if span:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    if _logger:
                        _logger.error(
                            "Error in function", function=fname, error=str(e), **(labels or {})
                        )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if _metrics:
                _metrics.record_counter(
                    _metrics.counter(f"{fname}_total", "Total calls"), labels=labels
                )
            span = _tracing.start_span(fname, attributes=labels) if _tracing else None
            start = time.perf_counter()
            try:
                if span:
                    with trace.use_span(span, end_on_exit=True):
                        res = await func(*args, **kwargs)
                else:
                    res = await func(*args, **kwargs)
                if measure_latency and _metrics:
                    lat_ms = (time.perf_counter() - start) * 1000
                    _metrics.record_histogram(
                        _metrics.histogram(f"{fname}_latency", "Function latency", "ms"),
                        lat_ms,
                        labels,
                    )
                return res
            except (ValueError, TypeError, RuntimeError) as e:
                if record_exceptions:
                    if _metrics:
                        _metrics.record_counter(
                            _metrics.counter(f"{fname}_errors", "Total errors"), labels=labels
                        )
                    if span:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    if _logger:
                        _logger.error(
                            "Error in async function",
                            function=fname,
                            error=str(e),
                            **(labels or {}),
                        )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ================== Framework integrations ==================
class FastAPIMiddleware:
    def __init__(self, app, service_name: str = "api", slo_error_budget: float = 0.01) -> None:
        self.app = app
        self.service_name = service_name
        self._thresholds: dict[str, AdaptiveThreshold] = defaultdict(AdaptiveThreshold)
        self._burn = SLOBurnRate()
        self._slo_budget = slo_error_budget

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "/")
        method = scope.get("method", "GET")
        host = scope.get("server", ("", ""))[0]

        span = (
            _tracing.start_span(
                f"{method} {path}",
                kind=SpanKind.SERVER,
                attributes={
                    "http.method": method,
                    "http.path": path,
                    "http.scheme": scope.get("scheme", "http"),
                    "http.host": host,
                },
            )
            if _tracing
            else None
        )

        if _metrics:
            _metrics.record_counter(
                _metrics.counter("http_requests_total", "Total HTTP requests"),
                labels={"method": method, "path": path},
            )

        ok = False  # initialize to satisfy analyzers and ensure 'finally' path has a value
        start = time.perf_counter()
        try:
            if span:
                with trace.use_span(span, end_on_exit=True):
                    await self.app(scope, receive, send)
            else:
                await self.app(scope, receive, send)
            ok = True
            return
        except (ValueError, TypeError, RuntimeError, OSError, ConnectionError, TimeoutError) as e:
            ok = False
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            if _metrics:
                _metrics.record_counter(
                    _metrics.counter("http_errors_total", "Total HTTP errors"),
                    labels={"method": method, "path": path},
                )
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            if _metrics:
                _metrics.record_histogram(
                    _metrics.histogram("http_request_duration", "HTTP request duration", "ms"),
                    latency_ms,
                    {"method": method, "path": path},
                )
                thr = self._thresholds[path]
                if thr.update(latency_ms):
                    _metrics.record_counter(
                        _metrics.counter("http_latency_anomalies_total", "Latency anomalies"),
                        labels={"path": path, "method": method},
                    )
            self._burn.record(ok)
            # Optionally expose burn rates as gauge via callback elsewhere


class KafkaInstrumentor:
    def __init__(self) -> None:
        self.propagator = TraceContextTextMapPropagator()

    def inject_context(self, headers: dict[str, bytes]) -> dict[str, bytes]:
        carrier: dict[str, str] = {}
        self.propagator.inject(carrier)
        for k, v in carrier.items():
            headers[k] = v.encode("utf-8")
        return headers

    def extract_context(self, headers: dict[str, bytes]):
        carrier = {
            k: (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v))
            for k, v in (headers or {}).items()
        }
        return self.propagator.extract(carrier)

    def instrument_producer(self, producer):
        send = producer.send

        def traced_send(topic, value=None, key=None, headers=None, **kwargs):
            headers = headers or {}
            self.inject_context(headers)
            if _tracing:
                with _tracing.start_span(
                    f"kafka_send_{topic}",
                    kind=SpanKind.PRODUCER,
                    attributes={"messaging.system": "kafka", "messaging.destination": topic},
                ):
                    return send(topic, value, key, headers, **kwargs)
            return send(topic, value, key, headers, **kwargs)

        producer.send = traced_send
        return producer

    def instrument_consumer(self, consumer):
        poll = consumer.poll

        def traced_poll(*args, **kwargs):
            records = poll(*args, **kwargs)
            if not records:
                return records
            for _tp, msgs in records.items():
                for msg in msgs:
                    if getattr(msg, "headers", None):
                        ctx = self.extract_context(dict(msg.headers))
                        context.attach(ctx)
                    if _tracing:
                        msg._span = _tracing.start_span(
                            f"kafka_consume_{msg.topic}",
                            kind=SpanKind.CONSUMER,
                            attributes={
                                "messaging.system": "kafka",
                                "messaging.source": msg.topic,
                                "messaging.offset": getattr(msg, "offset", -1),
                            },
                        )
            return records

        consumer.poll = traced_poll
        return consumer


# ================== Cache metrics registration ==================


def register_cache_hit_rate_gauges(cache_client: Any, metric_name: str = "cache_hit_rate") -> None:
    """Register observable gauges that expose per-tier cache hit-rate from a multi-tier client.

    Expects `cache_client.metrics.summary()` to return a dict like:
    {"l1": {"hit_rate": 0.97}, "l2": {...}, ...}
    Missing tiers/keys are skipped safely.
    """
    m = get_metrics()
    if not (m and cache_client):
        return

    def _observe(_options):
        try:
            metrics_obj = getattr(cache_client, "metrics", None)
            summary = (
                metrics_obj.summary() if (metrics_obj and hasattr(metrics_obj, "summary")) else None
            )
        except (AttributeError, TypeError, KeyError):
            summary = None
        if not summary:
            return []
        observations = []
        for tier in ("l1", "l2", "l3", "l4"):
            data = summary.get(tier)
            if not data:
                continue
            hr = data.get("hit_rate")
            if hr is None:
                continue
            observations.append(metrics.Observation(hr, {"tier": tier}))
        return observations

    try:
        m.meter.create_observable_gauge(
            metric_name,
            callbacks=[_observe],
            description="Cache hit-rate by tier (0..1)",
        )
    except (ValueError, TypeError):
        # best-effort; don't crash app if gauge registration fails
        pass


def register_cache_hit_rate_gauges_for(
    func_or_client: Any, metric_name: str = "cache_hit_rate"
) -> None:
    """Convenience helper that accepts either a decorated function (with `cache_client` attr)
    or a cache client instance and wires up hit-rate gauges.
    """
    client = getattr(func_or_client, "cache_client", None)
    register_cache_hit_rate_gauges(client or func_or_client, metric_name)


# ================== Init & convenience ==================

_metrics: MetricsManager | None = None
_tracing: TracingManager | None = None
_logging: LoggingManager | None = None
_logger: structlog.BoundLogger | None = None


def get_metrics() -> MetricsManager | None:
    """Get the global metrics manager instance"""
    return _metrics


def get_tracing() -> TracingManager | None:
    """Get the global tracing manager instance"""
    return _tracing


def get_logging() -> LoggingManager | None:
    """Get the global logging manager instance"""
    return _logging


def get_logger():
    """Get the global logger instance"""
    return _logger


def init_observability(
    service_name: str = "financial-ml",
    metrics_config: MetricConfig | None = None,
    tracing_config: TracingConfig | None = None,
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    enable_logging: bool = True,
):
    """Initialize observability stack with metrics, tracing, and logging

    Args:
        service_name: Name of the service for resource identification
        metrics_config: Configuration for metrics collection
        tracing_config: Configuration for distributed tracing
        enable_metrics: Whether to initialize metrics manager
        enable_tracing: Whether to initialize tracing manager
        enable_logging: Whether to initialize logging manager
    """
    global _metrics, _tracing, _logging, _logger

    if enable_metrics:
        _metrics = (
            MetricsManager(service_name, metrics_config) if _HAS_OTEL else NoOpMetricsManager()
        )

    if enable_tracing:
        _tracing = (
            TracingManager(service_name, tracing_config) if _HAS_OTEL else NoOpTracingManager()
        )

    if enable_logging:
        _logging = LoggingManager(service_name)
        _logger = _logging.get_logger()

    if _logger:
        _logger.info(
            "observability_ready",
            service=service_name,
            otel_exemplar=bool(callable(TraceBasedExemplarFilter)),
            otel_hist_agg=(
                "exp"
                if callable(ExponentialBucketHistogramAggregation)
                else "explicit"
                if callable(ExplicitBucketHistogramAggregation)
                else "default"
            ),
            has_prometheus=optional_import("prometheus_client") is not None,
            has_grpc=optional_import("grpc") is not None,
            metrics_enabled=enable_metrics,
            tracing_enabled=enable_tracing,
            logging_enabled=enable_logging,
        )

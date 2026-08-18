"""
Observability SDK - Golden Reference Library

This module is the ONLY sanctioned interface for telemetry.
Direct OpenTelemetry, Prometheus, or logging calls outside this module are FORBIDDEN.

Enforcement: Ruff rule + pre-commit hook (see pyproject.toml)
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, ParamSpec, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.metrics import Meter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from prometheus_client import Counter as PromCounter
from prometheus_client import Histogram as PromHistogram
from prometheus_client import Info

__all__ = [
    "Lane",
    "Component",
    "ErrorType",
    "ObservabilitySDK",
    "get_sdk",
    "span",
    "time_it",
    "record_artifact_registered",
    "record_cache_access",
    "record_inference_latency",
    "record_error",
]


P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class Lane(StrEnum):
    """Backtest lane discriminator (bounded cardinality)."""

    A = "A"  # Low-fidelity / fast
    B = "B"  # High-fidelity / slow


class Component(StrEnum):
    """System component identifiers (bounded cardinality)."""

    PREPROCESSOR = "preprocessor"
    INFERENCE = "inference"
    STRATEGY = "strategy"
    CACHE = "cache"
    REGISTRY = "registry"
    ORCHESTRATION = "orchestration"


class ErrorType(StrEnum):
    """Error type classification (bounded cardinality)."""

    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    DETERMINISM = "determinism"
    CONTRACT = "contract"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class CacheTier(StrEnum):
    """Cache tier identifiers."""

    L1 = "L1"  # In-process
    L2 = "L2"  # Shared memory
    L3 = "L3"  # Redis/Dragonfly
    L4 = "L4"  # Persistent store


@dataclass(frozen=True, slots=True)
class SpanContext:
    """
    Required context for all spans.

    These fields MUST be present on every span to enable correlation.
    High-cardinality fields (plan_hash, fold_id) are span attributes only,
    never metric labels.
    """

    run_id: str
    lane: Lane
    strategy_id: str
    # High-cardinality (trace attributes only, not metric labels)
    plan_hash: str | None = None
    fold_id: str | None = None
    worker_id: str | None = None


@dataclass
class ObservabilitySDK:
    """
    Centralized observability interface.

    All telemetry MUST go through this SDK. This ensures:
    - Consistent span/metric naming
    - Bounded label cardinality on metrics
    - Required fields enforced everywhere
    - Sampling policies applied consistently
    """

    service_name: str = "marketmind"
    _tracer: Tracer = field(init=False, repr=False)
    _meter: Meter = field(init=False, repr=False)

    # Prometheus metrics (bounded cardinality labels only)
    _inference_latency: PromHistogram = field(init=False, repr=False)
    _cache_ops: PromCounter = field(init=False, repr=False)
    _artifacts_registered: PromCounter = field(init=False, repr=False)
    _errors: PromCounter = field(init=False, repr=False)
    _build_info: Info = field(init=False, repr=False)

    # Label cardinality bounds
    ALLOWED_METRIC_LABELS: Final[frozenset[str]] = frozenset(
        {
            "lane",  # A/B only
            "component",  # bounded enum
            "error_type",  # bounded enum
            "cache_tier",  # L1/L2/L3/L4
            "cache_op",  # get/put/invalidate
            "strategy_family",  # optional grouping (bounded)
        }
    )

    # Fields that are HIGH CARDINALITY - trace only, never metrics
    HIGH_CARD_FIELDS: Final[frozenset[str]] = frozenset(
        {
            "run_id",
            "plan_hash",
            "fold_id",
            "worker_id",
            "strategy_id",  # Can grow; use strategy_family for metrics
            "bar_ts",
            "symbol",
            "artifact_id",
            "content_hash",
        }
    )

    def __post_init__(self) -> None:
        """Initialize OpenTelemetry and Prometheus resources."""
        # OpenTelemetry
        self._tracer = trace.get_tracer(self.service_name)
        self._meter = metrics.get_meter(self.service_name)

        # Prometheus metrics with BOUNDED labels only
        self._inference_latency = PromHistogram(
            "mm_inference_latency_ms",
            "Inference latency in milliseconds",
            ["lane", "component"],
            buckets=[0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
        )

        self._cache_ops = PromCounter(
            "mm_cache_operations_total",
            "Cache operations by tier and result",
            ["cache_tier", "cache_op", "hit"],
        )

        self._artifacts_registered = PromCounter(
            "mm_artifacts_registered_total",
            "Artifacts registered to registry",
            ["lane", "artifact_type"],
        )

        self._errors = PromCounter(
            "mm_errors_total",
            "Errors by type and component",
            ["error_type", "component"],
        )

        self._build_info = Info(
            "mm_build",
            "Build information",
        )

    def set_build_info(
        self,
        version: str,
        commit_sha: str,
        python_version: str,
    ) -> None:
        """Set build information (called once at startup)."""
        self._build_info.info(
            {
                "version": version,
                "commit_sha": commit_sha,
                "python_version": python_version,
            }
        )

    @contextmanager
    def span(
        self,
        name: str,
        ctx: SpanContext,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """
        Create a span with required context.

        Args:
            name: Span name (use component.operation format)
            ctx: Required span context
            kind: Span kind (internal, client, server, etc.)
            attributes: Additional span attributes (high-card OK here)

        Yields:
            Active span
        """
        # Build attributes with required fields
        span_attrs = {
            "run_id": ctx.run_id,
            "lane": ctx.lane.value,
            "strategy_id": ctx.strategy_id,
        }

        # Add optional high-cardinality fields
        if ctx.plan_hash:
            span_attrs["plan_hash"] = ctx.plan_hash
        if ctx.fold_id:
            span_attrs["fold_id"] = ctx.fold_id
        if ctx.worker_id:
            span_attrs["worker_id"] = ctx.worker_id

        # Merge additional attributes
        if attributes:
            span_attrs.update(attributes)

        with self._tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=span_attrs,
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def record_inference_latency(
        self,
        latency_ms: float,
        lane: Lane,
        component: Component = Component.INFERENCE,
    ) -> None:
        """
        Record inference latency metric.

        Args:
            latency_ms: Latency in milliseconds
            lane: Backtest lane
            component: Component identifier
        """
        self._inference_latency.labels(
            lane=lane.value,
            component=component.value,
        ).observe(latency_ms)

    def record_cache_access(
        self,
        tier: CacheTier,
        operation: str,
        hit: bool,
    ) -> None:
        """
        Record cache operation.

        Args:
            tier: Cache tier (L1/L2/L3/L4)
            operation: Operation type (get/put/invalidate)
            hit: Whether operation was a hit (for get) or success (for put)
        """
        self._cache_ops.labels(
            cache_tier=tier.value,
            cache_op=operation,
            hit=str(hit).lower(),
        ).inc()

    def record_artifact_registered(
        self,
        lane: Lane,
        artifact_type: str,
    ) -> None:
        """
        Record artifact registration.

        Args:
            lane: Backtest lane
            artifact_type: Type of artifact (bounded set expected)
        """
        self._artifacts_registered.labels(
            lane=lane.value,
            artifact_type=artifact_type,
        ).inc()

    def record_error(
        self,
        error_type: ErrorType,
        component: Component,
        ctx: SpanContext | None = None,
        exception: Exception | None = None,
    ) -> None:
        """
        Record error occurrence.

        Args:
            error_type: Error classification
            component: Component where error occurred
            ctx: Optional span context for trace correlation
            exception: Optional exception for logging
        """
        self._errors.labels(
            error_type=error_type.value,
            component=component.value,
        ).inc()

        # Log with context for debugging
        extra = {}
        if ctx:
            extra = {
                "run_id": ctx.run_id,
                "lane": ctx.lane.value,
                "strategy_id": ctx.strategy_id,
            }

        if exception:
            logger.error(
                f"Error in {component.value}: {error_type.value}",
                exc_info=exception,
                extra=extra,
            )

    def time_it(
        self,
        name: str,
        component: Component,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Decorator to time a function and record latency.

        Args:
            name: Metric name suffix
            component: Component identifier

        Returns:
            Decorator function
        """

        def decorator(fn: Callable[P, T]) -> Callable[P, T]:
            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    # Extract lane from kwargs if present
                    lane = kwargs.get("lane", Lane.A)
                    if isinstance(lane, str):
                        lane = Lane(lane)
                    self.record_inference_latency(elapsed_ms, lane, component)

            return wrapper

        return decorator


# Global SDK instance (initialized once)
_sdk: ObservabilitySDK | None = None


def get_sdk() -> ObservabilitySDK:
    """Get or create the global ObservabilitySDK instance."""
    global _sdk
    if _sdk is None:
        _sdk = ObservabilitySDK()
    return _sdk


# Convenience functions that delegate to global SDK


def span(
    name: str,
    ctx: SpanContext,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> contextmanager:
    """Create a span with required context. See ObservabilitySDK.span."""
    return get_sdk().span(name, ctx, kind, attributes)


def time_it(
    name: str,
    component: Component,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to time a function. See ObservabilitySDK.time_it."""
    return get_sdk().time_it(name, component)


def record_artifact_registered(lane: Lane, artifact_type: str) -> None:
    """Record artifact registration."""
    get_sdk().record_artifact_registered(lane, artifact_type)


def record_cache_access(tier: CacheTier, operation: str, hit: bool) -> None:
    """Record cache operation."""
    get_sdk().record_cache_access(tier, operation, hit)


def record_inference_latency(
    latency_ms: float,
    lane: Lane,
    component: Component = Component.INFERENCE,
) -> None:
    """Record inference latency metric."""
    get_sdk().record_inference_latency(latency_ms, lane, component)


def record_error(
    error_type: ErrorType,
    component: Component,
    ctx: SpanContext | None = None,
    exception: Exception | None = None,
) -> None:
    """Record error occurrence."""
    get_sdk().record_error(error_type, component, ctx, exception)


# =============================================================================
# ENFORCEMENT: Lint rules to forbid direct OTel/Prometheus imports
# =============================================================================
#
# Add to pyproject.toml:
#
# [tool.ruff.lint.per-file-ignores]
# # Only observability/sdk.py may import these
# "!py/observability/sdk.py" = ["TID251"]
#
# [tool.ruff.lint.flake8-tidy-imports.banned-api]
# "opentelemetry".msg = "Use observability.sdk instead of direct OpenTelemetry imports"
# "prometheus_client".msg = "Use observability.sdk instead of direct Prometheus imports"
#
# =============================================================================

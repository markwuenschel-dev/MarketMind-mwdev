# tests/unit/test_metrics.py
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

try:
    from pysrc.ops.observability import (
        _HAS_OTEL,
        CircuitBreaker,
        MetricConfig,
        MetricsManager,
        SafeOTLPMetricExporter,
        _build_histogram_aggregation,
    )

    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False
    _HAS_OTEL = False

    # Minimal stubs for type checking
    class CircuitBreaker:  # type: ignore[no-redef]
        def __init__(self, fail_threshold: int = 5, reset_after_sec: int = 30):
            pass

    class MetricConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MetricsManager:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class SafeOTLPMetricExporter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    def _build_histogram_aggregation():  # type: ignore[no-redef]
        return None


# ==================== Fixtures ====================


@pytest.fixture
def circuit_breaker():
    """Circuit breaker for exporter tests"""
    return CircuitBreaker(fail_threshold=3, reset_after_sec=1)


@pytest.fixture
def allowlist():
    """Endpoint allowlist for exporter tests"""
    return ("localhost:4317", "otel-collector.internal:4317")


@pytest.fixture
def mock_otel_stack():
    """Centralized OTel mocking"""
    with (
        patch("pysrc.ops.observability.Resource") as mock_res,
        patch("pysrc.ops.observability.MeterProvider") as mock_mp,
        patch("pysrc.ops.observability.metrics") as mock_metrics,
        patch("pysrc.ops.observability.PrometheusMetricReader"),
        patch("pysrc.ops.observability.PeriodicExportingMetricReader"),
        patch("pysrc.ops.observability.View"),
    ):
        mock_meter = MagicMock()
        mock_mp.return_value = MagicMock()
        mock_metrics.get_meter.return_value = mock_meter
        mock_res.create.return_value = MagicMock()

        yield {"meter": mock_meter, "resource": mock_res}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure deterministic resource attributes"""
    for key in ["SERVICE_VERSION", "DEPLOY_ENV", "INSTANCE_ID", "CLOUD_REGION", "EDGE_COLO"]:
        monkeypatch.delenv(key, raising=False)


# ==================== SafeOTLPMetricExporter Tests ====================


class TestSafeOTLPMetricExporter:
    """Comprehensive tests for SafeOTLPMetricExporter"""

    @pytest.mark.skipif(
        not _MODULE_AVAILABLE or not _HAS_OTEL, reason="OpenTelemetry not available"
    )
    def test_allowlist_validation_passes_for_allowed_endpoint(self, allowlist, circuit_breaker):
        """Test that allowed endpoints pass validation"""
        exporter = SafeOTLPMetricExporter(
            allowlist=allowlist,
            breaker=circuit_breaker,
            timeout=5.0,
            endpoint="localhost:4317",
            insecure=True,
        )
        assert exporter._breaker is circuit_breaker

    @pytest.mark.skipif(
        not _MODULE_AVAILABLE or not _HAS_OTEL, reason="OpenTelemetry not available"
    )
    def test_allowlist_validation_rejects_disallowed_endpoint(self, allowlist, circuit_breaker):
        """Test that disallowed endpoints are rejected"""
        with pytest.raises(ValueError, match="not in allow-list"):
            SafeOTLPMetricExporter(
                allowlist=allowlist,
                breaker=circuit_breaker,
                timeout=5.0,
                endpoint="evil.hacker.com:4317",
                insecure=True,
            )

    @pytest.mark.skipif(
        not _MODULE_AVAILABLE or not _HAS_OTEL, reason="OpenTelemetry not available"
    )
    def test_export_breaker_logic_skips_when_open(self, allowlist, circuit_breaker):
        """Test that export is skipped when circuit breaker is open"""
        circuit_breaker.fail_count = 10
        circuit_breaker.opened_at = time.time()

        with patch("pysrc.ops.observability.OTLPMetricExporter.export") as mock_export:
            exporter = SafeOTLPMetricExporter(
                allowlist=allowlist,
                breaker=circuit_breaker,
                timeout=5.0,
                endpoint="localhost:4317",
                insecure=True,
            )

            exporter.export(MagicMock())
            mock_export.assert_not_called()

    @pytest.mark.skipif(
        not _MODULE_AVAILABLE or not _HAS_OTEL, reason="OpenTelemetry not available"
    )
    def test_export_breaker_logic_resets_on_success(self, allowlist, circuit_breaker):
        """Test that circuit breaker resets on successful export"""
        with patch("pysrc.ops.observability.OTLPMetricExporter.export"):
            exporter = SafeOTLPMetricExporter(
                allowlist=allowlist,
                breaker=circuit_breaker,
                timeout=5.0,
                endpoint="localhost:4317",
                insecure=True,
            )

            exporter.export(MagicMock())
            assert circuit_breaker.fail_count == 0

    @pytest.mark.skipif(
        not _MODULE_AVAILABLE or not _HAS_OTEL, reason="OpenTelemetry not available"
    )
    def test_export_breaker_logic_increments_on_failure(self, allowlist, circuit_breaker):
        """Test that circuit breaker increments fail count on export failure"""
        with (
            patch("pysrc.ops.observability.OTLPMetricExporter.export") as mock_export,
            patch("pysrc.ops.observability._logger"),
        ):
            mock_export.side_effect = TimeoutError("timeout")

            exporter = SafeOTLPMetricExporter(
                allowlist=allowlist,
                breaker=circuit_breaker,
                timeout=5.0,
                endpoint="localhost:4317",
                insecure=True,
            )

            exporter.export(MagicMock())
            assert circuit_breaker.fail_count == 1


# ==================== _build_histogram_aggregation Tests ====================


class TestBuildHistogramAggregation:
    """Tests for histogram aggregation resolution logic"""

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_exponential_fallback_with_max_size(self):
        """Test exponential aggregation with max_size parameter"""
        with patch("pysrc.ops.observability.ExponentialBucketHistogramAggregation") as mock_exp:
            mock_exp.return_value = MagicMock()

            agg = _build_histogram_aggregation()

            assert agg is not None
            assert mock_exp.call_count >= 1

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_exponential_fallback_without_max_size_on_typeerror(self):
        """Test fallback when max_size parameter not supported"""
        with patch("pysrc.ops.observability.ExponentialBucketHistogramAggregation") as mock_exp:
            call_count = [0]

            def side_effect(*args: Any, **kwargs: Any) -> Any:
                call_count[0] += 1
                if call_count[0] == 1 and (args or kwargs):
                    raise TypeError("max_size not supported")
                return MagicMock()

            mock_exp.side_effect = side_effect
            agg = _build_histogram_aggregation()
            assert agg is not None

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_explicit_fallback_when_exponential_fails(self):
        """Test fallback to explicit aggregation when exponential fails"""
        with (
            patch("pysrc.ops.observability.ExponentialBucketHistogramAggregation") as mock_exp,
            patch("pysrc.ops.observability.ExplicitBucketHistogramAggregation") as mock_explicit,
        ):
            mock_exp.side_effect = ValueError("not supported")
            mock_explicit.return_value = MagicMock()

            agg = _build_histogram_aggregation()

            assert agg is not None
            mock_explicit.assert_called_once()

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_explicit_fallback_with_boundaries(self):
        """Test that explicit aggregation includes boundaries parameter"""
        with (
            patch("pysrc.ops.observability.ExponentialBucketHistogramAggregation") as mock_exp,
            patch("pysrc.ops.observability.ExplicitBucketHistogramAggregation") as mock_explicit,
        ):
            mock_exp.side_effect = TypeError("no exponential")
            mock_explicit.return_value = MagicMock()

            agg = _build_histogram_aggregation()

            call_args = mock_explicit.call_args
            assert "boundaries" in call_args[1]
            assert agg is not None

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_returns_none_when_all_fail(self):
        """Test graceful None return when all aggregation types fail"""
        with (
            patch("pysrc.ops.observability.ExponentialBucketHistogramAggregation") as mock_exp,
            patch("pysrc.ops.observability.ExplicitBucketHistogramAggregation") as mock_explicit,
        ):
            mock_exp.side_effect = ValueError("fail")
            mock_explicit.side_effect = TypeError("fail")

            agg = _build_histogram_aggregation()
            assert agg is None


# ==================== MetricsManager Initialization Tests ====================


class TestMetricsManagerInitialization:
    """Tests for MetricsManager initialization and configuration"""

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_resource_creation_with_env_vars(self, monkeypatch, mock_otel_stack):
        """Test that resource attributes are populated from environment variables"""
        monkeypatch.setenv("SERVICE_VERSION", "3.1.4")
        monkeypatch.setenv("DEPLOY_ENV", "staging")
        monkeypatch.setenv("INSTANCE_ID", "pod-xyz-123")
        monkeypatch.setenv("CLOUD_REGION", "us-west-2")
        monkeypatch.setenv("EDGE_COLO", "SFO")

        config = MetricConfig(buffered_emit=False)
        _ = MetricsManager("test-svc", config)

        call_args = mock_otel_stack["resource"].create.call_args
        attrs = call_args[1].get("attributes") or call_args[0][0]

        assert attrs["service.version"] == "3.1.4"
        assert attrs["deployment.environment"] == "staging"
        assert attrs["service.instance.id"] == "pod-xyz-123"
        assert attrs["cloud.region"] == "us-west-2"
        assert attrs["colo"] == "SFO"

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_view_setup_with_histogram_aggregation(self, mock_otel_stack):
        """Test that views are created when histogram aggregation is available"""
        with (
            patch("pysrc.ops.observability._build_histogram_aggregation") as mock_build,
            patch("pysrc.ops.observability.View") as mock_view,
        ):
            mock_build.return_value = MagicMock()

            config = MetricConfig(buffered_emit=False)
            _ = MetricsManager("test-svc", config)

            assert mock_view.call_count >= 1

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_view_setup_skipped_when_aggregation_none(self, mock_otel_stack):
        """Test that views are skipped when histogram aggregation is None"""
        with patch("pysrc.ops.observability._build_histogram_aggregation") as mock_build:
            mock_build.return_value = None

            config = MetricConfig(buffered_emit=False)
            mgr = MetricsManager("test-svc", config)

            # Should not raise, views list should be empty
            assert mgr is not None

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_exemplar_filter_setup_when_enabled_and_callable(self, mock_otel_stack):
        """Test that exemplar filter is set up when enabled and available"""
        with patch("pysrc.ops.observability.TraceBasedExemplarFilter", create=True) as mock_filter:
            mock_filter.return_value = MagicMock()

            config = MetricConfig(enable_exemplars=True, buffered_emit=False)
            _ = MetricsManager("test-svc", config)

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_exemplar_filter_setup_skipped_when_disabled(self, mock_otel_stack):
        """Test that exemplar filter is not created when disabled"""
        with patch("pysrc.ops.observability.TraceBasedExemplarFilter", create=True) as mock_filter:
            config = MetricConfig(enable_exemplars=False, buffered_emit=False)
            _ = MetricsManager("test-svc", config)

            mock_filter.assert_not_called()

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_exemplar_filter_setup_handles_callable_check(self, mock_otel_stack):
        """Test that exemplar filter setup handles missing TraceBasedExemplarFilter"""
        with patch("pysrc.ops.observability.TraceBasedExemplarFilter", None):
            config = MetricConfig(enable_exemplars=True, buffered_emit=False)
            mgr = MetricsManager("test-svc", config)

            # Should not raise
            assert mgr is not None

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_meter_provider_initialization(self, mock_otel_stack):
        """Test that MeterProvider is initialized and set globally"""
        config = MetricConfig(buffered_emit=False)
        mgr = MetricsManager("test-svc", config)

        # Provider should be created and meter should be available
        assert mgr.meter is not None

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_meta_metrics_creation_dropped_events(self, mock_otel_stack):
        """Test that dropped_events meta-metric counter is created"""
        config = MetricConfig(buffered_emit=True)
        mgr = MetricsManager("test-svc", config)

        meter = mock_otel_stack["meter"]
        counter_calls = [c[0][0] for c in meter.create_counter.call_args_list]

        assert "observability_dropped_events" in counter_calls
        mgr.shutdown()

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_meta_metrics_creation_cardinality_overflows(self, mock_otel_stack):
        """Test that cardinality_overflows meta-metric counter is created"""
        config = MetricConfig(buffered_emit=True)
        mgr = MetricsManager("test-svc", config)

        meter = mock_otel_stack["meter"]
        counter_calls = [c[0][0] for c in meter.create_counter.call_args_list]

        assert "observability_cardinality_overflows" in counter_calls
        mgr.shutdown()

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_meta_metrics_creation_queue_depth_gauge(self, mock_otel_stack):
        """Test that queue_depth observable gauge is created"""
        config = MetricConfig(buffered_emit=True)
        mgr = MetricsManager("test-svc", config)

        meter = mock_otel_stack["meter"]
        gauge_calls = [c[0][0] for c in meter.create_observable_gauge.call_args_list]

        assert "observability_queue_depth" in gauge_calls
        mgr.shutdown()

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_background_flusher_thread_started_when_buffered(self, mock_otel_stack):
        """Test that background flusher thread is started when buffering enabled"""
        config = MetricConfig(buffered_emit=True, flush_every_ms=50)
        mgr = MetricsManager("test-svc", config)

        assert mgr._flush_thread is not None
        assert mgr._flush_thread.is_alive()
        assert mgr._flush_thread.daemon is True
        assert mgr._flush_thread.name == "otel-metrics-flusher"

        mgr.shutdown()

    @pytest.mark.skipif(not _MODULE_AVAILABLE, reason="Module not available")
    def test_background_flusher_not_started_when_unbuffered(self, mock_otel_stack):
        """Test that background flusher thread is not created when buffering disabled"""
        config = MetricConfig(buffered_emit=False)
        mgr = MetricsManager("test-svc", config)

        assert mgr._flush_thread is None

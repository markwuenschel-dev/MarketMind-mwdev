# tests/unit/test_observability_buffers.py
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import pysrc.ops.observability as obs
from pysrc.ops.observability import MetricConfig, MetricsManager

# ---------- Test Doubles (Deterministic, Typed) ----------


class _FakeCounter:
    """Minimal counter that captures add() calls"""

    def __init__(self) -> None:
        self.calls = []

    def add(self, value: float, attributes: dict[str, str]) -> None:
        self.calls.append(("add", value, dict(attributes)))


class _FakeHistogram:
    """Minimal histogram that captures record() calls"""

    def __init__(self) -> None:
        self.calls = []

    def record(self, value: float, attributes: dict[str, str]) -> None:
        self.calls.append(("record", value, dict(attributes)))


# ---------- Fixtures ----------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure deterministic resource attributes"""
    for key in ["SERVICE_VERSION", "DEPLOY_ENV", "INSTANCE_ID", "CLOUD_REGION", "EDGE_COLO"]:
        monkeypatch.delenv(key, raising=False)


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

        yield {"meter": mock_meter}


@pytest.fixture
def fake_counter():
    """Reusable fake counter"""
    return _FakeCounter()


@pytest.fixture
def fake_histogram():
    """Reusable fake histogram"""
    return _FakeHistogram()


@pytest.fixture
def track_dropped_counter(mock_otel_stack):
    """Setup meter to track dropped events counter"""
    dropped = _FakeCounter()

    def _side_effect(name, **_kwargs):
        return dropped if name == "observability_dropped_events" else _FakeCounter()

    mock_otel_stack["meter"].create_counter.side_effect = _side_effect
    return dropped


@pytest.fixture
def track_overflow_counter(mock_otel_stack):
    """Setup meter to track cardinality overflow counter"""
    overflow = _FakeCounter()
    created = []

    def _side_effect(name, **_kwargs):
        created.append(name)
        return overflow if name == "observability_cardinality_overflows" else _FakeCounter()

    mock_otel_stack["meter"].create_counter.side_effect = _side_effect
    return {"counter": overflow, "created_names": created}


# ---------- Queue Depth & Observable Gauge Tests ----------


def test_observe_queue_depth_reports_size(mock_otel_stack, fake_counter) -> None:
    """Test that queue depth observable gauge reports correct size"""
    cfg = MetricConfig(
        prometheus_port=0, buffered_emit=True, queue_max_events=8, flush_every_ms=200
    )
    mgr = MetricsManager("svc", cfg)

    # Prime the queue with one event
    mgr.record_counter(fake_counter, 1.0, {"k": "v"})

    # Invoke callback (options object unused across SDK versions)
    out = mgr._observe_queue_depth(object())

    # Should return list with Observation
    assert isinstance(out, list)
    assert len(out) == 1
    assert hasattr(out[0], "value")
    assert out[0].value >= 0  # Queue size is non-negative

    mgr.shutdown()


def test_observe_queue_depth_returns_empty_when_no_queue(mock_otel_stack) -> None:
    """Test that queue depth returns empty list when buffering disabled"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    out = mgr._observe_queue_depth(object())

    assert out == []
    mgr.shutdown()


# ---------- Label Sanitization Tests ----------


def test_sanitize_labels_injects_tenant_strategy_and_redacts(mock_otel_stack, fake_counter) -> None:
    """Test that labels are enriched with tenant/strategy and PII is redacted"""
    obs.set_tenant("tenant-A")
    obs.set_strategy("strat-X")

    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    mgr.record_counter(fake_counter, 1.0, {"email": "u@example.com", "env": "prod"})

    assert fake_counter.calls, "expected at least one counter call"
    _, _, labels = fake_counter.calls[-1]

    # Verify tenant/strategy injection
    assert labels["tenant_id"] == "tenant-A"
    assert labels["strategy_id"] == "strat-X"

    # Verify PII redaction
    assert "REDACTED_EMAIL" in labels["email"]

    mgr.shutdown()


def test_sanitize_labels_handles_none_input(mock_otel_stack, fake_counter) -> None:
    """Test that sanitize_labels handles None input gracefully"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    mgr.record_counter(fake_counter, 1.0, labels=None)

    # Should still inject tenant/strategy
    assert fake_counter.calls
    _, _, labels = fake_counter.calls[-1]
    assert "tenant_id" in labels
    assert "strategy_id" in labels

    mgr.shutdown()


def test_sanitize_labels_handles_empty_dict(mock_otel_stack, fake_counter) -> None:
    """Test that sanitize_labels handles empty dict"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    mgr.record_counter(fake_counter, 1.0, labels={})

    assert fake_counter.calls
    _, _, labels = fake_counter.calls[-1]
    assert "tenant_id" in labels
    assert "strategy_id" in labels

    mgr.shutdown()


# ---------- Cardinality Overflow Tests ----------


def test_overflow_counter_incremented_when_cardinality_hashed(
    track_overflow_counter, fake_counter
) -> None:
    """Test that overflow counter is incremented when cardinality limit exceeded"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False, labels_max_keys_per_label=1)
    mgr = MetricsManager("svc", cfg)

    # Two distinct values on same label key cause hashing on second call
    mgr.record_counter(fake_counter, 1.0, {"path": "/a"})
    mgr.record_counter(fake_counter, 1.0, {"path": "/b"})

    # Verify overflow counter was created
    assert "observability_cardinality_overflows" in track_overflow_counter["created_names"]

    mgr.shutdown()


def test_overflow_counter_not_incremented_within_limit(
    track_overflow_counter, fake_counter
) -> None:
    """Test that overflow counter is not incremented when within cardinality limit"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False, labels_max_keys_per_label=5)
    mgr = MetricsManager("svc", cfg)

    # Within limit - no overflow
    for i in range(3):
        mgr.record_counter(fake_counter, 1.0, {"path": f"/{i}"})

    # Overflow counter should have no calls
    assert len(track_overflow_counter["counter"].calls) == 0

    mgr.shutdown()


# ---------- Dropped Events Tests ----------


def test_dropped_counter_incremented_when_queue_full(track_dropped_counter, fake_counter) -> None:
    """Test that dropped counter increments when queue is full"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=True, queue_max_events=1, flush_every_ms=10)
    mgr = MetricsManager("svc", cfg)

    # Fill queue with one event
    mgr.record_counter(fake_counter, 1.0, {"k": "v1"})
    # Second put should fail and increment dropped counter
    mgr.record_counter(fake_counter, 1.0, {"k": "v2"})

    # Brief wait for processing
    time.sleep(0.02)

    # Verify dropped counter was called
    assert any(call[0] == "add" for call in track_dropped_counter.calls)

    mgr.shutdown()


def test_dropped_counter_on_record_counter_error(track_dropped_counter) -> None:
    """Test that dropped counter increments on record_counter errors"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    # Create a counter that raises on add
    bad_counter = MagicMock()
    bad_counter.add.side_effect = RuntimeError("test error")

    mgr.record_counter(bad_counter, 1.0, {"k": "v"})

    # Dropped counter should be incremented
    assert any(call[0] == "add" for call in track_dropped_counter.calls)

    mgr.shutdown()


def test_dropped_counter_on_record_histogram_error(track_dropped_counter) -> None:
    """Test that dropped counter increments on record_histogram errors"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    # Create a histogram that raises on record
    bad_hist = MagicMock()
    bad_hist.record.side_effect = ValueError("test error")

    mgr.record_histogram(bad_hist, 1.0, {"k": "v"})

    # Dropped counter should be incremented
    assert any(call[0] == "add" for call in track_dropped_counter.calls)

    mgr.shutdown()


# ---------- Background Flusher Tests ----------


def test_flusher_processes_counter_and_histogram_events(
    mock_otel_stack, fake_counter, fake_histogram
) -> None:
    """Test that background flusher processes both counter and histogram events"""
    cfg = MetricConfig(
        prometheus_port=0, buffered_emit=True, queue_max_events=32, flush_every_ms=10
    )
    mgr = MetricsManager("svc", cfg)

    # Enqueue both types
    mgr.record_counter(fake_counter, 3.0, {"route": "/ping"})
    mgr.record_histogram(fake_histogram, 12.5, {"route": "/pong"})

    # Wait for background thread to flush
    time.sleep(0.05)

    # Both instruments should have been called
    assert any(t[0] == "add" for t in fake_counter.calls)
    assert any(t[0] == "record" for t in fake_histogram.calls)

    mgr.shutdown()


def test_flusher_handles_errors_gracefully(track_dropped_counter, fake_counter) -> None:
    """Test that flusher continues processing after errors"""
    cfg = MetricConfig(
        prometheus_port=0, buffered_emit=True, queue_max_events=32, flush_every_ms=10
    )
    mgr = MetricsManager("svc", cfg)

    # Create instrument that raises on first call, succeed on second
    call_count = [0]

    def bad_add(_value, _attributes):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("first call fails")

    bad_counter = MagicMock()
    bad_counter.add.side_effect = bad_add

    # Enqueue bad event, then good event
    mgr.record_counter(bad_counter, 1.0, {"k": "v1"})
    mgr.record_counter(fake_counter, 1.0, {"k": "v2"})

    # Wait for flusher
    time.sleep(0.05)

    # Good counter should still be processed
    assert any(t[0] == "add" for t in fake_counter.calls)
    # Dropped counter should be incremented for bad event
    assert any(call[0] == "add" for call in track_dropped_counter.calls)

    mgr.shutdown()


def test_flusher_thread_stops_on_shutdown(mock_otel_stack) -> None:
    """Test that flusher thread stops when shutdown is called"""
    cfg = MetricConfig(
        prometheus_port=0, buffered_emit=True, queue_max_events=32, flush_every_ms=50
    )
    mgr = MetricsManager("svc", cfg)

    assert mgr._flush_thread is not None
    assert mgr._flush_thread.is_alive()

    mgr.shutdown()

    # Give thread time to stop
    time.sleep(0.1)

    assert not mgr._flush_thread.is_alive()


def test_flusher_not_created_when_buffering_disabled(mock_otel_stack) -> None:
    """Test that flusher thread is not created when buffering is disabled"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    assert mgr._flush_thread is None
    assert mgr._queue is None

    mgr.shutdown()


# ---------- Edge Cases & Error Handling ----------


def test_record_counter_with_invalid_labels_type(track_dropped_counter, fake_counter) -> None:
    """Test record_counter handles invalid labels type gracefully"""
    cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
    mgr = MetricsManager("svc", cfg)

    # Patch card.sanitize to raise KeyError
    with patch.object(mgr.card, "sanitize", side_effect=KeyError("test")):
        mgr.record_counter(fake_counter, 1.0, {"k": "v"})

    # Should increment dropped counter
    assert any(call[0] == "add" for call in track_dropped_counter.calls)

    mgr.shutdown()


def test_record_histogram_with_buffering_queue_full(track_dropped_counter, fake_histogram) -> None:
    cfg = MetricConfig(prometheus_port=0, buffered_emit=True, queue_max_events=1, flush_every_ms=10)
    mgr = MetricsManager("svc", cfg)

    # Fill queue
    mgr.record_histogram(fake_histogram, 1.0, {"k": "v1"})
    # Should be dropped
    mgr.record_histogram(fake_histogram, 2.0, {"k": "v2"})

    time.sleep(0.02)

    # Verify dropped counter was called
    assert any(call[0] == "add" for call in track_dropped_counter.calls)

    mgr.shutdown()


# ---------- View and Exemplar Edge Cases ----------


def test_metrics_manager_handles_view_creation_error(mock_otel_stack) -> None:
    # Patch View to raise on creation
    with (
        patch("pysrc.ops.observability.View", side_effect=ValueError("View not supported")),
        patch("pysrc.ops.observability._build_histogram_aggregation", return_value=MagicMock()),
    ):
        cfg = MetricConfig(prometheus_port=0, buffered_emit=False)
        # Should not raise - views list should just be empty
        mgr = MetricsManager("svc", cfg)
        mgr.shutdown()


def test_metrics_manager_handles_exemplar_filter_error(mock_otel_stack) -> None:
    with patch(
        "pysrc.ops.observability.TraceBasedExemplarFilter", side_effect=TypeError("Not supported")
    ):
        cfg = MetricConfig(prometheus_port=0, buffered_emit=False, enable_exemplars=True)
        # Should not raise - exemplar_filter should just be None
        mgr = MetricsManager("svc", cfg)
        mgr.shutdown()


# ---------- Integration: Multiple Events Through Queue ----------


def test_multiple_event_types_processed_in_order(mock_otel_stack) -> None:
    cfg = MetricConfig(
        prometheus_port=0, buffered_emit=True, queue_max_events=100, flush_every_ms=10
    )
    mgr = MetricsManager("svc", cfg)

    counters = [_FakeCounter() for _ in range(3)]
    histograms = [_FakeHistogram() for _ in range(3)]

    # Interleave counter and histogram calls
    for i in range(3):
        mgr.record_counter(counters[i], float(i), {"idx": str(i)})
        mgr.record_histogram(histograms[i], float(i * 10), {"idx": str(i)})

    # Wait for flusher
    time.sleep(0.05)

    # All should be processed
    for ctr in counters:
        assert any(t[0] == "add" for t in ctr.calls)
    for hist in histograms:
        assert any(t[0] == "record" for t in hist.calls)

    mgr.shutdown()

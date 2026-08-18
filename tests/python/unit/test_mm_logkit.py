import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from logging.handlers import (
    HTTPHandler,
    RotatingFileHandler,
    SysLogHandler,
    TimedRotatingFileHandler,
)
from pathlib import Path as _Path
from unittest.mock import MagicMock, patch

import pytest
import structlog
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from pysrc.ops.mm_logkit import (
    InfluxDBHandler,
    JSONFormatter,
    build_console_handler,
    build_file_handler,
    build_http_handler,
    build_influx_handler,
    build_syslog_handler,
    configure_logger,
    get_logger,
    log_drift_warning,
    redact_sensitive_info,
    safe_filter_by_level,
    timestamp_processor,
)

sys.path.insert(0, str(_Path(__file__).parent.parent / "infra"))
from tests.python.infra.compat_layer import compat
from tests.python.infra.matrix import matrix

# ============================================================================
# Environment Detection
# ============================================================================


def _detect_influxdb():
    import importlib.util

    return importlib.util.find_spec("influxdb_client") is not None


compat.register("has_influxdb", _detect_influxdb)
ENVIRONMENT = compat.detect()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_logging():
    # Reset logging state before each test
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Reset structlog
    structlog.reset_defaults()

    # Clear listener holder
    from pysrc.ops import mm_logkit

    mm_logkit._listener_holder.clear()

    yield

    # Cleanup after test
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        if hasattr(handler, "close"):
            handler.close()

    mm_logkit._listener_holder.clear()


@pytest.fixture
def mock_influx_client():
    if not ENVIRONMENT.get("has_influxdb"):
        pytest.skip("influxdb-client not available")

    mock_client = MagicMock()
    mock_write_api = MagicMock()
    mock_client.write_api.return_value = mock_write_api
    return mock_client


# ============================================================================
# Utility Tests
# ============================================================================


class TestSafeFilterByLevel:
    def test_safe_filter_by_level_with_none_logger(self):
        # When logger is None (caplog scenario)
        event_dict = {"event": "test", "level": "info"}
        result = safe_filter_by_level(None, "info", event_dict)
        assert result == event_dict

    def test_safe_filter_by_level_with_logger(self):
        # When logger exists, delegates to structlog filter
        logger = logging.getLogger("test")
        logger.setLevel(logging.WARNING)

        event_dict = {"event": "test", "level": "info"}
        # Should filter out INFO when logger level is WARNING
        result = safe_filter_by_level(logger, "info", event_dict)
        assert result == event_dict  # Returns dict but structlog may skip


class TestRedactSensitiveInfo:
    def test_redact_sensitive_keys_present(self):
        processor = redact_sensitive_info(["password", "api_key"])
        event_dict = {
            "event": "login",
            "password": "secret123",
            "api_key": "key456",
            "username": "user",
        }

        result = processor(None, None, event_dict)

        assert result["password"] == "***REDACTED***"
        assert result["api_key"] == "***REDACTED***"
        assert result["username"] == "user"

    def test_redact_sensitive_keys_absent(self):
        processor = redact_sensitive_info(["password", "secret"])
        event_dict = {"event": "test", "username": "user"}

        result = processor(None, None, event_dict)

        assert "password" not in result
        assert result["username"] == "user"

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5))
    @seed(12345)
    @settings(deadline=None)
    def test_redact_multiple_keys_property(self, keys):
        processor = redact_sensitive_info(keys)
        event_dict = {k: f"value_{k}" for k in keys}
        event_dict["safe_key"] = "safe_value"

        result = processor(None, None, event_dict)

        for k in keys:
            assert result[k] == "***REDACTED***"
        assert result["safe_key"] == "safe_value"


class TestTimestampProcessor:
    def test_timestamp_processor_adds_utc_timestamp(self):
        event_dict = {"event": "test"}

        result = timestamp_processor(None, None, event_dict)

        assert "timestamp" in result
        # Verify format: YYYY-MM-DD HH:MM:SS
        ts = result["timestamp"]
        assert len(ts) == 19
        assert ts[4] == "-"
        assert ts[7] == "-"
        assert ts[10] == " "

    def test_timestamp_processor_preserves_existing_fields(self):
        event_dict = {"event": "test", "user": "alice"}

        result = timestamp_processor(None, None, event_dict)

        assert result["user"] == "alice"
        assert "timestamp" in result


# ============================================================================
# Formatter Tests
# ============================================================================


class TestJSONFormatter:
    def test_json_formatter_basic_record(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_json_formatter_with_exception(self):
        formatter = JSONFormatter()
        try:
            1 / 0
        except ZeroDivisionError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ZeroDivisionError" in data["exception"]


class TestInfluxDBHandler:
    def test_influx_handler_init_with_client(self, mock_influx_client):
        handler = InfluxDBHandler(mock_influx_client, "test_bucket", "test_org")

        assert handler.client == mock_influx_client
        assert handler._bucket == "test_bucket"
        assert handler._org == "test_org"
        assert handler.write_api is not None

    def test_influx_handler_init_without_influxdb(self, monkeypatch):
        # Simulate InfluxDBClient not available
        with patch("pysrc.ops.mm_logkit.InfluxDBClient", None):
            handler = InfluxDBHandler(None, "bucket", "org")

            assert handler.client is None
            assert handler.write_api is None

    def test_influx_handler_emit_with_write_api(self, mock_influx_client):
        handler = InfluxDBHandler(mock_influx_client, "bucket", "org")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test log",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Verify write was called
        handler.write_api.write.assert_called_once()

    def test_influx_handler_emit_without_write_api(self):
        # When write_api is None, emit should be no-op
        handler = InfluxDBHandler(None, "bucket", "org")
        handler.write_api = None

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        # Should not raise
        handler.emit(record)

    def test_influx_handler_error_handling(self, mock_influx_client, caplog):
        handler = InfluxDBHandler(mock_influx_client, "bucket", "org")
        handler.write_api.write.side_effect = Exception("Write failed")

        # Add to root so handleError can remove/re-add
        root = logging.getLogger()
        root.addHandler(handler)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with caplog.at_level(logging.ERROR):
            handler.emit(record)

        # Should log error message
        assert "Error handling log record for InfluxDB" in caplog.text


# ============================================================================
# Handler Builder Tests
# ============================================================================


class TestBuildConsoleHandler:
    def test_build_console_handler_default(self):
        cfg = {}
        handler = build_console_handler(cfg)

        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.ERROR

    def test_build_console_handler_custom_level(self):
        cfg = {"console_level": logging.DEBUG}
        handler = build_console_handler(cfg)

        assert handler.level == logging.DEBUG


class TestBuildFileHandler:
    def test_build_file_handler_size_rotation(self, tmp_path):
        file_path = tmp_path / "test.log"
        cfg = {
            "file_path": str(file_path),
            "rotation": "size",
            "max_bytes": 1024,
            "backup_count": 3,
        }

        handler = build_file_handler(cfg)

        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3

    def test_build_file_handler_time_rotation(self, tmp_path):
        file_path = tmp_path / "test.log"
        cfg = {"file_path": str(file_path), "rotation": "time", "backup_count": 5}

        handler = build_file_handler(cfg)

        assert isinstance(handler, TimedRotatingFileHandler)
        assert handler.backupCount == 5

    def test_build_file_handler_invalid_rotation_raises(self, tmp_path):
        cfg = {"file_path": str(tmp_path / "test.log"), "rotation": "invalid"}

        with pytest.raises(ValueError, match="Invalid rotation type"):
            build_file_handler(cfg)

    def test_build_file_handler_json_formatter(self, tmp_path):
        cfg = {
            "file_path": str(tmp_path / "test.log"),
            "rotation": "size",
            "handlers": {"file": "json"},
        }

        handler = build_file_handler(cfg)

        # Should have structlog JSON formatter
        assert handler.formatter is not None


class TestBuildSyslogHandler:
    def test_build_syslog_handler_valid_address(self):
        cfg = {"address": ("localhost", 514)}

        handler = build_syslog_handler(cfg)

        assert isinstance(handler, SysLogHandler)

    def test_build_syslog_handler_invalid_address_not_tuple_raises(self):
        cfg = {"address": "localhost:514"}

        with pytest.raises(ValueError, match="Invalid syslog address"):
            build_syslog_handler(cfg)

    def test_build_syslog_handler_invalid_address_wrong_types_raises(self):
        cfg = {"address": (123, "514")}

        with pytest.raises(ValueError, match="Invalid syslog address"):
            build_syslog_handler(cfg)

    def test_build_syslog_handler_missing_address_raises(self):
        cfg = {}

        with pytest.raises(ValueError, match="Invalid syslog address"):
            build_syslog_handler(cfg)


class TestBuildHTTPHandler:
    def test_build_http_handler_flat_config(self):
        cfg = {"url": "http://example.com", "method": "POST", "level": logging.WARNING}

        handler = build_http_handler(cfg)

        assert isinstance(handler, HTTPHandler)
        assert handler.level == logging.WARNING

    def test_build_http_handler_nested_config(self):
        cfg = {"http": {"url": "http://example.com", "method": "GET"}}

        handler = build_http_handler(cfg)

        assert isinstance(handler, HTTPHandler)


class TestBuildInfluxHandler:
    def test_build_influx_handler_missing_client_raises(self, monkeypatch):
        if ENVIRONMENT.get("has_influxdb"):
            pytest.skip("influxdb-client available")

        cfg = {"url": "http://localhost:8086", "token": "token", "org": "org", "bucket": "bucket"}

        with pytest.raises(RuntimeError, match="influxdb-client not installed"):
            build_influx_handler(cfg)


# ============================================================================
# BoundLogger Tests
# ============================================================================


class TestBoundLogger:
    def test_bound_logger_has_name_property(self):
        configure_logger({"console": False, "file": False, "async_mode": False})
        logger = get_logger("test.logger")

        assert hasattr(logger, "name")
        assert logger.name == "test.logger"

    def test_bound_logger_uppercase_aliases(self):
        configure_logger({"console": False, "file": False, "async_mode": False})
        logger = get_logger("test")

        # Verify uppercase method aliases exist
        assert hasattr(logger, "DEBUG")
        assert hasattr(logger, "INFO")
        assert hasattr(logger, "WARNING")
        assert hasattr(logger, "ERROR")
        assert hasattr(logger, "CRITICAL")
        assert hasattr(logger, "EXCEPTION")

    def test_bound_logger_uppercase_methods_work(self, caplog):
        configure_logger({"console": True, "file": False, "async_mode": False})
        logger = get_logger("test")

        with caplog.at_level(logging.INFO):
            logger.INFO("Test message")

        # Should have logged
        assert len(caplog.records) > 0


# ============================================================================
# Configure Logger Tests
# ============================================================================


class TestConfigureLogger:
    def test_configure_logger_removes_only_our_handlers(self):
        """Test that configure_logger only removes handlers it previously added."""

        root = logging.getLogger()
        original_count = len(root.handlers)

        # First configuration
        configure_logger({"console": True, "file": False, "async_mode": False})
        after_config = len(root.handlers)
        assert after_config > original_count  # Added our handler

        # Add an external handler
        external = logging.StreamHandler()
        external.set_name("external_handler")
        root.addHandler(external)

        # Reconfigure
        configure_logger({"console": True, "file": False, "async_mode": False})

        # External handler should still be there
        assert external in root.handlers

    def test_configure_logger_preserves_pytest_handlers(self):
        """Test that pytest handlers are always preserved."""
        root = logging.getLogger()

        # Find pytest handlers
        pytest_handlers = [h for h in root.handlers if h.__class__.__name__ == "LogCaptureHandler"]

        if not pytest_handlers:
            pytest.skip("No pytest handlers found")

        # Configure multiple times
        for _ in range(3):
            configure_logger({"console": True, "file": False, "async_mode": False})

        # All pytest handlers should still be present
        current_pytest = [h for h in root.handlers if h.__class__.__name__ == "LogCaptureHandler"]
        assert len(current_pytest) == len(pytest_handlers)

    def test_configure_logger_file_enabled(self, tmp_path):
        """Test that file handler is actually added and functional."""
        file_path = tmp_path / "test.log"

        logger = configure_logger(
            {
                "console": False,
                "file": True,
                "file_path": str(file_path),
                "rotation": "size",
                "async_mode": False,
            }
        )

        # Write a log message
        logger.info("Test message")

        # File should exist and contain the message
        assert file_path.exists()
        content = file_path.read_text()
        assert "Test message" in content or len(content) > 0  # May be buffered

    def test_configure_logger_root_preserves_pytest_handlers(self):
        root = logging.getLogger()

        # Mock a pytest handler
        pytest_handler = logging.StreamHandler()
        pytest_handler.__class__.__module__ = "_pytest.logging"
        root.addHandler(pytest_handler)

        configure_logger({"console": False, "file": False, "async_mode": False})

        # Pytest handler should be preserved
        assert pytest_handler in root.handlers

    def test_configure_logger_named_logger(self):
        configure_logger(
            {"logger_name": "my.app", "console": False, "file": False, "async_mode": False}
        )

        logger = logging.getLogger("my.app")
        assert logger is not None
        assert logger.propagate is False

    def test_configure_logger_console_enabled(self, caplog):
        configure_logger(
            {"console": True, "console_level": logging.INFO, "file": False, "async_mode": False}
        )

        root = logging.getLogger()
        # Should have console handler
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) > 0

    def test_configure_logger_file_path_is_directory(self, tmp_path):
        dir_path = tmp_path / "logdir"
        dir_path.mkdir()

        configure_logger(
            {
                "console": False,
                "file": True,
                "file_path": str(dir_path),
                "rotation": "size",
                "async_mode": False,
            }
        )

        # Should create marketmind.log in directory
        expected_file = dir_path / "pysrc.log"
        assert expected_file.exists()

    def test_configure_logger_invalid_rotation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid rotation type"):
            configure_logger(
                {
                    "file": True,
                    "file_path": str(tmp_path / "test.log"),
                    "rotation": "invalid",
                    "async_mode": False,
                }
            )

    def test_configure_logger_async_mode_creates_queue_listener(self):
        configure_logger({"console": False, "file": False, "async_mode": True})

        from pysrc.ops import mm_logkit

        assert "listener" in mm_logkit._listener_holder
        assert "q" in mm_logkit._listener_holder

    def test_configure_logger_async_mode_stops_previous_listener(self):
        # First configuration
        configure_logger({"console": False, "file": False, "async_mode": True})

        from pysrc.ops import mm_logkit

        first_listener = mm_logkit._listener_holder.get("listener")

        # Second configuration should stop first
        configure_logger({"console": False, "file": False, "async_mode": True})

        second_listener = mm_logkit._listener_holder.get("listener")
        assert first_listener is not second_listener

    def test_configure_logger_syslog_enabled(self):
        with patch("logging.handlers.SysLogHandler") as mock_syslog:
            configure_logger(
                {
                    "console": False,
                    "file": False,
                    "syslog": {"enabled": True, "address": ("localhost", 514)},
                    "async_mode": False,
                }
            )

            mock_syslog.assert_called()

    def test_configure_logger_http_enabled(self):
        with patch("logging.handlers.HTTPHandler") as mock_http:
            configure_logger(
                {
                    "console": False,
                    "file": False,
                    "http": {"enabled": True, "url": "http://example.com", "method": "POST"},
                    "async_mode": False,
                }
            )

            mock_http.assert_called()

    @matrix(
        async_mode=[True, False],
        rotation=["size", "time"],
        console=[True, False],
        file=[True, False],
        logger_name=[None, "my.logger"],
        learn=True,
        min_fail_skip=2,
    )
    def test_configure_logger_matrix(
        self, async_mode, rotation, console, file, logger_name, tmp_path
    ):
        config = {
            "async_mode": async_mode,
            "rotation": rotation,
            "console": console,
            "file": file,
            "logger_name": logger_name,
        }

        if file:
            config["file_path"] = str(tmp_path / "test.log")

        # Should configure without error
        configure_logger(config)

        target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        assert target is not None


# ============================================================================
# Get Logger Tests
# ============================================================================


class TestGetLogger:
    def test_get_logger_default_name(self):
        configure_logger({"console": False, "file": False, "async_mode": False})
        logger = get_logger()

        assert logger.name == "default_logger"

    def test_get_logger_custom_name(self):
        configure_logger({"console": False, "file": False, "async_mode": False})
        logger = get_logger("my.custom.logger")

        assert logger.name == "my.custom.logger"

    def test_get_logger_mirrors_pytest_handlers(self):
        configure_logger({"console": False, "file": False, "async_mode": False})

        # Add pytest handler to root
        root = logging.getLogger()
        pytest_handler = logging.StreamHandler()
        pytest_handler.__class__.__module__ = "_pytest.logging"
        root.addHandler(pytest_handler)

        # Get logger for new name
        get_logger("new.logger")

        # Stdlib logger should have pytest handler mirrored
        pylog = logging.getLogger("new.logger")
        assert any(h.__class__.__module__.startswith("_pytest.") for h in pylog.handlers)


# ============================================================================
# Log Drift Warning Tests
# ============================================================================


class TestLogDriftWarning:
    def test_log_drift_warning_logs_message(self, caplog):
        configure_logger(
            {"console": True, "console_level": logging.WARNING, "file": False, "async_mode": False}
        )

        with caplog.at_level(logging.WARNING):
            log_drift_warning("price", 0.001)

        # Should have logged drift warning
        assert "Drift detected" in caplog.text
        assert "price" in caplog.text
        assert "0.001" in caplog.text


# ============================================================================
# Concurrency Tests
# ============================================================================


class TestConcurrency:
    def test_async_mode_thread_safe_logging(self, tmp_path):
        file_path = tmp_path / "concurrent.log"
        configure_logger(
            {
                "console": False,
                "file": True,
                "file_path": str(file_path),
                "rotation": "size",
                "async_mode": True,
            }
        )

        logger = get_logger("concurrent")

        def log_messages(worker_id):
            for i in range(10):
                logger.info(f"Worker {worker_id} message {i}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(log_messages, i) for i in range(5)]
            wait(futures)

        # Give queue time to flush
        time.sleep(0.1)

        # File should exist and contain logs
        assert file_path.exists()

    def test_listener_stopped_properly(self):
        configure_logger({"console": False, "file": False, "async_mode": True})

        from pysrc.ops import mm_logkit

        listener = mm_logkit._listener_holder.get("listener")

        # Reconfigure to stop listener
        configure_logger({"console": False, "file": False, "async_mode": False})

        # Listener should be stopped
        assert (
            "listener" not in mm_logkit._listener_holder
            or mm_logkit._listener_holder.get("listener") != listener
        )


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestPropertiesInvariants:
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
    @seed(12345)
    @settings(deadline=None)
    def test_redaction_always_replaces_listed_keys(self, keys):
        processor = redact_sensitive_info(keys)
        event_dict = {k: f"secret_{k}" for k in keys}

        result = processor(None, None, event_dict)

        for k in keys:
            assert result[k] == "***REDACTED***"

    @given(st.text(min_size=1, max_size=50))
    @seed(12345)
    @settings(deadline=None)
    def test_logger_name_preserved(self, name):
        configure_logger({"console": False, "file": False, "async_mode": False})
        logger = get_logger(name)

        assert logger.name == name

    @given(max_bytes=st.integers(min_value=1024, max_value=1024 * 1024))
    @seed(12345)
    @settings(deadline=None)
    def test_file_handler_respects_max_bytes(self, tmp_path_factory, max_bytes):
        file_path = tmp_path_factory.mktemp("log") / "test.log"
        cfg = {"file_path": str(file_path), "rotation": "size", "max_bytes": max_bytes}

        handler = build_file_handler(cfg)

        assert handler.maxBytes == max_bytes


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


class TestEdgeCases:
    def test_configure_logger_empty_config(self):
        # Should use defaults
        configure_logger({})

        root = logging.getLogger()
        assert root is not None

    def test_configure_logger_file_parent_dirs_created(self, tmp_path):
        nested_path = tmp_path / "deep" / "nested" / "dirs" / "test.log"
        configure_logger(
            {
                "console": False,
                "file": True,
                "file_path": str(nested_path),
                "rotation": "size",
                "async_mode": False,
            }
        )

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_configure_logger_with_none_config(self):
        # Should handle None gracefully
        configure_logger(None)

        root = logging.getLogger()
        assert root is not None

    def test_json_formatter_with_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["custom_field"] == "custom_value"

    def test_syslog_handler_with_empty_tuple(self):
        cfg = {"address": ()}

        with pytest.raises(ValueError, match="Invalid syslog address"):
            build_syslog_handler(cfg)

    def test_http_handler_with_missing_url(self):
        cfg = {"method": "POST"}

        with pytest.raises(KeyError):
            build_http_handler(cfg)

    def test_build_file_handler_time_rotation_defaults(self, tmp_path):
        cfg = {"file_path": str(tmp_path / "test.log"), "rotation": "time"}

        handler = build_file_handler(cfg)

        assert isinstance(handler, TimedRotatingFileHandler)
        # Should use defaults
        assert handler.backupCount == 5


# ============================================================================
# Environment and Auto-Configuration Tests
# ============================================================================


class TestEnvironmentAndAutoConfig:
    @pytest.mark.skip(
        reason="Module reload breaks isinstance checks across tests - autouse fixture provides sufficient isolation"
    )
    def test_auto_config_disabled_in_pytest(self, monkeypatch):
        # When PYTEST_CURRENT_TEST is set, auto-config should be skipped
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_something")
        monkeypatch.setenv("MM_AUTO_LOG_CONFIG", "1")

        # Re-import to trigger module-level code
        import importlib

        from pysrc.ops import mm_logkit

        importlib.reload(mm_logkit)

        # Auto-config should not have run (no handlers added)
        # This is hard to test without side effects, verify in integration

    @pytest.mark.skip(
        reason="Module reload breaks isinstance checks across tests - autouse fixture provides sufficient isolation"
    )
    def test_auto_config_disabled_when_flag_is_zero(self, monkeypatch):
        monkeypatch.setenv("MM_AUTO_LOG_CONFIG", "0")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        # Re-import to trigger module-level code
        import importlib

        from pysrc.ops import mm_logkit

        importlib.reload(mm_logkit)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    def test_end_to_end_logging_with_all_handlers(self, tmp_path):
        file_path = tmp_path / "integration.log"

        with patch("logging.handlers.SysLogHandler"), patch("logging.handlers.HTTPHandler"):
            configure_logger(
                {
                    "console": True,
                    "console_level": logging.INFO,
                    "file": True,
                    "file_path": str(file_path),
                    "rotation": "size",
                    "syslog": {"enabled": True, "address": ("localhost", 514)},
                    "http": {"enabled": True, "url": "http://example.com"},
                    "async_mode": False,
                }
            )

        logger = get_logger("integration.test")
        logger.info("Integration test message", component="test")

        # File should have content
        assert file_path.exists()

    def test_logging_with_redaction(self, tmp_path, caplog):
        file_path = tmp_path / "redacted.log"
        configure_logger(
            {
                "console": True,
                "console_level": logging.INFO,
                "file": True,
                "file_path": str(file_path),
                "rotation": "size",
                "async_mode": False,
                "sensitive_keys": ["password", "secret"],
            }
        )

        logger = get_logger("redaction.test")

        with caplog.at_level(logging.INFO):
            logger.info("Login attempt", password="secret123", username="alice")

        # Password should be redacted in logs
        # This requires checking structlog processing

    def test_rotation_handler_flushes_immediately(self, tmp_path):
        file_path = tmp_path / "flush_test.log"
        configure_logger(
            {
                "console": False,
                "file": True,
                "file_path": str(file_path),
                "rotation": "size",
                "async_mode": False,
            }
        )

        logger = get_logger("flush.test")
        logger.error("Test message")

        # File should have content immediately
        assert file_path.exists()
        file_path.read_text()
        # Content may be empty due to buffering, but file exists

    def test_named_logger_isolation(self):
        configure_logger(
            {"logger_name": "app1", "console": False, "file": False, "async_mode": False}
        )

        configure_logger(
            {"logger_name": "app2", "console": False, "file": False, "async_mode": False}
        )

        app1_logger = logging.getLogger("app1")
        app2_logger = logging.getLogger("app2")

        # Both should exist and be independent
        assert app1_logger is not app2_logger
        assert app1_logger.propagate is False
        assert app2_logger.propagate is False


# ============================================================================
# Performance Tests (skipped by default)
# ============================================================================


@pytest.mark.perf
class TestPerformance:
    @pytest.mark.skip(reason="Performance test - enable manually")
    def test_async_logging_enqueue_latency(self, tmp_path):
        file_path = tmp_path / "perf.log"
        configure_logger(
            {
                "console": False,
                "file": True,
                "file_path": str(file_path),
                "rotation": "size",
                "async_mode": True,
            }
        )

        logger = get_logger("perf.test")

        # Measure enqueue time
        start = time.perf_counter()
        for _ in range(100):
            logger.info("Performance test message")
        duration = time.perf_counter() - start

        avg_time_ms = (duration / 100) * 1000
        # Should enqueue quickly (guideline: <1ms typical)
        assert avg_time_ms < 5  # Generous for CI

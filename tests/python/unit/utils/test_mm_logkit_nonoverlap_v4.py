# tests/unit/test_mm_logkit_nonoverlap_v4.py

import importlib
import logging
import logging.handlers
import sys
from unittest.mock import MagicMock, patch

import pytest

# Prefer pysrc.ops.mm_logkit if present; fall back to top-level mm_logkit for local dev
from pysrc.ops import mm_logkit as mml

# -----------------------------------------------------------------------------
# Import-time + basic infra
# -----------------------------------------------------------------------------


class TestImportAndBasics:
    def test_existing_logging_queue_not_overridden(self, monkeypatch):
        # Force logging.Queue to already exist before reload. This should take
        # the "already defined, do nothing" branch at module top rather than
        # installing queue.Queue again (L19 -> skip setattr). Architectural goal:
        # prove idempotence when embedding this logger into hosts that already
        # patched logging.
        sentinel = object()
        monkeypatch.setattr(logging, "Queue", sentinel, raising=False)

        importlib.reload(mml)

        assert logging.Queue is sentinel

    def test_coerce_level_int_passthrough(self):
        # _coerce_level should return the integer unchanged when already numeric
        # (branch L75 returning early rather than mapping string -> logging.X)
        assert mml._coerce_level(15) == 15

    def test_maybe_attach_pytest_handlers_no_pytest(self, monkeypatch):
        # Force _is_pytest() -> False so _maybe_attach_pytest_handlers is a no-op
        # (branch L86 immediate return). This exercises the "not running
        # under pytest capture" behavior for library embedding.
        monkeypatch.setattr(mml, "_is_pytest", lambda: False)

        lg = logging.getLogger("no.pytest.attach")
        before = list(lg.handlers)
        mml._maybe_attach_pytest_handlers(lg)
        after = list(lg.handlers)

        assert before == after


# -----------------------------------------------------------------------------
# Redaction filter branches not hit elsewhere
# -----------------------------------------------------------------------------


class TestRedactFilterAdditionalBranches:
    def test_filter_on_mapping_message_scrubs_keys(self):
        # Message is a Mapping -> path L124-L127: copy, redact in-place
        rec = logging.LogRecord(
            "map.redact",
            logging.INFO,
            "",
            0,
            {"password": "secret123", "ok": "yes"},
            (),
            None,
        )
        f = mml.RedactFilter()
        assert f.filter(rec) is True
        assert isinstance(rec.msg, dict)
        assert rec.msg["password"] == "***REDACTED***"
        assert rec.msg["ok"] == "yes"

    def test_filter_on_nonstring_message_only_scrubs_extras(self):
        # Message is neither Mapping nor str -> branch L128 skips both first arms
        # Extras on the record itself still get scrubbed (L131-L134)
        rec = logging.LogRecord(
            "num.redact",
            logging.INFO,
            "",
            0,
            12345,  # non-str, non-mapping message
            (),
            None,
        )
        rec.token = "abc123"  # sensitive default key
        f = mml.RedactFilter()
        f.filter(rec)

        assert rec.msg == 12345
        assert rec.token == "***REDACTED***"


# -----------------------------------------------------------------------------
# KVFormatter gaps
# -----------------------------------------------------------------------------


class TestKVFormatterGaps:
    def test_skip_empty_msg_field(self):
        # Branch L218-L220: if msg is falsy, formatter should NOT append msg=...
        fmt = mml.KVFormatter()
        rec = logging.LogRecord(
            "kv.skipmsg",
            logging.INFO,
            "",
            0,
            "",  # empty string -> falsy
            (),
            None,
        )
        out = fmt.format(rec)
        assert "msg=" not in out

    def test_custom_attrs_and_exc_info_included(self):
        # Branches:
        # - L221-L227: non-reserved attrs get serialized
        # - L225-L227: exc_info path emits exc=... blob
        fmt = mml.KVFormatter()

        try:
            raise ValueError("boom")
        except ValueError:
            rec = logging.LogRecord(
                "kv.custom",
                logging.ERROR,
                "",
                0,
                "hello",
                (),
                sys.exc_info(),  # exc_info triggers exc=...
            )

        # attach a custom field that's not reserved
        rec.user = "alice"

        out = fmt.format(rec)

        # msg serialized
        assert 'msg="hello"' in out
        # custom attr serialized
        assert 'user="alice"' in out
        # exc_info serialized
        assert "exc=" in out
        assert "ValueError" in out


# -----------------------------------------------------------------------------
# BoundLogger edge paths not covered elsewhere
# -----------------------------------------------------------------------------


class TestBoundLoggerEdgeCases:
    def test_emit_invalid_extra_logs_debug_and_drops_it(self):
        # We force an invalid 'extra' (string instead of dict) to drive the
        # except path at L276-L283, which logs a debug message instead of
        # exploding, and then proceeds.
        pylog = MagicMock(spec=logging.Logger)
        pylog.name = "bound.invalid.extra"

        bl = mml.BoundLogger(pylog)

        bl.info("msg", extra="notadict", custom="v")

        # The debug() diagnostic path should have been called
        assert pylog.debug.call_count >= 1
        dbg_args, dbg_kwargs = pylog.debug.call_args
        assert "Failed to merge extra context" in dbg_args[0]

        # The actual info() call should still have happened
        assert pylog.info.call_count >= 1

    def test_uppercase_aliases_and_exception_helper(self):
        # Exercise uppercase aliases (L306-L311) and .exception() logic (L301-L304)
        pylog = MagicMock(spec=logging.Logger)
        pylog.name = "bound.aliases"

        bl = mml.BoundLogger(pylog)

        bl.DEBUG("d")
        bl.INFO("i")
        bl.WARNING("w")
        bl.ERROR("e")
        bl.CRITICAL("c")
        bl.EXCEPTION("boom")  # should set exc_info=True and route to error()

        assert pylog.debug.called
        assert pylog.info.called
        assert pylog.warning.called
        assert pylog.error.called
        assert pylog.critical.called
        # last call to error() (from EXCEPTION) should have exc_info True
        err_args, err_kwargs = pylog.error.call_args
        assert err_args[0] == "boom"
        assert err_kwargs.get("exc_info") is True


# -----------------------------------------------------------------------------
# _ensure_async_for: reuse existing state, no duplicate QueueListener start
# -----------------------------------------------------------------------------


class TestEnsureAsyncReuseState:
    def test_subsequent_calls_skip_listener_restart(self):
        # First call should create state, start listener, mark started True.
        # Second call with same name should reuse that state without calling
        # QueueListener() again and without re-start logic. This covers:
        # - branch where state already exists (L352 skip creation)
        # - branch where state.started is True so we skip L357-L363
        with patch("logging.handlers.QueueListener") as ql_cls:
            inst = MagicMock()
            # happy-path start() (no RuntimeError) so started=True branch
            inst.start.return_value = None
            ql_cls.return_value = inst

            qh1 = mml._ensure_async_for("reuse.state", [logging.StreamHandler()])
            assert isinstance(qh1, logging.handlers.QueueHandler)
            assert ql_cls.call_count == 1

            qh2 = mml._ensure_async_for("reuse.state", [logging.StreamHandler()])
            assert isinstance(qh2, logging.handlers.QueueHandler)
            # still only constructed once
            assert ql_cls.call_count == 1


# -----------------------------------------------------------------------------
# Handler factories: console/file/influx
# -----------------------------------------------------------------------------


class TestConsoleAndFileFactories:
    def test_console_handler_uses_json_formatter_when_requested(self):
        # cfg["handlers"]["console"] == "json" should install JSONFormatter
        h = mml.build_console_handler({"handlers": {"console": "json"}})
        assert isinstance(h.formatter, mml.JSONFormatter)

    def test_console_handler_swallows_bad_handlers_shape(self):
        # handlers is non-dict -> AttributeError inside build_console_handler()
        # should be swallowed and still return a working StreamHandler
        h = mml.build_console_handler({"handlers": None})
        assert isinstance(h, logging.StreamHandler)

    def test_build_file_handler_requires_path(self):
        # When file_path missing, builder raises (branch L386)
        with pytest.raises(ValueError, match="file_path is required"):
            mml.build_file_handler({})

    def test_build_influx_handler_dependency_gate_and_client(self, monkeypatch):
        # First: no InfluxDBClient and no client -> RuntimeError (L444-L445)
        monkeypatch.setattr(mml, "InfluxDBClient", None, raising=False)
        with pytest.raises(RuntimeError):
            mml.build_influx_handler({"bucket": "b", "org": "o"})

        # Second: provide explicit client so branch succeeds without the dep
        class DummyClient:
            def write_api(self):
                return MagicMock()

        dummy = DummyClient()
        h = mml.build_influx_handler({"client": dummy, "bucket": "b", "org": "o"})
        assert isinstance(h, mml.InfluxDBHandler)
        assert h.client is dummy


# -----------------------------------------------------------------------------
# Lower-level handler builders (_make_std_handler, rotating/timed file)
# -----------------------------------------------------------------------------


class TestStdAndFileHandlerBranches:
    def test_make_std_handler_sets_level_when_provided(self):
        h = mml._make_std_handler({"level": "WARNING"})
        assert isinstance(h, logging.StreamHandler)
        assert h.level == logging.WARNING

    def test_make_rotating_file_handler_no_filename_returns_none(self):
        assert mml._make_rotating_file_handler({}) is None

    def test_make_rotating_file_handler_sets_level_and_filter(self, tmp_path):
        path = tmp_path / "rot.log"
        spec = {"filename": str(path), "level": "DEBUG"}
        h = mml._make_rotating_file_handler(spec)

        assert isinstance(h, logging.handlers.RotatingFileHandler)
        assert h.level == logging.DEBUG
        # RedactFilter should be attached for downstream safety
        assert any(isinstance(f, mml.RedactFilter) for f in h.filters)

    def test_make_timed_rotating_file_handler_no_filename_returns_none(self):
        assert mml._make_timed_rotating_file_handler({}) is None

    def test_make_timed_rotating_file_handler_sets_level_and_filter(self, tmp_path):
        path = tmp_path / "time.log"
        spec = {"filename": str(path), "level": "WARNING"}
        h = mml._make_timed_rotating_file_handler(spec)

        assert isinstance(h, logging.handlers.TimedRotatingFileHandler)
        assert h.level == logging.WARNING
        # RedactFilter should be attached here as well
        assert any(isinstance(f, mml.RedactFilter) for f in h.filters)


# -----------------------------------------------------------------------------
# Syslog / HTTP handler factories: untouched branches
# -----------------------------------------------------------------------------
class TestSyslogAndHttpFactories:
    def test_syslog_handler_default_address_real_class(self, monkeypatch):
        # Force the "is type" branch but raise OSError so factory returns None and avoids sockets
        class DummySysLogHandler:
            LOG_USER = 1
            USER = 1

            def __init__(self, *a, **k):
                raise OSError("no syslog socket available")

        monkeypatch.setattr(logging.handlers, "SysLogHandler", DummySysLogHandler)
        h = mml._make_syslog_handler({"address": "/dev/log"})
        assert h is None

    def test_syslog_handler_instance_without_return_value(self, monkeypatch):
        # Not a type; rv is None; calling raises -> returns None
        class DummyNoRV:
            def __init__(self):
                self.return_value = None

            def __call__(self, *a, **k):
                raise TypeError("boom")

        monkeypatch.setattr(logging.handlers, "SysLogHandler", DummyNoRV())
        h = mml._make_syslog_handler({"address": ("localhost", 514)})
        assert h is None

    def test_syslog_handler_without_level_skips_manual_setting(self):
        # Patch with MagicMock (not a type) and omit "level" so setLevel isn't called
        with patch("logging.handlers.SysLogHandler") as syslog_cls:
            inst = syslog_cls.return_value
            h = mml._make_syslog_handler({"address": ("localhost", 514)})
            assert h is inst
            assert inst.setFormatter.called
            inst.setLevel.assert_not_called()

    def test_http_handler_no_host_returns_none(self):
        # Fast exit when host missing
        assert mml._make_http_handler({}) is None

    def test_http_handler_real_ctor_sets_level_and_is_instance(self):
        # Real class branch: instance created, formatter set, level coerced
        h = mml._make_http_handler(
            {"host": "example.com", "url": "/x", "method": "post", "level": "WARNING"}
        )
        assert isinstance(h, logging.handlers.HTTPHandler)
        assert h.level == logging.WARNING

    def test_http_handler_instance_without_return_value(self, monkeypatch):
        # Not a type; rv is None; calling raises -> returns None
        class DummyNoRV:
            def __init__(self):
                self.return_value = None

            def __call__(self, *a, **k):
                raise TypeError("bad ctor")

        monkeypatch.setattr(logging.handlers, "HTTPHandler", DummyNoRV())
        h = mml._make_http_handler({"host": "example.com"})
        assert h is None

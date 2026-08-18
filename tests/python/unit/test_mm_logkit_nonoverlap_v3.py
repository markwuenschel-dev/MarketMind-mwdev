# tests/unit/test_mm_logkit_nonoverlap_v3.py
import json
import logging
import logging.handlers
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Prefer pysrc.ops.mm_logkit if present; fall back to top-level mm_logkit
try:
    from pysrc.ops import mm_logkit as mml
except ModuleNotFoundError:
    import mm_logkit as mml


# -----------------------------
# JSONFormatter deep conversions
# -----------------------------
class TestJSONFormatterDeepPaths:
    def test_json_formatter_memoryview_and_nonstring_keys(self):
        fmt = mml.JSONFormatter()
        rec = logging.LogRecord("t", logging.INFO, "", 0, "msg", (), None)
        rec.blob = memoryview(b"\xffabc")
        rec.mapping = {1: "one", "two": 2}
        out = fmt.format(rec)
        data = json.loads(out)
        assert "extra" in data
        # memoryview coerced to utf-8 string (with replacement)
        assert isinstance(data["extra"]["blob"], str)
        # non-string keys stringified inside nested mapping
        assert data["extra"]["mapping"]["1"] == "one"
        assert data["extra"]["mapping"]["two"] == 2

    def test_json_formatter_bytearray_and_object_repr(self):
        fmt = mml.JSONFormatter()
        rec = logging.LogRecord("t", logging.INFO, "", 0, "hi", (), None)
        rec.raw = bytearray(b"xyz")
        rec.obj = SimpleNamespace(a=1)  # coerced via repr()
        out = fmt.format(rec)
        data = json.loads(out)
        assert data["extra"]["raw"] == "xyz"
        assert isinstance(data["extra"]["obj"], str)  # repr string


# -----------------------------------------
# _fmt_from_spec additional alias & datefmt
# -----------------------------------------
class TestFmtFromSpecAliases:
    def test_application_json_alias(self):
        f = mml._fmt_from_spec({"kind": "application/json", "datefmt": "%Y"})
        assert isinstance(f, mml.JSONFormatter)
        assert f.datefmt == "%Y"


# ---------------------------------------------------------
# configure_logger with handlers=None mirrors pytest handler
# ---------------------------------------------------------
class TestConfigureLoggerHandlersNonePath:
    def test_mirrors_pytest_handlers_when_none(self):
        root = logging.getLogger()
        root.handlers[:] = []
        ph = logging.StreamHandler()
        ph.__class__.__module__ = "_pytest.logging"
        root.addHandler(ph)

        # handlers=None triggers mirror path
        bl = mml.configure_logger("mirror.none", handlers=None)
        assert isinstance(bl, mml.BoundLogger)
        pylog = logging.getLogger("mirror.none")
        assert any(h.__class__.__module__.startswith("_pytest.") for h in pylog.handlers)


# ---------------------------------------------
# _teardown_async_for error swallowing branches
# ---------------------------------------------
class DummyListener:
    def __init__(self, to_raise):
        self.to_raise = to_raise
        self.stopped = False

    def stop(self):
        self.stopped = True
        raise self.to_raise


class TestTeardownAsyncFor:
    def test_runtimeerror_during_stop_is_swallowed_and_state_cleared(self):
        name = "async.stop.runtime"
        mml._ASYNC[name] = mml._AsyncState(
            q=logging.Queue(-1), listener=DummyListener(RuntimeError()), started=True
        )  # type: ignore[attr-defined]
        mml._teardown_async_for(name)
        assert name not in mml._ASYNC

    def test_valueerror_during_stop_is_swallowed_and_state_cleared(self):
        name = "async.stop.value"
        mml._ASYNC[name] = mml._AsyncState(
            q=logging.Queue(-1), listener=DummyListener(ValueError()), started=True
        )  # type: ignore[attr-defined]
        mml._teardown_async_for(name)
        assert name not in mml._ASYNC


# ----------------------------------------------------
# _ensure_async_for: start() raises RuntimeError path
# ----------------------------------------------------
class TestEnsureAsyncStartErrorPath:
    def test_start_runtimeerror_marks_started_true(self):
        with patch("logging.handlers.QueueListener") as ql_cls:
            inst = MagicMock()
            # First construction, start raises RuntimeError
            inst.start.side_effect = RuntimeError("boom")
            ql_cls.return_value = inst
            qh = mml._ensure_async_for("async.start.error", [logging.StreamHandler()])
            assert isinstance(qh, logging.handlers.QueueHandler)
            # state is recorded and marked started True
            state = mml._ASYNC["async.start.error"]
            assert state.started is True


# ------------------------------------------------------
# get_logger: structlog wrap failure falls back to stdlib
# ------------------------------------------------------
class TestGetLoggerStructlogFallback:
    def test_wrap_failure_returns_boundlogger(self, monkeypatch):
        # Pretend structlog is available but wrap_logger fails
        monkeypatch.setattr(mml, "HAVE_STRUCTLOG", True)

        class FakeStructlog:
            def wrap_logger(self, *a, **k):
                raise TypeError("nope")

        monkeypatch.setattr(mml, "structlog", FakeStructlog())
        bl = mml.get_logger("wrap.fail")
        assert isinstance(bl, mml.BoundLogger)


# ----------------------------------------
# _make_http_handler defaults & level path
# ----------------------------------------
class TestMakeHttpHandlerDetails:
    def test_default_method_and_level_string(self):
        with patch("logging.handlers.HTTPHandler") as http_cls:
            http_cls.return_value = logging.handlers.HTTPHandler("example.com", "/")
            h = mml._make_http_handler({"host": "example.com", "level": "WARNING"})
            assert isinstance(h, logging.handlers.HTTPHandler)
            assert h.level == logging.WARNING


# ---------------------------------------
# _make_syslog_handler level setting path
# ---------------------------------------
class TestMakeSyslogHandlerDetails:
    def test_level_string_is_applied(self):
        with patch("logging.handlers.SysLogHandler") as syslog_cls:
            syslog_cls.return_value = logging.handlers.SysLogHandler(address=("localhost", 514))
            h = mml._make_syslog_handler({"address": ("localhost", 514), "level": "ERROR"})
            assert isinstance(h, logging.handlers.SysLogHandler)
            assert h.level == logging.ERROR


# ---------------------------------------------------------
# BoundLogger._emit: empty 'extra' becomes None (pass-through)
# ---------------------------------------------------------
class TestBoundLoggerEmptyExtraPassThrough:
    def test_empty_extra_passed_as_none(self):
        pylog = MagicMock(spec=logging.Logger)
        pylog.name = "x"
        bl = mml.BoundLogger(pylog)
        bl.info("m", extra={})  # empty extra triggers None
        # First positional arg is message; ensure kwargs has extra=None
        assert pylog.info.call_args.kwargs.get("extra", None) is None


# ------------------------------------------
# RedactFilter: case-insensitive extra scrub
# ------------------------------------------
class TestRedactFilterExtrasCaseInsensitive:
    def test_authorization_header_is_redacted(self):
        f = mml.RedactFilter()
        rec = logging.LogRecord("t", logging.INFO, "", 0, "ok", (), None)
        rec.Authorization = "Bearer 123"  # mixed case
        f.filter(rec)
        assert rec.Authorization == "***REDACTED***"

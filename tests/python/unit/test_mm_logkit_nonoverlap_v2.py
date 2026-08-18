import json
import logging
import logging.handlers
from unittest.mock import patch

# Prefer pysrc.ops.mm_logkit if present; fall back to top-level mm_logkit for local dev
try:
    from pysrc.ops import mm_logkit as mml
except ModuleNotFoundError:
    import mm_logkit as mml


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def count_pytest_handlers(logger: logging.Logger) -> int:
    return sum(
        1 for h in logger.handlers if getattr(type(h), "__module__", "").startswith("_pytest.")
    )


# -----------------------------------------------------------------------------
# FmtFromSpec: datefmt wiring and invalid kind fallback
# -----------------------------------------------------------------------------


class TestFmtFromSpecGaps:
    def test_fmt_from_spec_sets_datefmt_on_kv(self):
        fmt = mml._fmt_from_spec({"kind": "kv", "datefmt": "%Y"})
        assert isinstance(fmt, mml.KVFormatter)
        # KVFormatter inherits logging.Formatter; verify datefmt propagation
        assert fmt.datefmt == "%Y"

    def test_fmt_from_spec_invalid_kind_falls_back(self):
        fmt = mml._fmt_from_spec({"kind": "not-a-kind"})
        assert isinstance(fmt, logging.Formatter)
        assert not isinstance(fmt, (mml.KVFormatter, mml.JSONFormatter))


# -----------------------------------------------------------------------------
# _make_std_handler: formatter selection via spec.kind
# -----------------------------------------------------------------------------


class TestStdHandlerKindSelection:
    def test_std_handler_uses_json_formatter(self):
        h = mml._make_std_handler({"kind": "json"})
        assert isinstance(h.formatter, mml.JSONFormatter)

    def test_std_handler_uses_kv_formatter(self):
        h = mml._make_std_handler({"kind": "kv"})
        assert isinstance(h.formatter, mml.KVFormatter)


# -----------------------------------------------------------------------------
# JSONFormatter robustness for non-serializable extras and unicode/control chars
# -----------------------------------------------------------------------------


class TestJSONFormatterRobustness:
    def test_json_formatter_handles_non_serializable_extras(self):
        fmt = mml.JSONFormatter()
        rec = logging.LogRecord("t", logging.INFO, "", 0, "hello", (), None)
        rec.custom_obj = object()
        rec.a_set = {"x", "y"}
        out = fmt.format(rec)
        data = json.loads(out)
        # Either extras are coerced to string or omitted; both acceptable
        extra = data.get("extra", {})
        if extra:
            assert isinstance(extra.get("custom_obj"), str) or extra.get("custom_obj") is None
            # sets are turned into a list or string by most serializers; just assert json compatibility
            assert isinstance(extra.get("a_set"), (list, str)) or extra.get("a_set") is None

    def test_json_formatter_unicode_and_control_chars(self):
        fmt = mml.JSONFormatter()
        msg = "snowman: ☃ newline:\n tab:\t emoji: 😺"
        rec = logging.LogRecord("t", logging.INFO, "", 0, msg, (), None)
        out = fmt.format(rec)
        data = json.loads(out)
        assert "snowman" in data["message"]
        # ensure the message round-trips and contains markers
        assert "emoji" in data["message"]


# -----------------------------------------------------------------------------
# _build_handlers: RedactFilter presence on built handlers
# -----------------------------------------------------------------------------


class TestBuildHandlersFilters:
    def test_redact_filter_attached_to_each_supported_handler(self, tmp_path):
        specs = [
            {"type": "stream"},
            {"type": "rotating_file", "filename": str(tmp_path / "a.log")},
            {"type": "timed_rotating_file", "filename": str(tmp_path / "b.log")},
        ]
        hs = mml._build_handlers(specs)
        assert len(hs) == 3
        for h in hs:
            assert any(isinstance(f, mml.RedactFilter) for f in h.filters), (
                f"No RedactFilter on {h}"
            )


# -----------------------------------------------------------------------------
# Optional deps: happy-path via factory stubs (no overlap with "not available" tests)
# -----------------------------------------------------------------------------


class TestOptionalDepsHappyPath:
    def test_watchtower_factory_used_when_available(self, monkeypatch):
        monkeypatch.setattr(mml, "HAVE_WATCHTOWER", True)
        dummy_handler = logging.StreamHandler()
        with patch.object(mml, "_make_watchtower_handler", return_value=dummy_handler) as mk:
            mml.configure_logger("opt.watch", handlers=[{"type": "watchtower"}])
            pylog = logging.getLogger("opt.watch")
            assert dummy_handler in pylog.handlers
            mk.assert_called_once()

    def test_gcloud_factory_used_when_available(self, monkeypatch):
        monkeypatch.setattr(mml, "HAVE_GCLOUD", True)
        dummy_handler = logging.StreamHandler()
        with patch.object(mml, "_make_gcloud_handler", return_value=dummy_handler) as mk:
            mml.configure_logger("opt.gcloud", handlers=[{"type": "gcloud"}])
            pylog = logging.getLogger("opt.gcloud")
            assert dummy_handler in pylog.handlers
            mk.assert_called_once()

    def test_influx_factory_used_when_available(self, monkeypatch):
        monkeypatch.setattr(mml, "HAVE_INFLUX", True)
        dummy_handler = logging.StreamHandler()
        with patch.object(mml, "_make_influx_handler", return_value=dummy_handler) as mk:
            mml.configure_logger("opt.influx", handlers=[{"type": "influxdb"}])
            pylog = logging.getLogger("opt.influx")
            assert dummy_handler in pylog.handlers
            mk.assert_called_once()


# -----------------------------------------------------------------------------
# Pytest handler mirroring: idempotence (no dupes on repeated calls)
# -----------------------------------------------------------------------------


class TestPytestMirroringIdempotence:
    def test_get_logger_does_not_duplicate_pytest_handlers(self):
        name = "idemp.mirror"
        # First call
        mml.get_logger(name)
        pylog = logging.getLogger(name)
        before = count_pytest_handlers(pylog)
        # Second call should not add more pytest handlers
        mml.get_logger(name)
        after = count_pytest_handlers(pylog)
        assert after == before


# -----------------------------------------------------------------------------
# Async teardown: switching from async=True to async=False stops and clears state
# -----------------------------------------------------------------------------


class TestAsyncTeardown:
    def test_reconfigure_disables_async_and_clears_queue_state(self):
        name = "async.teardown"
        mml.configure_logger(name, handlers=[{"type": "stream"}], async_mode=True)
        assert name in mml._ASYNC
        pylog = logging.getLogger(name)
        assert any(isinstance(h, logging.handlers.QueueHandler) for h in pylog.handlers)

        # Reconfigure to sync mode
        mml.configure_logger(name, handlers=[{"type": "stream"}], async_mode=False)
        # State cleared
        assert name not in mml._ASYNC
        pylog2 = logging.getLogger(name)
        assert all(not isinstance(h, logging.handlers.QueueHandler) for h in pylog2.handlers)


# -----------------------------------------------------------------------------
# QueueListener arg contract: respect_handler_level=True
# -----------------------------------------------------------------------------


class TestQueueListenerArgs:
    def test_queue_listener_called_with_respect_handler_level(self):
        with patch("logging.handlers.QueueListener") as ql_cls:
            h = logging.StreamHandler()
            qh = mml._ensure_async_for("ql.args", [h])
            # The constructor should have been called with respect_handler_level=True
            args, kwargs = ql_cls.call_args
            assert kwargs.get("respect_handler_level") is True
            assert isinstance(qh, logging.handlers.QueueHandler)

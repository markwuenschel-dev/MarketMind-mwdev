from __future__ import annotations

import builtins
import contextlib
import glob
import os
import shutil
import socket as _socket
import sys
import types
from types import SimpleNamespace

import pytest

from tests.python.infra.compat_adapter import attach_caps
from tests.python.infra.self_evolving_engine import create_intelligent_test_infrastructure

# Plugins: seeds, data_fixtures, hardware; optional stats_gates (DSR/minTRL helpers)
pytest_plugins = [
    "tests.python._plugins.seeds",
    "tests.python._plugins.data_fixtures",
    "tests.python._plugins.hardware",
    "tests.python._plugins.stats_gates",
]


# --- Kept from original: Keeps the repo root clean. ---
@pytest.fixture(scope="session", autouse=True)
def _contain_run_manifests(request):
    """
    Keep the repo root clean: move any manifest_<hash>.json into .pytest_cache/manifests
    at session end. This plays nice with xdist (workers may emit files concurrently).
    """
    target_dir = os.environ.get("MANIFEST_DIR", ".pytest_cache/manifests")
    os.makedirs(target_dir, exist_ok=True)

    def _sweep():
        try:
            for path in glob.glob("manifest_*.json"):
                with contextlib.suppress(FileNotFoundError, PermissionError, shutil.Error, OSError):
                    shutil.move(path, os.path.join(target_dir, os.path.basename(path)))
        except (OSError, FileNotFoundError):
            pass

    request.addfinalizer(_sweep)


# --- sample_features, small_prices_df, prices_small_*, df_prices, etc. live in tests.python._plugins.data_fixtures ---


# --- Fixed from original: Ensures `builtins.run` is correctly exposed. ---
@pytest.fixture(scope="session", autouse=True)
def _install_legacy_shims(intelligent_stack):
    learning_engine, processing_engine, _caps = intelligent_stack

    def _run(df, spec_or_cfg, *, backend="polars", optimize=False):
        spec = spec_or_cfg.get("spec_inline", spec_or_cfg) if isinstance(spec_or_cfg, dict) else {}
        cfg = SimpleNamespace(
            parallel_enabled=True,
            force_sequential=(backend == "pandas" and not optimize),
            max_workers=(os.cpu_count() or 4),
            parallel_executor="thread",
        )
        return processing_engine.process(df, spec, cfg)

    builtins.run = _run

    mod = types.ModuleType("tests.python.infra.helper_shims")

    def build_plan(steps, order, *, policy="fail_fast"):
        return {"steps": list(steps), "order": order or {}, "policy": policy}

    mod.run = _run
    mod.build_plan = build_plan
    for pkg in ("tests", "tests.python", "tests.python.infra"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    sys.modules["tests.python.infra.helper_shims"] = mod
    return


def pytest_addoption(parser):
    parser.addoption("--perf", action="store_true", help="run perf/scale tests")
    parser.addoption(
        "--matrix", action="store", default="", help="filter key=value[,key=value...] for matrix()"
    )


def _ensure_hypothesis_plugin(pm) -> None:
    """Register Hypothesis' pytest plugin before collection, but only if not already loaded."""
    try:
        import hypothesis  # noqa: F401
    except ImportError:
        return  # Hypothesis not installed; nothing to do

    # If the plugin is already present under any known name, do nothing.
    known_names = (
        "_hypothesis_pytestplugin",  # canonical
        "hypothesispytest",
        "hypothesisplugin",
        "hypothesis",
    )
    if any(pm.hasplugin(n) for n in known_names):
        return

    # Also scan loaded plugin modules by module name to be extra safe.
    for plug in pm.get_plugins():
        modname = getattr(plug, "__name__", "")
        if "hypothesis" in modname:
            return

    # Try canonical plugin import first; fall back to the public alias.
    try:
        pm.import_plugin("_hypothesis_pytestplugin")
    except (ImportError, ValueError, RuntimeError):
        try:
            pm.import_plugin("hypothesis.extra.pytestplugin")
        except (ImportError, ValueError, RuntimeError):
            # If Hypothesis isn't installed or plugin registration fails,
            # do nothing; collection will proceed (tests that require
            # Hypothesis will fail explicitly rather than mis-collecting).
            pass


def pytest_load_initial_conftests(early_config, parser, args):
    """
    Ensure Hypothesis' pytest plugin is loaded *before* any tests are imported/collected.
    Without this, @given(...) parameters like 'key'/'value' are mis-read as fixtures
    and collection errors ('fixture "key" not found') occur.
    """
    _ensure_hypothesis_plugin(early_config.pluginmanager)


def pytest_configure(config):
    # Preserve existing behavior, e.g., matrix env
    try:
        val = config.getoption("--matrix")
        if val:
            os.environ.setdefault("PYTEST_MATRIX", val)
    except (ValueError, AttributeError):
        pass

    # Idempotent safety: ensure plugin is present even if early hook was skipped
    _ensure_hypothesis_plugin(config.pluginmanager)


# --- Fixed from updated: Conditional benchmark with correct implementation. ---
try:
    import importlib.util

    if importlib.util.find_spec("pytest_benchmark.plugin") is not None:
        import pytest_benchmark.plugin  # noqa: F401
except (ModuleNotFoundError, ImportError):

    @pytest.fixture
    def benchmark():
        def _bench(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _bench


@pytest.fixture
def preproc_spec_robust():
    return {"features": [{"op": "zscore", "in": "price", "out": "price_robust", "window": 5}]}


# --- caps, _compat_detect_once, _compat_meminfo_shim live in tests.python._plugins.hardware ---


@pytest.fixture(scope="session")
def intelligent_stack(caps):
    learning_engine, processing_engine = create_intelligent_test_infrastructure()
    attach_caps(learning_engine, caps)
    return learning_engine, processing_engine, caps


# --- Block network by default. Loopback stays allowed for local asyncio/socketpair use. ---
@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    if "net" in request.keywords:
        return

    def _is_loopback(address) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        host = address[0]
        return host in {"127.0.0.1", "::1", "localhost"}

    class _BlockedSocket(_socket.socket):
        def connect(self, *a, **k):
            if a and _is_loopback(a[0]):
                return super().connect(*a, **k)
            raise RuntimeError("Network access blocked (use @pytest.mark.net)")

    monkeypatch.setattr(_socket, "socket", _BlockedSocket)


# --- Determinism: per-test seeds from tests.python._plugins.seeds (deterministic_seed). ---
def _apply_determinism_and_hypothesis_profiles():
    os.environ.setdefault("PYTHONHASHSEED", "42")
    try:
        from hypothesis import settings as _Hsettings

        _Hsettings.register_profile("ci", _Hsettings(max_examples=300, deadline=500))
        _Hsettings.register_profile("dev", _Hsettings(max_examples=2000, deadline=None))
        _Hsettings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
    except (ModuleNotFoundError, ImportError):
        pass


@pytest.fixture(scope="session", autouse=True)
def _determinism_profiles_session():
    _apply_determinism_and_hypothesis_profiles()
    return


# --- New feature from updated: Reliable ephemeral port allocator. ---
def allocate_port():
    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

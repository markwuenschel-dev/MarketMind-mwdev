"""
Hardware and compat detection: caps, _compat_detect_once, device/skip logic.

Extracted from conftest; used by tests that depend on compat (e.g. meminfo, backend).
"""

from __future__ import annotations

import inspect as _inspect
import os
from typing import Any

import pytest

from tests.python.infra.compat_layer import compat

try:
    from pysrc.pipeline import orchestrator as dpo
except (ModuleNotFoundError, ImportError):
    dpo = None


def _register_default_probes() -> None:
    """Register default compat probes (e.g. meminfo) if dpo is available."""
    if dpo is None:
        return
    try:
        fn = getattr(dpo, "_maybe_mem_info", None)
        if callable(fn):
            compat.register(
                "meminfo_wants_self",
                lambda: "self" in _inspect.signature(fn).parameters,
            )
    except (AttributeError, ValueError, TypeError):
        pass


_register_default_probes()


@pytest.fixture(scope="session", autouse=True)
def _compat_detect_once(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Run compat detection once per session; cache and set COMPAT_* env vars."""
    results = compat.detect()
    cache = getattr(request.config, "cache", None)
    if cache is not None:
        cache.set("compat/results", results)
    for k, v in results.items():
        val = (
            str(bool(v)).upper()
            if not (isinstance(v, str) and v.startswith("__ERROR__"))
            else "ERROR"
        )
        os.environ[f"COMPAT_{k.upper()}"] = val
    return results


@pytest.fixture(scope="session")
def caps(request: pytest.FixtureRequest, _compat_detect_once: dict[str, Any]) -> dict[str, Any]:
    """Session-scoped compat detection results (cached)."""
    cache = getattr(request.config, "cache", None)
    if cache is not None:
        cached = cache.get("compat/results", None)
        if cached is not None:
            return cached
    return _compat_detect_once


@pytest.fixture(autouse=True)
def _compat_meminfo_shim(monkeypatch: pytest.MonkeyPatch, caps: dict[str, Any]) -> None:
    """Shim dpo._maybe_mem_info when caps indicate it expects 'self' (avoids signature errors)."""
    if dpo is None:
        yield
        return
    wants_self = caps.get("meminfo_wants_self")
    if wants_self is True or (isinstance(wants_self, str) and wants_self.upper() == "TRUE"):
        monkeypatch.setattr(dpo, "_maybe_mem_info", lambda *a, **k: {}, raising=False)
    yield

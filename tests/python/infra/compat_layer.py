# tests/python/infra/compat_layer.py
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging  # For debug logging
import os
import platform  # For env fingerprint
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as TMO
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout  # For process-safe learn()

# --- Module Logger ---
logger = logging.getLogger("compat_layer")

# --- pipeline_config knobs (env-overridable) ---
_MAX_PROBE_THREADS = int(os.getenv("COMPAT_THREADS", "8"))
_PROBE_TIMEOUT_S = float(os.getenv("COMPAT_PROBE_TIMEOUT_S", "1.5"))
_CACHE_TTL_S = int(os.getenv("PYTEST_COMPAT_TTL_S", "7200"))
_CACHE_FILE = Path(os.getenv("PYTEST_COMPAT_CACHE", ".pytest_cache/compat_detect.json"))
_LEARN_FILE = Path(os.getenv("PYTEST_COMPAT_LEARN", ".pytest_cache/compat_learn.json"))

# Global upper bound for the entire detect() run.
# This prevents worst-case total wall time from growing as O(num_probes * per_probe_timeout).
_GLOBAL_DETECT_TIMEOUT_S = float(os.getenv("COMPAT_GLOBAL_TIMEOUT_S", "10.0"))


# --- infra helpers ---
def _code_id(obj: Any) -> str:
    # Hash the source of the probe (or its repr() fallback) so cache auto-invalidates
    try:
        src = inspect.getsource(obj if inspect.isfunction(obj) else obj.__class__)
    except Exception:
        src = repr(obj)
    h = hashlib.sha256(src.encode("utf-8", "ignore")).hexdigest()[:16]
    return h


def _env_fingerprint() -> dict[str, str]:
    # Capture runtime conditions that affect probe answers.
    # If these change, we don't trust the cache.
    return {
        "py": platform.python_version(),
        "plat": platform.platform(),
        "cuda_visible": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        # Add more knobs here if probes depend on them
    }


def _atomic_write(path: Path, text: str) -> None:
    # Atomic replace so other processes never see a half-written file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _read_cache(current_env_fp: dict[str, str]) -> dict[str, Any]:
    # Return cached payload if:
    # - cache file exists
    # - within TTL
    # - env fingerprint matches
    try:
        if _CACHE_FILE.exists() and (time.time() - _CACHE_FILE.stat().st_mtime) <= _CACHE_TTL_S:
            cached_data = json.loads(_CACHE_FILE.read_text())
            if cached_data.get("env_fp") == current_env_fp:
                return cached_data
            else:
                logger.debug("compat_layer cache miss: env_fp mismatch")
        else:
            if _CACHE_FILE.exists():
                logger.debug("compat_layer cache miss: TTL expired")
    except Exception as e:
        logger.debug("compat_layer _read_cache failed", exc_info=e)
        pass
    return {}


def _write_cache(payload: dict[str, Any]) -> None:
    # Best-effort write; never raise in test infra
    try:
        _atomic_write(_CACHE_FILE, json.dumps(payload, indent=2))
    except Exception as e:
        logger.debug("compat_layer _write_cache failed", exc_info=e)
        pass


@contextlib.contextmanager
def patched_attr(mod, name, value):
    # Temporarily patch a module attribute, then restore
    prev = getattr(mod, name, None)
    had = hasattr(mod, name)
    try:
        setattr(mod, name, value)
        yield
    finally:
        try:
            if had:
                setattr(mod, name, prev)
            else:
                delattr(mod, name)
        except Exception:
            # Cleanup failure shouldn't bring down tests
            logger.debug("compat_layer patched_attr cleanup failed", exc_info=True)
            pass


@contextlib.contextmanager
def tmp_file_ctx(name: str, data: bytes):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / name
        p.write_bytes(data)
        yield p


# --- public API ---
class CompatLayer:
    def __init__(self):
        self._probes: dict[str, Callable[[], Any]] = {}
        self._learn_file = _LEARN_FILE  # configurable via PYTEST_COMPAT_LEARN

    def register(self, name: str, fn: Callable[[], Any]) -> None:
        self._probes[name] = fn

    def detect(self) -> dict[str, Any]:
        # Build cache keys from both probe code and runtime environment
        code_fps = {n: _code_id(f) for n, f in self._probes.items()}
        current_env_fp = _env_fingerprint()

        cache = _read_cache(current_env_fp)

        # Fast path: cache hit
        if cache.get("code_fps") == code_fps and "results" in cache:
            logger.debug("compat_layer cache hit")
            return cache["results"]

        # Cache miss => run probes
        logger.debug("compat_layer cache miss: running probes")

        results: dict[str, Any] = {}
        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=_MAX_PROBE_THREADS) as ex:
            fut_map = {ex.submit(fn): name for name, fn in self._probes.items()}

            # We iterate deterministically in registration order.
            # For each probe:
            #   - per-probe timeout is capped at _PROBE_TIMEOUT_S
            #   - total time is capped at _GLOBAL_DETECT_TIMEOUT_S
            for fut, name in list(fut_map.items()):
                elapsed = time.monotonic() - t0
                remaining_global = _GLOBAL_DETECT_TIMEOUT_S - elapsed

                if remaining_global <= 0:
                    # We've hit the global ceiling. Mark all remaining probes as global timeout.
                    results[name] = "__ERROR__:Timeout(Global)"
                    logger.debug(
                        "compat_layer global timeout hit before probe %s could complete",
                        name,
                    )
                    continue

                # We'll wait at most the smaller of:
                # - per-probe timeout
                # - remaining global budget
                timeout_budget = min(_PROBE_TIMEOUT_S, remaining_global)

                try:
                    results[name] = fut.result(timeout=timeout_budget)
                except TMO:
                    # Individual probe hung or we ran out of remaining_global for this probe
                    results[name] = "__ERROR__:Timeout"
                    logger.debug("compat_layer probe %s timed out", name)
                except Exception as e:
                    # Probe raised
                    results[name] = f"__ERROR__:{type(e).__name__}"
                    logger.debug("compat_layer probe %s failed", name, exc_info=e)

        # Write cache with new data
        _write_cache(
            {
                "code_fps": code_fps,
                "env_fp": current_env_fp,
                "results": results,
                "ts": time.time(),
            }
        )
        return results

    def learn(self, record: dict[str, Any]) -> None:
        # Append a record to the learn file in a process-safe way.
        # We lock the entire read/modify/write transaction so xdist workers
        # can't clobber each other's writes.
        lock_path = self._learn_file.with_suffix(self._learn_file.suffix + ".lock")
        lock = FileLock(lock_path, timeout=5)  # seconds

        try:
            with lock:
                data = []
                if self._learn_file.exists():
                    try:
                        data = json.loads(self._learn_file.read_text())
                        if not isinstance(data, list):
                            data = []
                    except json.JSONDecodeError:
                        logger.debug("compat_layer learn() found corrupted JSON, resetting.")
                        data = []

                data.append(record)

                # Optional retention policy:
                # data = data[-1000:]

                _atomic_write(self._learn_file, json.dumps(data, indent=2))

        except Timeout:
            # Couldn't acquire the file lock in time. Best-effort: skip instead of corrupting.
            logger.debug(
                "compat_layer learn() timed out acquiring file lock for %s",
                self._learn_file,
            )
            pass
        except Exception as e:
            # We never throw in test infra, but we *do* want breadcrumbs.
            logger.debug("compat_layer learn() write failed", exc_info=e)
            pass


# singleton
compat = CompatLayer()

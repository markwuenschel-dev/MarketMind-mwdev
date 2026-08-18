"""MarketMind application package ``py``.

This top-level name matches the on-disk layout (``py/pipeline``, ``py/cli``, …).
Pytest still imports the legacy PyPI ``py`` distribution for ``pysrc.path.local``;
we merge that library's ``path`` (and ``iniconfig`` if present) into this module
when the vendor package is installed (see dev dependency ``py`` in ``pyproject.toml``).
"""

from __future__ import annotations

import os

__all__: list[str] = []

# The old vendor-merge shim mutated sys.modules during import and breaks
# normal imports on Python 3.13. Keep runtime imports inert by default.
_ENABLE_VENDOR_PY_MERGE = os.getenv("MARKETMIND_ENABLE_VENDOR_PY_MERGE", "0") in {
    "1",
    "true",
    "TRUE",
}

if _ENABLE_VENDOR_PY_MERGE:
    raise RuntimeError(
        "Vendor 'py' merge is disabled pending a safe implementation. "
        "Do not enable MARKETMIND_ENABLE_VENDOR_PY_MERGE yet."
    )

""" from __future__ import annotations

import importlib
import sys
from pathlib import Path

__all__: list[str] = []


def _merge_vendor_pylib() -> None:
    Merge PyPI ``py`` (``pysrc.path``, submodules) so pytest's ``pysrc.path.local`` resolves.
    Editable installs put this repo's ``py`` package first on ``sys.path``; pytest still
    expects the legacy ``py`` distribution's submodule graph. We load that distribution
    with the repo root removed from ``sys.path``, then reattach its submodules in
    ``sys.modules`` and copy selected attributes onto this module.

    this = sys.modules.get(__name__)
    if this is None:
        return
    repo_root = Path(__file__).resolve().parent.parent
    try:
        repo_resolved = repo_root.resolve()
    except OSError:
        return

    def _entry_root(entry: str) -> Path:
        return Path(entry).resolve() if entry else Path.cwd().resolve()

    filtered = [p for p in sys.path if _entry_root(p) != repo_resolved]
    if not filtered or filtered == sys.path:
        return
    saved_path = sys.path[:]
    sys.modules.pop(__name__, None)
    snap: dict[str, object] = {}
    try:
        sys.path[:] = filtered
        importlib.import_module(__name__)
        # Eagerly load lazy submodules so ``pysrc._path`` etc. enter ``sys.modules`` for the snapshot.
        for sub in ("path", "_path", "_std", "iniconfig", "_vendored_packages"):
            try:
                importlib.import_module(f"{__name__}.{sub}")
            except ModuleNotFoundError:
                pass
        snap = {k: v for k, v in list(sys.modules.items()) if k == __name__ or k.startswith(f"{__name__}.")}
    except Exception:
        return
    finally:
        sys.path[:] = saved_path
    snap.pop(__name__, None)
    sys.modules[__name__] = this
    for key, mod in snap.items():
        sys.modules[key] = mod
    path_mod = snap.get(f"{__name__}.path")
    if path_mod is not None:
        setattr(this, "path", path_mod)
    ini_mod = snap.get(f"{__name__}.iniconfig")
    if ini_mod is not None:
        setattr(this, "iniconfig", ini_mod)


_merge_vendor_pylib() """

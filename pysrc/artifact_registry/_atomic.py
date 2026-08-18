from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_bytes(target_path: Path, data: bytes, *, fsync: bool = True) -> None:
    """
    Atomically write *data* to *target_path*.

    Strategy:
        - Ensure parent directory exists.
        - Write to a temporary file in the same directory.
        - Flush + fsync the file (if requested).
        - os.replace(temp, target) for atomic swap on POSIX and Windows.
        - Best-effort directory fsync on POSIX; skipped on Windows.
    """
    target = Path(target_path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Use a deterministic prefix to aid debugging; suffix is arbitrary.
    tmp_path = parent / f".{target.name}.tmp"

    # Open temp file explicitly to avoid platform-specific NamedTemporaryFile quirks.
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
    finally:
        # If an exception escapes before fd is wrapped by fdopen, close it.
        with contextlib.suppress(OSError):
            os.close(fd)

    os.replace(tmp_path, target)

    if fsync:
        _fsync_parent_dir(parent)


def atomic_write_json(target_path: Path, obj: Any, *, fsync: bool = True) -> None:
    """
    Atomically write JSON to *target_path* using stable formatting.

    Notes:
        - Uses sort_keys=True for deterministic key ordering.
        - Indented output to remain diff-friendly.
        - For RFC8785 JCS canonicalization, call the gate canonicalizer
          directly and then use atomic_write_bytes instead.
    """
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    atomic_write_bytes(target_path, text.encode("utf-8"), fsync=fsync)


def _fsync_parent_dir(path: Path) -> None:
    """
    Best-effort fsync of a directory.

    On POSIX, use os.open(..., O_DIRECTORY) and fsync the directory fd.
    On Windows, this is a no-op; the OS provides different guarantees and
    there is no portable directory fsync in the stdlib.
    """
    try:
        # O_DIRECTORY is not available on Windows.
        dir_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return

    try:
        os.fsync(dir_fd)
    except OSError:
        # Directory fsync is best-effort; failure should not abort the write.
        pass
    finally:
        with contextlib.suppress(OSError):
            os.close(dir_fd)

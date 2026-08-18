"""Quarantined research-lane tests (excluded from default CI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LANE_PYSRC_ROOTS = (
    _REPO_ROOT / "archive" / "research_lanes" / "ii0c",
    _REPO_ROOT / "archive" / "research_lanes" / "w1",
    _REPO_ROOT / "archive" / "research_lanes" / "mlc",
    _REPO_ROOT / "archive" / "research_lanes" / "rg09",
)


def _prepend_lane_paths() -> None:
    for lane_root in _LANE_PYSRC_ROOTS:
        pysrc_root = lane_root / "pysrc"
        if not pysrc_root.is_dir():
            continue
        token = str(pysrc_root)
        if token not in sys.path:
            sys.path.insert(0, token)


_prepend_lane_paths()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "quarantine" in str(item.path).replace("\\", "/"):
            item.add_marker(pytest.mark.quarantine)

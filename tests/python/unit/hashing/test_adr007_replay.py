from __future__ import annotations

import platform

import pytest

from pysrc.ops.hashing.adr007_replay import (
    ADR007_SUITES,
    case_by_id,
    expected_output_for_case,
    replay_case,
    replay_summary,
)
from tests.python._fixtures.hashing import load_manifest

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d2")]


def _iter_cases() -> list[object]:
    params: list[object] = []
    for suite in ADR007_SUITES:
        manifest = load_manifest(suite)
        for case in manifest.cases:
            params.append(pytest.param(suite, case.case_id, id=f"{suite}:{case.case_id}"))
    return params


@pytest.mark.parametrize(("suite", "case_id"), _iter_cases())
def test_adr007_cases_replay_to_declared_expected_output(suite: str, case_id: str) -> None:
    case = case_by_id(suite, case_id)
    assert replay_case(suite, case) == expected_output_for_case(suite, case)


def test_replay_summary_uses_runtime_python_version() -> None:
    summary = replay_summary()
    expected = ".".join(platform.python_version_tuple()[:2])

    assert summary["language"] == f"python-{expected}"

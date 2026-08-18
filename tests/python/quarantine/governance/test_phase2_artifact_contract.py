"""Quarantined: Phase II artifact contract governance tests (not active research path)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Governance artifact-contract tests quarantined in research-first cutover."
)

# Original module relocated from tests/python/unit/meta/test_phase2_artifact_contract.py
# Run manually only when validating legacy Phase II-0 contract emission.

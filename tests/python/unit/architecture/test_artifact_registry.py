"""Artifact registry single-source tests."""

from __future__ import annotations

import pytest

from pysrc.pipeline.meta_router_products import (
    CANONICAL_ARTIFACTS,
    LEGACY_FLAT_ARTIFACTS,
    artifact_spec,
    durable_filenames,
)


@pytest.mark.determinism("d1")
def test_every_legacy_key_resolves_to_spec() -> None:
    for key in LEGACY_FLAT_ARTIFACTS:
        spec = artifact_spec(key)
        assert spec.filename == LEGACY_FLAT_ARTIFACTS[key].filename


@pytest.mark.determinism("d1")
def test_canonical_products_have_schema_or_legacy_link() -> None:
    for spec in CANONICAL_ARTIFACTS.values():
        assert spec.schema_ref or spec.legacy_key or spec.durability == "durable"


@pytest.mark.determinism("d1")
def test_durable_allowlist_derived_from_registry() -> None:
    expected = {
        spec.filename for spec in CANONICAL_ARTIFACTS.values() if spec.durability == "durable"
    }
    assert expected.issubset(durable_filenames())

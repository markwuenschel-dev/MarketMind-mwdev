"""Tests for Phase II scaffold seed policy."""

from __future__ import annotations

import pytest

from pysrc.meta.seed_policy import (
    build_run_identity,
    build_seed_lineage,
    derive_phase2_seed,
    derive_run_id,
    run_seed_root_from_int,
    scaffold_int_seed_from_content_tag,
)


@pytest.mark.determinism("d1")
def test_derive_run_id_stable_for_same_seed() -> None:
    assert derive_run_id(42) == derive_run_id(42)


@pytest.mark.determinism("d1")
def test_derive_run_id_differs_across_seeds() -> None:
    assert derive_run_id(1) != derive_run_id(2)


@pytest.mark.determinism("d1")
def test_run_identity_block_includes_seed_and_run_id() -> None:
    rid = build_run_identity(7)
    block = rid.to_block()
    assert block["seed"] == 7
    assert block["run_id"] == derive_run_id(7)
    assert block["non_promotable"] is True
    assert block["schema_version"] == "phase2_scaffold.run_identity.v1"


@pytest.mark.determinism("d1")
def test_scaffold_int_seed_from_content_tag_stable() -> None:
    tag = "sha256:" + "ab" * 32
    assert scaffold_int_seed_from_content_tag(tag) == scaffold_int_seed_from_content_tag(tag)


@pytest.mark.determinism("d1")
def test_scaffold_int_seed_pairs_with_derive_run_id() -> None:
    tag = "sha256:" + "cd" * 32
    seed = scaffold_int_seed_from_content_tag(tag)
    rid = build_run_identity(seed)
    assert rid.run_id == derive_run_id(seed)
    assert rid.seed == seed


@pytest.mark.determinism("d1")
def test_phase2_seed_derivation_is_stable_and_namespace_bound() -> None:
    root = run_seed_root_from_int(42)
    a = derive_phase2_seed(
        run_seed_root=root,
        namespace="task_sampling",
        context_string="fixture-a",
    )
    b = derive_phase2_seed(
        run_seed_root=root,
        namespace="task_sampling",
        context_string="fixture-a",
    )
    c = derive_phase2_seed(
        run_seed_root=root,
        namespace="checkpoint_reference_eval",
        context_string="fixture-a",
    )
    assert a.derived_seed_hex == b.derived_seed_hex
    assert a.uint64_seed == b.uint64_seed
    assert a.derived_seed_hex != c.derived_seed_hex
    assert len(a.derived_seed_hex) == 64


@pytest.mark.determinism("d1")
def test_seed_lineage_block_carries_canonical_root_and_derivations() -> None:
    root = run_seed_root_from_int(7)
    lineage = build_seed_lineage(
        run_seed_root=root,
        derivations=(("task_sampling", "fixture-a"),),
    )
    block = lineage.to_block()
    assert block["schema_version"] == "phase2.seed_lineage.v1"
    assert block["run_seed_root"] == root
    assert block["derived_seeds"][0]["namespace"] == "task_sampling"
    assert len(block["derived_seeds"][0]["derived_seed_hex"]) == 64

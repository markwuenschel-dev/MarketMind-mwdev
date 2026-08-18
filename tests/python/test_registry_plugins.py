import pytest

from pysrc.pipeline.stages.cleaning import (
    list_registered_cleaning_steps,
    registry_state_hash,
    resolve_cleaning_step,
)


@pytest.mark.determinism("d0")
def test_registry_lookup_builtin(deterministic_seed: int):
    _ = deterministic_seed
    registration = resolve_cleaning_step("impute.missing", "1")
    assert registration.step_type == "impute.missing"
    assert registration.version == "1"
    assert callable(registration.step_cls)


@pytest.mark.determinism("d0")
def test_registry_state_hash_stable(deterministic_seed: int):
    _ = deterministic_seed
    registrations = list_registered_cleaning_steps()
    assert registrations
    assert registry_state_hash() == registry_state_hash()

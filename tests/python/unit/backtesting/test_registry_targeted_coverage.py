from __future__ import annotations

import importlib
from typing import Any

import pytest

from pysrc.backtesting.contracts import registry as contract_registry
from pysrc.backtesting.contracts.errors import UnknownIdError


@pytest.fixture(autouse=True)
def reset_registry_state() -> None:
    engines = dict(contract_registry._ENGINES)
    execution_models = dict(contract_registry._EXECUTION_MODELS)
    cost_models = dict(contract_registry._COST_MODELS)
    ledgers = dict(contract_registry._LEDGERS)
    validators = dict(contract_registry._VALIDATORS)
    stores = dict(contract_registry._STORES)
    defaults_loaded = contract_registry._DEFAULTS_LOADED
    try:
        contract_registry._ENGINES.clear()
        contract_registry._EXECUTION_MODELS.clear()
        contract_registry._COST_MODELS.clear()
        contract_registry._LEDGERS.clear()
        contract_registry._VALIDATORS.clear()
        contract_registry._STORES.clear()
        contract_registry._DEFAULTS_LOADED = False
        yield
    finally:
        contract_registry._ENGINES.clear()
        contract_registry._ENGINES.update(engines)
        contract_registry._EXECUTION_MODELS.clear()
        contract_registry._EXECUTION_MODELS.update(execution_models)
        contract_registry._COST_MODELS.clear()
        contract_registry._COST_MODELS.update(cost_models)
        contract_registry._LEDGERS.clear()
        contract_registry._LEDGERS.update(ledgers)
        contract_registry._VALIDATORS.clear()
        contract_registry._VALIDATORS.update(validators)
        contract_registry._STORES.clear()
        contract_registry._STORES.update(stores)
        contract_registry._DEFAULTS_LOADED = defaults_loaded


@pytest.mark.determinism("d1")
def test_ensure_defaults_skips_optional_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_import_module(name: str) -> object:
        seen.append(name)
        if name == "pysrc.backtesting.engines.jax.engine":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    contract_registry._ensure_defaults()

    assert "pysrc.backtesting.engines.jax.engine" in seen
    assert contract_registry._DEFAULTS_LOADED is True


@pytest.mark.determinism("d1")
def test_ensure_defaults_raises_for_required_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> object:
        if name == "pysrc.backtesting.engines.vectorized.engine":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError):
        contract_registry._ensure_defaults()


@pytest.mark.determinism("d1")
def test_all_registry_wrappers_register_resolve_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_registry, "_ensure_defaults", lambda: None)

    created: dict[str, Any] = {
        "engine": object(),
        "execution_model": object(),
        "cost_model": object(),
        "ledger": object(),
        "validator": object(),
        "store": object(),
    }

    contract_registry.register_engine("b_engine", lambda: created["engine"])
    contract_registry.register_execution_model("b_exec", lambda: created["execution_model"])
    contract_registry.register_cost_model("b_cost", lambda: created["cost_model"])
    contract_registry.register_ledger("b_ledger", lambda: created["ledger"])
    contract_registry.register_validator("b_validator", lambda: created["validator"])
    contract_registry.register_store("b_store", lambda: created["store"])

    contract_registry.register_engine("a_engine", lambda: "sorted-first")
    contract_registry.register_execution_model("a_exec", lambda: "sorted-first")
    contract_registry.register_cost_model("a_cost", lambda: "sorted-first")
    contract_registry.register_ledger("a_ledger", lambda: "sorted-first")
    contract_registry.register_validator("a_validator", lambda: "sorted-first")
    contract_registry.register_store("a_store", lambda: "sorted-first")

    assert contract_registry.resolve_engine("b_engine") is created["engine"]
    assert contract_registry.resolve_execution_model("b_exec") is created["execution_model"]
    assert contract_registry.resolve_cost_model("b_cost") is created["cost_model"]
    assert contract_registry.resolve_ledger("b_ledger") is created["ledger"]
    assert contract_registry.resolve_validator("b_validator") is created["validator"]
    assert contract_registry.resolve_store("b_store") is created["store"]

    assert contract_registry.list_engines() == ["a_engine", "b_engine"]
    assert contract_registry.list_execution_models() == ["a_exec", "b_exec"]
    assert contract_registry.list_cost_models() == ["a_cost", "b_cost"]
    assert contract_registry.list_ledgers() == ["a_ledger", "b_ledger"]
    assert contract_registry.list_validators() == ["a_validator", "b_validator"]
    assert contract_registry.list_stores() == ["a_store", "b_store"]


@pytest.mark.determinism("d1")
def test_resolve_unknown_execution_model_reports_available_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract_registry, "_ensure_defaults", lambda: None)
    contract_registry.register_execution_model("known.exec", lambda: "x")

    with pytest.raises(UnknownIdError) as exc_info:
        contract_registry.resolve_execution_model("missing.exec")

    error = exc_info.value
    assert error.component_kind == "execution_model"
    assert error.requested_id == "missing.exec"
    assert error.available_ids == ("known.exec",)

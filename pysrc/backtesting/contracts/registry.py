from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from pysrc.backtesting.contracts.errors import UnknownIdError

Factory = Callable[[], Any]


_ENGINES: dict[str, Factory] = {}
_EXECUTION_MODELS: dict[str, Factory] = {}
_COST_MODELS: dict[str, Factory] = {}
_LEDGERS: dict[str, Factory] = {}
_VALIDATORS: dict[str, Factory] = {}
_STORES: dict[str, Factory] = {}
_DEFAULTS_LOADED = False

_OPTIONAL_DEFAULT_MODULES = {
    "pysrc.backtesting.engines.jax.engine",
    "pysrc.backtesting.engines.backtrader.adapter",
}

_DEFAULT_MODULES = (
    "pysrc.backtesting.engines.vectorized.engine",
    "pysrc.backtesting.engines.event_driven.engine",
    "pysrc.backtesting.engines.jax.engine",
    "pysrc.backtesting.engines.backtrader.adapter",
    "pysrc.backtesting.engines.chain",
    "pysrc.backtesting.models.fill",
    "pysrc.backtesting.models.fees",
    "pysrc.backtesting.models.slippage",
    "pysrc.backtesting.models.latency",
    "pysrc.backtesting.models.liquidity",
    "pysrc.backtesting.models.borrow_interest",
    "pysrc.backtesting.spine.ledger.ledger",
    "pysrc.backtesting.validation.statistical.validator",
    "pysrc.backtesting.validation.mechanical.validator",
)


def _ensure_defaults() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    for module_name in _DEFAULT_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            # Optional backends stay skippable at registry bootstrap time.
            if module_name in _OPTIONAL_DEFAULT_MODULES:
                continue
            raise
    _DEFAULTS_LOADED = True


def _register(registry: dict[str, Factory], component_id: str, factory: Factory) -> None:
    registry[component_id] = factory


def _resolve(registry: dict[str, Factory], component_kind: str, component_id: str) -> Any:
    _ensure_defaults()
    try:
        return registry[component_id]()
    except KeyError as exc:
        raise UnknownIdError(
            component_kind=component_kind,
            requested_id=component_id,
            available_ids=tuple(sorted(registry)),
            hint=f"Register the {component_kind} via register_{component_kind}().",
        ) from exc


def _list(registry: dict[str, Factory]) -> list[str]:
    _ensure_defaults()
    return sorted(registry)


def register_engine(component_id: str, factory: Factory) -> None:
    _register(_ENGINES, component_id, factory)


def resolve_engine(component_id: str) -> Any:
    return _resolve(_ENGINES, "engine", component_id)


def list_engines() -> list[str]:
    return _list(_ENGINES)


def register_execution_model(component_id: str, factory: Factory) -> None:
    _register(_EXECUTION_MODELS, component_id, factory)


def resolve_execution_model(component_id: str) -> Any:
    return _resolve(_EXECUTION_MODELS, "execution_model", component_id)


def list_execution_models() -> list[str]:
    return _list(_EXECUTION_MODELS)


def register_cost_model(component_id: str, factory: Factory) -> None:
    _register(_COST_MODELS, component_id, factory)


def resolve_cost_model(component_id: str) -> Any:
    return _resolve(_COST_MODELS, "cost_model", component_id)


def list_cost_models() -> list[str]:
    return _list(_COST_MODELS)


def register_ledger(component_id: str, factory: Factory) -> None:
    _register(_LEDGERS, component_id, factory)


def resolve_ledger(component_id: str) -> Any:
    return _resolve(_LEDGERS, "ledger", component_id)


def list_ledgers() -> list[str]:
    return _list(_LEDGERS)


def register_validator(component_id: str, factory: Factory) -> None:
    _register(_VALIDATORS, component_id, factory)


def resolve_validator(component_id: str) -> Any:
    return _resolve(_VALIDATORS, "validator", component_id)


def list_validators() -> list[str]:
    return _list(_VALIDATORS)


def register_store(component_id: str, factory: Factory) -> None:
    _register(_STORES, component_id, factory)


def resolve_store(component_id: str) -> Any:
    return _resolve(_STORES, "store", component_id)


def list_stores() -> list[str]:
    return _list(_STORES)

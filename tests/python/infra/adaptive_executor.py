# tests/python/infra/adaptive_executor.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

try:
    from _pytest.outcomes import Skipped, XFailed  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover

    class Skipped(Exception):  # type: ignore
        pass

    class XFailed(Exception):  # type: ignore
        pass


from tests.python.infra.scenario_models import (
    AnyScenario,
    EnsembleScenario,
    HandlerResult,
    HighRiskPatternScenario,
    OptimalSequenceScenario,
    ParallelThresholdScenario,
)

try:
    from tests.python.infra.compat_adapter import get_cap  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover

    def get_cap(engine: Any, key: str, default: Any = None) -> Any:
        caps = getattr(engine, "caps", {}) or {}
        return caps.get(key, default)


# ----------------------------------------------------------------------------------
# Registry surfaces
# ----------------------------------------------------------------------------------

# (module_key, scenario_type) -> handler meta
# handler meta: {"fn": callable, "requires": Dict[str, Any], "version": "1.0"}
_HANDLERS: dict[tuple[str, str], dict[str, Any]] = {}

# pluggable classification hooks so new domains can bolt in without editing core
_CLASSIFIERS: list[Callable[[AnyScenario], str | None]] = []


def register(
    module_key: str,
    scenario_type: str,
    *,
    requires: dict[str, Any] | None = None,
    version: str = "1.0",
):
    # decorator for per-module harnesses
    def _wrap(fn: Callable[[AnyScenario, Any], dict[str, Any]]):
        _HANDLERS[(module_key, scenario_type)] = {
            "fn": fn,
            "requires": requires or {},
            "version": version,
        }
        return fn

    return _wrap


def register_classifier(fn: Callable[[AnyScenario], str | None]):
    # classifier should return a scenario_type string or None
    _CLASSIFIERS.append(fn)
    return fn


# ----------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------

# allowed outcome set
_KNOWN_TYPES = {
    "strategy",
    "module",
    "ensemble",
    "engine",
    "cleaning",
    "preprocess",
    "feature_extraction",
    "feature_selection",
    "normalization",
    "outlier_pass",
    "unknown",
}


def _base_classify_scenario_type(scn: AnyScenario) -> str:
    # engine-origin strict models
    if isinstance(
        scn,
        (ParallelThresholdScenario, HighRiskPatternScenario, OptimalSequenceScenario),
    ):
        return "engine"

    # ensemble-style flexible bucket with metadata
    if isinstance(scn, EnsembleScenario):
        origins = getattr(scn, "_discovered_from", []) or []
        stage = getattr(scn, "_pipeline_stage", None)
        kind = getattr(scn, "kind", "")
        overrides = getattr(scn, "overrides", {}) or {}

        # 1) explicit pipeline stage hints
        if stage in (
            "cleaning",
            "preprocess",
            "feature_extraction",
            "feature_selection",
            "normalization",
            "outlier_pass",
        ):
            return stage

        # 2) module-level toggles (e.g., <module>@L## tags)
        if any(isinstance(o, str) and o.startswith("<module>@") for o in origins):
            return "module"

        # 3) ensure ensemble wiring cases (like price_col) route to 'ensemble'
        #    even if origins mention a leaf Strategy
        if isinstance(kind, str) and kind.startswith("ensemble_") and ("price_col" in overrides):
            return "ensemble"

        # 4) direct ensemble ctor/wiring in origins
        if any(isinstance(o, str) and "EnsemblePipelineStrategy" in o for o in origins):
            return "ensemble"

        # 5) leaf strategy ctor / guardrails
        for origin in origins:
            if not isinstance(origin, str):
                continue
            if (
                "Strategy" in origin
                and not origin.startswith("EnsemblePipelineStrategy")
                and not origin.startswith("<module>")
            ):
                return "strategy"

            if "ensemble_all_bad" in kind:
                return "ensemble"  # or "strategy" depending on what you want
            if "ensemble_loop" in kind:
                return "ensemble"

        # 6) fallback: kind prefix
        if isinstance(kind, str) and kind.startswith("ensemble_"):
            return "ensemble"

        return "unknown"

    # totally new families -> unknown
    return "unknown"


def classify_scenario_type(scn: AnyScenario) -> str:
    # custom classifiers first
    for clf in _CLASSIFIERS:
        try:
            t = clf(scn)
        except Exception:  # noqa: BLE001 - must catch all from user classifiers
            t = None
        if t:
            return t
    # then base rules
    t = _base_classify_scenario_type(scn)
    return t if t in _KNOWN_TYPES else "unknown"


# ----------------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------------


def _is_known_type(t: str) -> bool:
    return t in _KNOWN_TYPES and t != "unknown"


def _resolve_handler(module_key: str, scenario_type: str) -> dict[str, Any] | None:
    return _HANDLERS.get((module_key, scenario_type))


def _capabilities_ok(engine: Any, requires: dict[str, Any]) -> tuple[bool, str]:
    if not requires:
        return True, ""
    missing: list[str] = []
    for key, expected in requires.items():
        actual = get_cap(engine, key, None)
        if isinstance(expected, bool):
            if bool(actual) is not bool(expected):
                missing.append(f"{key}=={expected}")
        else:
            if actual != expected:
                missing.append(f"{key}=={expected}")
    if missing:
        return False, ", ".join(missing)
    return True, ""


def run(
    module_key: str,
    scenario: AnyScenario,
    caplog,
    *,
    engine: Any = None,
) -> HandlerResult:
    # decide scenario_type
    scenario_type = classify_scenario_type(scenario)
    print(
        f"EXECUTOR: scenario.kind={getattr(scenario, 'kind', '?')}, classified as scenario_type={scenario_type}"
    )

    # unknown -> xfail (visible gap)
    if scenario_type == "unknown":
        print("EXECUTOR: xfail - unknown type")
        pytest.xfail(
            f"no classifier/handler for scenario: {getattr(scenario, 'kind', type(scenario).__name__)}"
        )

    # known but no handler -> skip
    meta = _resolve_handler(module_key, scenario_type)
    if not meta:
        print(f"EXECUTOR: skip - no handler for ({module_key}, {scenario_type})")
        pytest.skip(f"no handler for module '{module_key}' and type '{scenario_type}'")
    print("EXECUTOR: found handler, calling it")
    # capability gating
    ok, why = _capabilities_ok(engine, meta.get("requires", {}))
    if not ok:
        pytest.skip(f"capability-gated: requires {why}")

    fn = meta["fn"]

    # execute handler; don't swallow pytest control-flow exceptions
    raw: dict[str, Any]  # explicit declaration
    try:
        raw = fn(scenario, caplog)
    except (Skipped, XFailed):  # let pytest handle these
        raise
    except Exception as e:
        pytest.fail(
            f"handler crashed: {fn.__name__} for ({module_key}, {scenario_type}). "
            f"{type(e).__name__}: {e}\nThis is a test harness failure, not a product failure."
        )
        raise  # pragma: no cover - pytest.fail raises, but satisfy type checker

    # normalize / validate contract
    try:
        result = HandlerResult.model_validate(raw)
    except Exception as e:
        pytest.fail(f"handler {fn.__name__} returned malformed result. Got: {raw}\nError: {e}")
        raise  # pragma: no cover - pytest.fail raises, but satisfy type checker

    return result

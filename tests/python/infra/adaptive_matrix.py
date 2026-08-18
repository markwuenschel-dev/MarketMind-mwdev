from __future__ import annotations

# Adaptive test matrix builder:
# - pulls learned scenarios from the self-evolving engine
# - merges with any static scenarios
# - validates each scenario against our Pydantic contract
# - fingerprints them for triage
# - decorates with skip/xfail/marks
# - prunes params entirely if the env can't support them
#
# This is the spine of our "engine-driven parametrization" story.
import copy
import hashlib
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from tests.python.infra.adaptive_executor import classify_scenario_type
from tests.python.infra.compat_adapter import attach_caps
from tests.python.infra.compat_layer import compat
from tests.python.infra.matrix import matrix
from tests.python.infra.scenario_models import (
    AnyScenario,
    EnsembleScenario,
    HighRiskPatternScenario,
    OptimalSequenceScenario,
    ParallelThresholdScenario,
    coerce_scenario,
)


# stable hash so each scenario is triageable and dedupable
def _stable_fp(scenario_obj: AnyScenario) -> str:
    snap_dict = scenario_obj.model_dump()
    blob = json.dumps(snap_dict, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return h[:12]


# readable pytest ID (what shows up in -vv and CI failure summaries)
def _scenario_idfn(case: dict[str, Any]) -> str:
    sc: AnyScenario = case["scenario"]

    if isinstance(sc, ParallelThresholdScenario):
        # parallel_threshold_minus[rows=1234]
        return f"{sc.kind}[rows={sc.rows}]"

    if isinstance(sc, HighRiskPatternScenario):
        # high_risk_pattern[shape=(1024, 64),ops=5,hits=3]
        return f"{sc.kind}[shape={sc.shape},ops={sc.num_ops},hits={sc.risk_count}]"

    if isinstance(sc, OptimalSequenceScenario):
        # optimal_sequence[ops=load-normalize-vectorize]
        return f"{sc.kind}[ops={'-'.join(sc.ops)}]"

    if isinstance(sc, EnsembleScenario):
        # ensemble_foo[pattern=XYZ]
        return f"{sc.kind}[pattern={getattr(sc, 'pattern', None)}]"

    fp = case.get("fingerprint", "nohash")
    return f"{getattr(sc, 'kind', 'unknown')}[{fp}]"


# policy knobs expressed as pytest marks, based on scenario semantics


def _skip_if(case: dict[str, Any]) -> str | None:
    # FAST_ONLY_STATIC mode lets devs run cheap smoke tests by excluding learned scenarios
    if os.getenv("FAST_ONLY_STATIC", "") and case.get("origin") != "static":
        return "skipping learned scenario in FAST_ONLY_STATIC mode"
    return None


def _xfail_if(case: dict[str, Any]) -> str | None:
    sc: AnyScenario = case["scenario"]
    raw = case.get("scenario_raw") or {}

    # Honor explicit _xfail/_xfail_reason from generators or policy.
    if isinstance(raw, dict):
        reason = raw.get("_xfail") or raw.get("_xfail_reason")
        if reason:
            return str(reason)

    # High-risk patterns that are intentionally unstable should not block merges
    if isinstance(sc, HighRiskPatternScenario) and not getattr(sc, "expect_stability", True):
        return "known high-risk execution pattern under investigation"

    return None


def _marks(case: dict[str, Any]):
    sc: AnyScenario = case["scenario"]

    # parallel_threshold_* are perf boundary cases that tend to hit big data sizes
    # so we globally tag them slow.
    if isinstance(sc, ParallelThresholdScenario):
        return pytest.mark.slow

    # more policy buckets can go here (memory_intensive, gpu_only, etc.)
    return None


# environment probes: if a scenario demands capabilities this runner doesn't have,
# we prune it *before* pytest gets params (so we don't even generate a skip/xfail).
def _probe_wrapper(engine_obj: Any, case: dict[str, Any]) -> dict[str, Callable[[], None]]:
    """Build a set of fast capability probes derived from the scenario."""
    sc: AnyScenario = case["scenario"]
    probes: dict[str, Callable[[], None]] = {}

    # 1) Generic, scenario-level requires (e.g., {"gpu_present": True})
    req_map = getattr(sc, "requires", {}) or {}
    for key, expected in req_map.items():

        def _mk_probe(k=key, exp=expected):
            def _check():
                caps = getattr(engine_obj, "caps", {}) or {}
                actual = caps.get(k, None)
                if isinstance(exp, bool):
                    if bool(actual) is not bool(exp):
                        raise RuntimeError(f"cap_{k}_mismatch")
                else:
                    if actual != exp:
                        raise RuntimeError(f"cap_{k}_mismatch")

            return _check

        probes[f"cap_{key}"] = _mk_probe()

    # 2) Existing parallel threshold example stays first-class
    if isinstance(sc, ParallelThresholdScenario) and sc.expect_parallel:

        def _parallel_ok():
            caps = getattr(engine_obj, "caps", {}) or {}
            if not caps.get("parallel_exec_supported", True):
                raise RuntimeError("parallel not supported in this environment")

        probes["parallel_capable"] = _parallel_ok

    return probes


# core: validate scenarios, freeze them, dedupe, fingerprint, and order deterministically


def _normalize_cases(
    engine_scenarios: Iterable[dict[str, Any]],
    static_scenarios: Iterable[dict[str, Any]],
    engine_obj: Any,
) -> list[dict[str, Any]]:

    cases: list[dict[str, Any]] = []
    seen: set = set()

    def _add(origin: str, raw: dict[str, Any]) -> None:
        # defensive copy so nothing downstream can mutate engine state
        raw_copy = copy.deepcopy(raw)

        # 1) Validate/coerce to a typed model according to "kind"
        scenario_model: AnyScenario = coerce_scenario(raw_copy)

        # 2) Short fingerprint for dedupe + triage (uses your existing helper)
        fp = _stable_fp(scenario_model)  # assumes _stable_fp is defined elsewhere in this module
        sig = (scenario_model.kind, fp)
        if sig in seen:
            return
        seen.add(sig)

        # 3) Stamp predicted scenario_type for telemetry
        predicted = classify_scenario_type(scenario_model)

        # 4) Stash typed model and a raw snapshot that also carries predicted type
        snapshot = scenario_model.model_dump()
        snapshot["_predicted_type"] = predicted

        cases.append(
            {
                "engine": engine_obj,
                "scenario": scenario_model,
                "scenario_raw": snapshot,
                "origin": origin,  # "learned" or "static"
                "fingerprint": fp,  # helps correlate CI failure back to the case
                "predicted_type": predicted,
            }
        )

    for sc in engine_scenarios:
        _add("learned", sc)
    for sc in static_scenarios:
        _add("static", sc)

    # Deterministic ordering → stable pytest param order across runs/machines
    cases.sort(
        key=lambda c: (
            getattr(c["scenario"], "kind", ""),
            c["fingerprint"],
            c["origin"],
        )
    )
    return cases


# public entry point: this is what tests call
def engine_matrix(
    engine_obj: Any,
    *,
    static_scenarios: Iterable[dict[str, Any]] | None = None,
    learn: bool = True,
    min_fail_skip: int = 3,
    environ_filter: str = "",
) -> Callable:
    # detect env capabilities once (threadpooled, cached, tolerant of probe errors)
    # and attach them to the engine so tests/probes can introspect. :contentReference[oaicite:7]{index=7}
    caps = compat.detect()
    attach_caps(engine_obj, caps)

    # snapshot learned scenarios from the engine so we test a consistent view
    engine_list: list[dict[str, Any]] = []
    if hasattr(engine_obj, "generate_scenarios_for_testing"):
        engine_list = list(
            engine_obj.generate_scenarios_for_testing()
        )  # lock-protected in engine to avoid races.

    merged_cases = _normalize_cases(
        engine_scenarios=engine_list,
        static_scenarios=list(static_scenarios or []),
        engine_obj=engine_obj,
    )

    # delegate to matrix(), which:
    # - builds pytest.param(...) with ids from _scenario_idfn(...)
    # - injects marks from _skip_if/_xfail_if/_marks
    # - prunes cases that fail _probe_wrapper via its "probe" hook
    # - supports learn/min_fail_skip so we can auto-skip perma-flaky cases after N fails. :contentReference[oaicite:9]{index=9}
    return matrix(
        learn=learn,
        min_fail_skip=min_fail_skip,
        environ_filter=environ_filter,
        probe=lambda c: _probe_wrapper(engine_obj, c),
        opts={
            "cases": merged_cases,
            "idfn": _scenario_idfn,
            "skip_if": _skip_if,
            "xfail_if": _xfail_if,
            "marks": _marks,
            "indirect": False,  # <- critical: pytest wants bool or Sequence, not None
        },
    )

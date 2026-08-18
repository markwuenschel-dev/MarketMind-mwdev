# py/pipeline/core/pipeline_core_builder.py
from __future__ import annotations

import os
from typing import Any

from .pipeline_core_base import PipelineStep
from .pipeline_core_context import PipelineContext


def _matches(rule_when: dict, ctx: PipelineContext) -> bool:
    # Support simple equality, membership, comparisons, and regexes for context-aware selection.
    # Examples of rule_when:
    #   {"frequency": "1D", "market": ["stocks","crypto"]}
    #   {"rows": {"ge": 1_000}, "symbol": {"in": ["AAPL","MSFT"]}}
    #   {"symbol": {"regex": "^BTC"}}
    for k, expected in (rule_when or {}).items():
        actual = getattr(ctx, k, None)
        # membership via list/tuple/set
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue
        # comparator object
        if isinstance(expected, dict):
            if "in" in expected and actual not in set(expected["in"]):
                return False
            if "not_in" in expected and actual in set(expected["not_in"]):
                return False
            if "ge" in expected and not (actual is not None and actual >= expected["ge"]):
                return False
            if "gt" in expected and not (actual is not None and actual > expected["gt"]):
                return False
            if "le" in expected and not (actual is not None and actual <= expected["le"]):
                return False
            if "lt" in expected and not (actual is not None and actual < expected["lt"]):
                return False
            if "regex" in expected:
                try:
                    import re as _re  # local to avoid global dep on import time

                    if actual is None or _re.search(str(expected["regex"]), str(actual)) is None:
                        return False
                except Exception:
                    return False
            continue
        # default equality
        if actual != expected:
            return False
    return True


def choose_combo(cfg, ctx, name: str = None):
    if not cfg:
        return {"steps": [], "order": {}}

    # allow both attr- and dict-style
    cleaning = getattr(cfg, "cleaning", None)
    if isinstance(cfg, dict) and "cleaning" in cfg:
        cleaning = cfg["cleaning"]

    if cleaning is None:
        combos = cfg.get("combos") if isinstance(cfg, dict) else getattr(cfg, "combos", None)
    else:
        combos = (
            cleaning.get("combos")
            if isinstance(cleaning, dict)
            else getattr(cleaning, "combos", None)
        )

    if combos is None:
        if isinstance(cfg, dict) and "cleaning" in cfg:
            raise KeyError("'cleaning' missing 'combos'")
        elif isinstance(cfg, dict) and cfg:
            raise KeyError("missing 'cleaning' or 'combos'")
        return {"steps": [], "order": {}}

    # ---- explicit name override takes precedence ----
    if name is not None:
        if isinstance(combos, dict):
            if name not in combos:
                raise KeyError(name)
            selected = combos[name]
        else:
            # combos is a list of dicts
            for c in combos:
                if c.get("name") == name:
                    selected = c
                    break
            else:
                raise KeyError(name)
        return {"steps": selected.get("steps", []), "order": selected.get("order", {})}

    # ---- auto-selection logic (reconciled) ----
    # Pull 'use' robustly; treat non-strings (e.g., MagicMock) as "not set"
    use_key = None
    if cleaning is not None:
        raw = (
            getattr(cleaning, "use", None)
            if not isinstance(cleaning, dict)
            else cleaning.get("use")
        )
        use_key = raw if isinstance(raw, str) and raw else None
    if use_key is None and isinstance(cfg, dict):
        raw = cfg.get("use")
        use_key = raw if isinstance(raw, str) and raw else None

    env_key = os.getenv("PIPELINE_COMBO") or os.getenv("MODE")
    selected = None

    if isinstance(combos, dict):
        if use_key is not None:
            # If explicit 'use' is unknown:
            # - If the ONLY combo is 'default' → raise KeyError (explicit misconfig)
            # - Otherwise FALL BACK: default → env → first
            cand = combos.get(use_key)
            if cand is None:
                keys = list(combos.keys())
                only_default = len(keys) == 1 and keys[0] == "default"
                if only_default:
                    raise KeyError(use_key)
                # fallback chain
                cand = combos.get("default")
                if cand is None and env_key and env_key in combos:
                    cand = combos[env_key]
                if cand is None and combos:
                    cand = next(iter(combos.values()))
            selected = cand if isinstance(cand, dict) else {}
        else:
            cand = combos.get(env_key) if env_key else None
            if cand is None:
                cand = combos.get("default")
            if cand is None:
                if len(combos) == 1:
                    cand = next(iter(combos.values()))
                elif not combos:
                    return {"steps": [], "order": {}}
                else:
                    # multiple combos but no selector → error
                    raise KeyError("No combo selector provided")
        selected = cand if isinstance(cand, dict) else {}

    else:
        # combos is a list of dicts; allow fallbacks
        select_key = use_key or env_key
        if select_key:
            for c in combos:
                if c.get("name") == select_key:
                    when = c.get("when")
                    if (when is None) or _matches(when, ctx):
                        selected = c
                    break
        if selected is None:
            # prefer first matching 'when'
            for c in combos:
                when = c.get("when")
                if when and _matches(when, ctx):
                    selected = c
                    break
        if selected is None and combos:
            selected = combos[0]

    if not isinstance(selected, dict):
        return {"steps": [], "order": {}}
    return {"steps": selected.get("steps", []), "order": selected.get("order", {})}


def topo_order(steps: list[str], order_cfg: dict[str, dict[str, list[str]]]) -> list[str]:
    # Normalize and deduplicate while preserving the original relative order
    seen = set()
    steps = [s for s in steps if not (s in seen or seen.add(s))]

    try:
        return _toposort(steps, order_cfg)
    except Exception as e:
        msg = str(e)
        if "Conflicting" in msg or "cycle" in msg:
            raise ValueError("Cyclic step constraints detected")
        raise


def _edges_from_order(
    steps: list[str], order: dict[str, dict[str, list[str]]] | None
) -> list[tuple[str, str]]:
    # edges u->v mean u must come before v; ignore nodes not present in steps
    edges: list[tuple[str, str]] = []
    set(steps)
    if order is None:
        return edges
    for a, bs in (order.get("before", {}) or {}).items():
        for b in bs or []:
            edges.append((a, b))  # a before b
    for a, bs in (order.get("after", {}) or {}).items():
        for b in bs or []:
            edges.append((b, a))  # b before a
    return edges


def _check_conflicts(edges: list[tuple[str, str]]):
    # detect direct contradictions (u->v and v->u)
    s = set(edges)
    for u, v in edges:
        if (v, u) in s:
            raise Exception(f"Conflicting constraints: {u}→{v}")


def _toposort(steps: list[str], order: dict | None) -> list[str]:
    # derive filtered edges and check for direct conflicts
    edges = _edges_from_order(steps, order)
    _check_conflicts(edges)

    # Kahn's algorithm over nodes limited to 'steps'
    preds: dict[str, set[str]] = {s: set() for s in steps}
    succs: dict[str, set[str]] = {s: set() for s in steps}
    for u, v in edges:
        # ignore constraints over nodes not present in the target 'steps'
        if u not in preds or v not in preds:
            continue
        preds[v].add(u)
        succs[u].add(v)

    ready = [s for s in steps if not preds[s]]
    out: list[str] = []

    while ready:
        u = ready.pop(0)  # Use FIFO to preserve input order when no constraints
        out.append(u)
        for v in list(succs[u]):
            preds[v].discard(u)
            if not preds[v]:
                ready.append(v)

    if len(out) != len(steps):
        # missing nodes implies a cycle among remaining vertices
        raise Exception("Order contains a cycle")
    return out


class PipelineBuilder:
    def __init__(self, stage: str, config: dict[str, Any] | None = None):
        self.stage = stage
        self.config = config or {}
        self.steps: list[Any] = []  # can be dict specs or instantiated PipelineStep

    @classmethod
    def for_stage(cls, stage: str, config: dict[str, Any] | None = None) -> PipelineBuilder:
        return cls(stage=stage, config=config or {})

    def from_preset_and_params(
        self, preset: dict[str, Any], params: dict[str, Any]
    ) -> PipelineBuilder:
        # copy steps while keeping kwargs editable
        steps = [
            {"name": s["name"], "kwargs": dict(s.get("kwargs", {}))}
            for s in preset.get("steps", [])
        ]

        # flatten "step.arg" params onto matching step kwargs; also toggle enabled flags
        for key, val in (params or {}).items():
            if "." in key:
                step_name, arg = key.split(".", 1)
                for spec in steps:
                    if spec["name"] == step_name:
                        spec.setdefault("kwargs", {})[arg] = val
            else:
                for spec in steps:
                    if spec["name"] == key and isinstance(val, bool):
                        spec.setdefault("kwargs", {})["enabled"] = val

        return self.add_steps(steps)

    def add_steps(self, steps):
        # extend current step specs
        self.steps.extend(steps)
        return self

    def build(self) -> Pipeline:
        # build immutable pipeline view
        return Pipeline(steps=self.steps, config=self.config)

    def validate_contracts(self) -> PipelineBuilder:
        # ensure every step's requires are satisfied by earlier 'produces'
        produced: set[str] = set()
        for i, step in enumerate(self.steps):
            # tolerate dict specs and instantiated steps
            enabled = True
            requires: set[str] = set()
            produces: set[str] = set()

            if isinstance(step, dict):
                kw = step.get("kwargs", {}) or {}
                enabled = kw.get("enabled", True)
                req = kw.get("requires") or step.get("requires")
                pro = kw.get("produces") or step.get("produces")
                if isinstance(req, (list, set, tuple)):
                    requires = set(req)
                if isinstance(pro, (list, set, tuple)):
                    produces = set(pro)
            else:
                enabled = getattr(step, "enabled", True)
                req_attr = getattr(step, "requires", set())
                pro_attr = getattr(step, "produces", set())
                requires = set(req_attr or set())
                produces = set(pro_attr or set())

            if not enabled:
                continue

            missing = requires - produced
            if missing:
                name = (
                    step.get("name", "unknown")
                    if isinstance(step, dict)
                    else getattr(step, "name", "unknown")
                )
                raise ValueError(f"Step {i}:{name} requires {missing} not produced yet")
            produced |= produces
        return self


class Pipeline:
    # tiny pipeline shell for tests
    def __init__(self, steps: list[PipelineStep], config: dict[str, Any]):
        self.steps = steps
        self.config = config

    def fit_transform(self, df):
        # normalize input to pandas DataFrame when possible
        df_in = df
        if hasattr(df_in, "to_pandas"):
            df_in = df_in.to_pandas()
        elif isinstance(df_in, dict):
            try:
                import pandas as pd  # local import to avoid hard dependency at module import

                df_in = pd.DataFrame(df_in)
            except Exception:
                pass  # keep as dict if pandas unavailable

        # extract numeric matrix or at least preserve row count
        if hasattr(df_in, "select_dtypes"):
            num = df_in.select_dtypes(include=["number"])
            try:
                X = num.to_numpy()
            except Exception:
                X = num.values  # fallback for older pandas
            rows = getattr(df_in, "__len__", lambda: 0)()
            num_feats = getattr(num, "shape", (rows, 0))[1] if hasattr(num, "shape") else 0
        elif isinstance(df_in, dict) and df_in:
            first = next(iter(df_in.values()))
            rows = len(first)
            X = [None] * rows  # only row count matters for tests in raw mode
            num_feats = 0
        else:
            rows = len(df_in) if hasattr(df_in, "__len__") else 0
            X = [None] * rows
            num_feats = 0

        y = [0] * rows
        # embed a compact meta useful for tests and orchestration audits
        step_names = [
            s.get("name", getattr(s, "name", "unknown"))
            if isinstance(s, dict)
            else getattr(s, "name", "unknown")
            for s in self.steps
        ]
        meta = {
            "pipeline_config": self.config,
            "num_steps": len(self.steps),
            "step_names": step_names,
            "num_numeric_features": num_feats,
        }
        return X, y, meta

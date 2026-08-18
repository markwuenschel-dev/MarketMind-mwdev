import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, cast
from typing import Hashable as _HashableAlias

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

@dataclass
class BranchCase:
    value: Any
    ctx: str
    body_summary: str
    body_details: Dict[str, Any]
    co_conditions: Set[Tuple[str, Any]] = field(default_factory=set)


@dataclass
class BranchDim:
    selector: str
    cases: List[BranchCase] = field(default_factory=list)
    default_value: Optional[Any] = None


@dataclass
class ScenarioStats:
    total_scenarios: int = 0
    by_kind: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    selectors: Set[str] = field(default_factory=set)
    with_defaults: int = 0
    with_fallbacks: int = 0
    combo_scenarios: int = 0
    merged_duplicates: int = 0
    new_scenarios: int = 0
    branch_coverage_pct: Optional[float] = None
    interaction_coverage_pct: Optional[float] = None
    negative_space_gaps: List[Dict[str, Any]] = field(default_factory=list)
    schema_errors: int = 0
    snapshots_taken: int = 0
    drift_count: int = 0


@dataclass
class PolicyConfig:
    mutually_exclusive: List[Dict[str, Any]] = field(default_factory=list)
    exclude_combos: List[Dict[str, Any]] = field(default_factory=list)
    risk_weights: Dict[str, int] = field(default_factory=dict)
    max_cartesian: int = 20
    snapshot_timeout: float = 30.0
    harness_command: Optional[str] = None


_UNSUPPORTED = object()

def _freeze_for_sig(v: Any) -> _HashableAlias:
    # primitives are already hashable
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    # recursively freeze sequences
    if isinstance(v, tuple):
        return tuple(_freeze_for_sig(x) for x in v)
    if isinstance(v, list):
        return tuple(_freeze_for_sig(x) for x in v)
    # sets: sort with key=repr to avoid cross-type compare errors
    if isinstance(v, set):
        return tuple(sorted((_freeze_for_sig(x) for x in v), key=repr))
    # dicts: sort by key; freeze values
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze_for_sig(val)) for k, val in v.items()))
    # fallback: if hashable, keep; else use repr for determinism
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)

def make_overrides_signature(overrides: Dict[str, Any]) -> Tuple[Tuple[str, _HashableAlias], ...]:
    return tuple(sorted((k, _freeze_for_sig(v)) for k, v in overrides.items()))

# ===========================================================
# Schema loading and validation
# ===========================================================

def load_schema_from_file(schema_path: str) -> Dict[str, Dict[str, Any]]:
    """Load harness schema from JSON or YAML file"""
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    content = path.read_text()

    if path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise RuntimeError("PyYAML required for YAML schema files: pip install pyyaml")
        schema = yaml.safe_load(content)
    elif path.suffix == '.json':
        schema = json.loads(content)
    else:
        raise ValueError(f"Unsupported schema format: {path.suffix}")

    return normalize_schema_format(schema)


def load_schema_from_module(module_path: str) -> Dict[str, Dict[str, Any]]:
    """Import Python module and extract HARNESS_SCHEMA constant"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("harness_schema", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'HARNESS_SCHEMA'):
        raise ValueError(f"Module {module_path} has no HARNESS_SCHEMA")

    return normalize_schema_format(module.HARNESS_SCHEMA)


def normalize_schema_format(raw_schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize various schema formats to internal representation"""
    normalized = {}

    for selector, spec in raw_schema.items():
        if isinstance(spec, dict):
            entry = {
                "inject_as": spec.get("inject_as", selector),
                "type": spec.get("type", set()),
            }

            if "allowed_values" in spec:
                entry["type"] = set(spec["allowed_values"])
            elif "values" in spec:
                entry["type"] = set(spec["values"])

            if "coerce" in spec and callable(spec["coerce"]):
                entry["coerce"] = spec["coerce"]
            elif spec.get("coerce_bool"):
                entry["coerce"] = cast(Any, lambda v: "1" if v is True else "0" if v is False else v)

            normalized[selector] = entry
        else:
            normalized[selector] = {
                "inject_as": selector,
                "type": set(spec) if isinstance(spec, (list, tuple, set)) else set(),
            }

    return normalized


def get_default_schema() -> Dict[str, Dict[str, Any]]:
    """Fallback schema when none provided"""
    return {
        "config.mode": {
            "inject_as": "config.mode",
            "type": {"fast", "safe", "strict"},
        },
        "debug": {
            "inject_as": "runtime.debug",
            "type": {True, False},
        },
    }


# ===========================================================
# Policy configuration loading
# ===========================================================

def load_policy_from_file(policy_path: str) -> PolicyConfig:
    """Load policy configuration from JSON/YAML"""
    path = Path(policy_path)
    if not path.exists():
        return PolicyConfig()

    content = path.read_text()

    if path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            print("Warning: PyYAML not available, skipping policy file", file=sys.stderr)
            return PolicyConfig()
        data = yaml.safe_load(content)
    elif path.suffix == '.json':
        data = json.loads(content)
    else:
        return PolicyConfig()

    return PolicyConfig(
        mutually_exclusive=data.get("mutually_exclusive", []),
        exclude_combos=data.get("exclude_combos", []),
        risk_weights=data.get("risk_weights", {}),
        max_cartesian=data.get("max_cartesian", 20),
        snapshot_timeout=data.get("snapshot_timeout", 30.0),
        harness_command=data.get("harness_command"),
    )


# ===========================================================
# Canonical selector normalization
# ===========================================================
# --- hash-safe freezing for dedup/signatures ---

from typing import Hashable

def _freeze_for_signature(v: Any) -> Hashable:
    # primitives are already hashable
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    # lists/tuples -> tuple of frozen values
    if isinstance(v, (list, tuple)):
        return tuple(_freeze_for_signature(x) for x in v)
    # sets -> sorted tuple of frozen values (order-independent)
    if isinstance(v, set):
        return tuple(sorted(_freeze_for_signature(x) for x in v))
    # dicts -> sorted tuple of (key, frozen(value))
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze_for_signature(vv)) for k, vv in v.items()))
    # callables -> stable identifier
    if callable(v):
        qual = getattr(v, "__qualname__", None) or getattr(v, "__name__", None) or "<callable>"
        mod = getattr(v, "__module__", None) or "<module>"
        return ("<callable>", mod, qual)
    # everything else -> type name only (stable enough for dedup; avoids id()-based hash)
    return ("<obj>", type(v).__name__)

def make_scenario_signature(sc: Dict[str, Any]) -> tuple:
    # exclude 'name' so identical scenarios with different labels dedup
    keys = ("kind", "overrides", "expectations")
    parts = []
    for k in keys:
        if k in sc:
            parts.append((k, _freeze_for_signature(sc[k])))
    return tuple(parts)


def _canonical_env_selector(sel: str) -> str:
    m = re.match(r"^os\.environ\.(.+)$", sel)
    if m:
        return f"ENV.{m.group(1)}"
    m = re.match(r"^os\.getenv\.(.+)$", sel)
    if m:
        return f"ENV.{m.group(1)}"
    return sel


def _flatten_attribute_chain(node: Any) -> Optional[List[str]]:
    if node is None:
        return None
    parts: List[str] = []
    cur: Any = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _safe_unparse(node: Any, max_len: int = 60) -> str:
    if node is None:
        return "None"

    lit = literal_value(node)
    if lit is not _UNSUPPORTED:
        s = repr(lit)
        return s[:max_len] + ("..." if len(s) > max_len else "")

    if isinstance(node, ast.Name):
        return node.id

    try:
        text = ast.unparse(node)
        return text[:max_len] + ("..." if len(text) > max_len else "")
    except (AttributeError, TypeError, ValueError):
        pass

    if isinstance(node, ast.Attribute):
        chain = _flatten_attribute_chain(node)
        if chain:
            joined = ".".join(chain)
            return joined[:max_len] + ("..." if len(joined) > max_len else "")

    return "<expr>"


# ===========================================================
# AST extraction
# ===========================================================

class BranchExtractor(ast.NodeVisitor):
    def __init__(self):
        self.branches: Dict[str, BranchDim] = {}
        self.cur_ctx_stack: List[str] = []
        self.defaults: Dict[str, Any] = {}
        self.inferred_values: Dict[str, Set[Any]] = defaultdict(set)

    def push_ctx(self, label: str):
        self.cur_ctx_stack.append(label)

    def pop_ctx(self):
        self.cur_ctx_stack.pop()

    def current_context(self) -> str:
        return ".".join(self.cur_ctx_stack) if self.cur_ctx_stack else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.push_ctx(node.name)
        self.generic_visit(node)
        self.pop_ctx()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.push_ctx(node.name)
        self.generic_visit(node)
        self.pop_ctx()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.push_ctx(node.name)
        self.generic_visit(node)
        self.pop_ctx()

    def visit_Assign(self, node: ast.Assign):
        val = literal_value(node.value)
        if val is not _UNSUPPORTED:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    self.defaults[tgt.id] = val
                elif isinstance(tgt, ast.Attribute):
                    sel = render_selector(tgt)
                    if sel:
                        self.defaults[sel] = val

        self._maybe_record_enum_like(node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        ctx = self.current_context()
        current = node
        chain_selectors = set()

        while True:
            body_summary, body_details = analyze_branch_body(current.body)
            selvals = analyze_test(current.test)

            all_pairs = set(selvals) if len(selvals) > 1 else set()

            for raw_sel, val in selvals:
                sel = _canonical_env_selector(raw_sel)
                chain_selectors.add(sel)

                dim = self.branches.get(sel)
                if dim is None:
                    dim = BranchDim(selector=sel)
                    self.branches[sel] = dim

                other_conditions: Set[Tuple[str, Any]] = set()
                for (alt_sel, alt_val) in all_pairs:
                    alt_norm = _canonical_env_selector(alt_sel)
                    if not (alt_norm == sel and alt_val == val):
                        other_conditions.add((alt_norm, alt_val))

                dim.cases.append(
                    BranchCase(
                        value=val,
                        ctx=f"{ctx}@L{current.lineno}",
                        body_summary=body_summary or "no_signal",
                        body_details=body_details,
                        co_conditions=other_conditions,
                    )
                )

                if val != "__fallback__":
                    self.inferred_values[sel].add(val)

            if (
                    current.orelse
                    and len(current.orelse) == 1
                    and isinstance(current.orelse[0], ast.If)
            ):
                current = current.orelse[0]
                continue

            if current.orelse:
                else_summary, else_details = analyze_branch_body(current.orelse)
                if len(chain_selectors) == 1:
                    only_selector = next(iter(chain_selectors))
                    dim = self.branches.get(only_selector)
                    if dim is None:
                        dim = BranchDim(selector=only_selector)
                        self.branches[only_selector] = dim

                    dim.cases.append(
                        BranchCase(
                            value="__fallback__",
                            ctx=f"{ctx}@L{current.lineno}_else",
                            body_summary=else_summary or "no_signal",
                            body_details=else_details,
                            co_conditions=set(),
                        )
                    )

            break

        self.generic_visit(node)

    def _maybe_record_enum_like(self, node: ast.Assign):
        if len(node.targets) != 1:
            return
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return

        var_name = target.id
        if not re.match(r"^[A-Z0-9_]+$", var_name):
            return

        if isinstance(node.value, (ast.Tuple, ast.List, ast.Set)):
            vals: Set[Any] = set()
            for elt in node.value.elts:
                v = literal_value(elt)
                if v is not _UNSUPPORTED:
                    vals.add(v)
            if vals:
                self.inferred_values[f"enum_hint.{var_name}"].update(vals)


def analyze_branch_body(body: List[ast.stmt]) -> Tuple[str, Dict[str, Any]]:
    raises_exc = None
    raises_msg = None
    ret_preview = None
    logs_levels: Set[str] = set()
    control_flows: Set[str] = set()

    for stmt in body:
        if isinstance(stmt, ast.Raise):
            exc_node = stmt.exc
            if isinstance(exc_node, ast.Call):
                if isinstance(exc_node.func, ast.Name):
                    raises_exc = exc_node.func.id
                if exc_node.args:
                    raises_msg = _safe_unparse(exc_node.args[0], max_len=60)
            else:
                if isinstance(exc_node, ast.Name):
                    raises_exc = exc_node.id
            if raises_exc is None:
                raises_exc = "Exception"

        if isinstance(stmt, ast.Return):
            ret_preview = _safe_unparse(stmt.value, max_len=60)

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute):
                if func.attr in ["error", "warning", "debug", "info", "critical"]:
                    logs_levels.add(func.attr)

        if isinstance(stmt, ast.Continue):
            control_flows.add("continue")
        if isinstance(stmt, ast.Break):
            control_flows.add("break")

    tokens: List[str] = []
    if raises_exc:
        tokens.append(f"raises_{raises_exc}")
    if ret_preview not in [None, "None"]:
        tokens.append(f"returns_{ret_preview}")
    for lvl in sorted(logs_levels):
        tokens.append(f"logs_{lvl}")
    for cf in sorted(control_flows):
        tokens.append("continues" if cf == "continue" else "breaks")

    if not tokens:
        tokens.append("no_signal")

    body_summary = ",".join(tokens)

    body_details = {
        "raises": {"exc": raises_exc, "msg": raises_msg} if raises_exc else None,
        "returns": ret_preview if ret_preview not in [None, "None"] else None,
        "logs": sorted(logs_levels) if logs_levels else [],
        "control": sorted(control_flows) if control_flows else [],
    }

    return body_summary, body_details


def analyze_test(test: Any) -> List[Tuple[str, Any]]:
    if test is None:
        return []
    out: List[Tuple[str, Any]] = []

    if isinstance(test, ast.BoolOp):
        for value in test.values:
            out.extend(analyze_test(value))
        return out

    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = analyze_test(test.operand)
        flipped: List[Tuple[str, Any]] = []
        for sel, val in inner:
            if isinstance(val, bool):
                flipped.append((sel, not val))
            else:
                flipped.append((sel, val))
        return flipped

    if isinstance(test, ast.Name):
        out.append((test.id, True))
        out.append((test.id, False))
        return out

    if isinstance(test, ast.Attribute):
        selector = render_selector(test)
        if selector:
            out.append((selector, True))
            out.append((selector, False))
            return out

    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        left = test.left
        right = test.comparators[0] if test.comparators else None

        if isinstance(op, (ast.Eq, ast.NotEq)):
            selector = render_selector(left)
            value = literal_value(right)
            if selector and value is not _UNSUPPORTED:
                out.append((selector, value))
                if isinstance(op, ast.NotEq) and isinstance(value, bool):
                    out.append((selector, not value))
                return out

        if isinstance(op, ast.In):
            selector = render_selector(left)
            values = list_values(right)
            if selector and values:
                for v in values:
                    out.append((selector, v))
                return out

        if isinstance(op, ast.NotIn):
            selector = render_selector(left)
            values = list_values(right)
            if selector and values:
                for v in values:
                    out.append((selector, f"not_{v}"))
                return out

        if isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            selector = render_selector(left)
            threshold = literal_value(right)
            if selector and threshold is not _UNSUPPORTED:
                op_name = op.__class__.__name__.lower()
                out.append((f"{selector}_{op_name}_{threshold}", True))
                return out

        if isinstance(op, (ast.Is, ast.IsNot)):
            selector = render_selector(left)
            value = literal_value(right)
            if selector and value is not _UNSUPPORTED:
                out.append((selector, value))
                return out

    if isinstance(test, ast.Call):
        if isinstance(test.func, ast.Name):
            fname = test.func.id
            if fname in ["hasattr", "isinstance", "callable"] and test.args:
                obj_sel = render_selector(test.args[0])
                if obj_sel:
                    out.append((f"{fname}_{obj_sel}", True))
                    out.append((f"{fname}_{obj_sel}", False))
                    return out

    return out


def render_selector(node: Any) -> Optional[str]:
    if node is None:
        return None

    if isinstance(node, ast.Name):
        return _canonical_env_selector(node.id)

    if isinstance(node, ast.Attribute):
        chain = _flatten_attribute_chain(node)
        if chain:
            base = ".".join(chain)
            return _canonical_env_selector(base)
        return None

    if isinstance(node, ast.Subscript):
        base_sel = render_selector(node.value)
        key = subscript_key(node.slice)
        if base_sel and key:
            if base_sel in ("os.environ", "os.getenv"):
                return _canonical_env_selector(f"{base_sel}.{key}")
            return _canonical_env_selector(f"{base_sel}.{key}")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "getenv"
                    and node.args
            ):
                env_key = literal_value(node.args[0])
                if env_key is not _UNSUPPORTED:
                    return _canonical_env_selector(f"os.getenv.{env_key}")

            if node.func.attr in ["get", "get_setting", "getattr"] and node.args:
                base_sel = render_selector(node.func.value)
                key = literal_value(node.args[0])
                if base_sel and key is not _UNSUPPORTED:
                    return _canonical_env_selector(f"{base_sel}.{key}")

        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if (
                    fname.startswith("is_")
                    or fname.startswith("has_")
                    or fname.endswith("_enabled")
                    or fname.endswith("_available")
                    or fname.endswith("_disabled")
            ):
                return _canonical_env_selector(fname)

    return None


def subscript_key(sl: Any) -> Optional[str]:
    if sl is None:
        return None
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value
    return None


def literal_value(node: Any) -> Any:
    if node is None:
        return _UNSUPPORTED
    if isinstance(node, ast.Constant):
        return node.value
    return _UNSUPPORTED


def list_values(node: Any) -> List[Any]:
    vals: List[Any] = []
    if node is None:
        return vals
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            v = literal_value(elt)
            if v is not _UNSUPPORTED:
                vals.append(v)
    return vals


# ===========================================================
# Scenario classification and expectation synthesis
# ===========================================================

def infer_kind(selector: str, value: Any, body_details: Dict[str, Any]) -> str:
    sel_lower = str(selector).lower()
    val_lower = str(value).lower()

    errorish_signal = (
            body_details.get("raises") is not None
            or any(level in ["error", "critical", "warning"]
                   for level in body_details.get("logs", []))
    )

    disabled_words = any(
        bad_word in sel_lower or bad_word in val_lower
        for bad_word in ["error", "fail", "bad", "invalid", "disabled"]
    )

    if errorish_signal or disabled_words:
        return "ensemble_all_bad"
    if "debug" in sel_lower or "verbose" in sel_lower:
        return "ensemble_loop"
    return "ensemble_loop"


def build_expectations(
        overrides: Dict[str, Any],
        kind: str,
        body_details: Dict[str, Any],
        co_conditions: Set[Tuple[str, Any]],
) -> Dict[str, Any]:
    expectations: Dict[str, Any] = {}
    assertions: Dict[str, Any] = {}

    rinfo = body_details.get("raises")
    if rinfo:
        assertions["raises"] = rinfo["exc"]
        if rinfo.get("msg"):
            assertions["message_contains"] = rinfo["msg"]

    ret_val = body_details.get("returns")
    if ret_val:
        assertions["returns_like"] = ret_val

    logs_levels = body_details.get("logs") or []
    if logs_levels:
        assertions["logs_levels"] = logs_levels

    if assertions:
        expectations["assert"] = assertions

    hint_parts: List[str] = []
    if co_conditions:
        co_str = ", ".join(f"{s}={v}" for (s, v) in sorted(co_conditions))
        hint_parts.append(f"requires: {co_str}")

    if kind == "ensemble_all_bad":
        if rinfo:
            expectations["TODO"] = f"verify {rinfo['exc']} with proper message/handling"
        else:
            expectations["TODO"] = "verify error behavior and guardrails"
    else:
        if rinfo:
            expectations["TODO"] = f"ensure {rinfo['exc']} is raised correctly"
        elif ret_val:
            expectations["TODO"] = "verify return value matches contract"
        elif logs_levels:
            expectations["TODO"] = f"check {'/'.join(logs_levels)} logs are emitted"
        else:
            expectations["TODO"] = "verify behavior for this configuration"

    if hint_parts:
        expectations["_hint"] = "; ".join(hint_parts)

    return expectations


# ===========================================================
# LLM integration with caching
# ===========================================================

class LLMCache:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".scenario_cache")
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def _cache_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{self._cache_key(prompt)}.json"
        if not cache_file.exists():
            return None

        try:
            raw = cache_file.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, prompt: str, result: Dict[str, Any]):
        cache_file = self.cache_dir / f"{self._cache_key(prompt)}.json"
        cache_file.write_text(json.dumps(result, indent=2))


def llm_refine_expectations(
        expectations: Dict[str, Any],
        scenario: Dict[str, Any],
        harness_schema: Dict[str, Any],
        llm_cache: Optional[LLMCache] = None,
) -> Dict[str, Any]:
    """
    Call Claude API to enrich expectations with harness-specific assertions.
    Uses caching to avoid redundant API calls.
    """

    # Build prompt context
    overrides = scenario.get("overrides", {})
    behavior = scenario.get("_branch_behavior", [])
    kind = scenario.get("kind", "unknown")

    prompt = f"""Given this test scenario configuration:

Overrides: {json.dumps(overrides, indent=2)}
Branch behavior: {', '.join(behavior)}
Scenario kind: {kind}
Current expectations: {json.dumps(expectations, indent=2)}

Harness schema excerpt:
{json.dumps({k: v for k, v in harness_schema.items() if k in overrides}, indent=2)}

Enrich the expectations with specific assertions that match the harness contract.
Focus on:
1. Concrete field checks for return values
2. Specific log message patterns
3. State machine invariants
4. Side effect validation

Return only a JSON object with the enhanced expectations."""

    if llm_cache:
        cached = llm_cache.get(prompt)
        if cached:
            return cached

    try:
        # Real API call (requires ANTHROPIC_API_KEY env var)
        try:
            import anthropic
            HAS_ANTHROPIC = True
        except ImportError:
            anthropic = None
            HAS_ANTHROPIC = False

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract JSON from response
        content = response.content[0].text

        # Try to parse JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?})\s*```', content, re.DOTALL)
        if json_match:
            enriched = json.loads(json_match.group(1))
        else:
            # Try direct parse
            enriched = json.loads(content)

        if llm_cache:
            llm_cache.set(prompt, enriched)

        return enriched

    except Exception as e:
        # Graceful fallback
        print(f"Warning: LLM enrichment failed: {e}", file=sys.stderr)
        expectations["_llm_error"] = str(e)
        return expectations


def _scenario_name_from_overrides(overrides: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in sorted(overrides.items()):
        v_label = "other" if v == "__fallback__" else v
        parts.append(f"{k}={v_label}")
    return " & ".join(parts)


# ===========================================================
# Scenario candidate generation
# ===========================================================

def make_scenario_candidates(
        branches: Dict[str, BranchDim],
        defaults: Dict[str, Any],
        generate_combos: bool,
) -> List[Dict[str, Any]]:
    for selector, dim in branches.items():
        base_name = selector.split(".")[0]
        if dim.default_value is None and base_name in defaults:
            dim.default_value = defaults[base_name]
        if dim.default_value is None and selector in defaults:
            dim.default_value = defaults[selector]

    candidates: List[Dict[str, Any]] = []

    for selector in sorted(branches.keys()):
        dim = branches[selector]

        for case in dim.cases:
            overrides_single = {selector: case.value}
            kind_single = infer_kind(selector, case.value, case.body_details)
            expectations_single = build_expectations(
                overrides_single,
                kind_single,
                case.body_details,
                case.co_conditions,
            )

            scenario_single: Dict[str, Any] = {
                "name": _scenario_name_from_overrides(overrides_single),
                "kind": kind_single,
                "overrides": overrides_single,
                "expectations": expectations_single,
                "_discovered_from": [case.ctx],
                "_branch_behavior": [case.body_summary],
            }

            if dim.default_value is not None and case.value == dim.default_value:
                scenario_single["_is_default"] = True
            if case.value == "__fallback__":
                scenario_single["_is_fallback"] = True
            if case.co_conditions:
                scenario_single["_co_conditions"] = sorted([tuple(x) for x in case.co_conditions])

            candidates.append(scenario_single)

            if generate_combos and case.co_conditions:
                combo_overrides = {selector: case.value}
                for (cc_sel, cc_val) in case.co_conditions:
                    combo_overrides[cc_sel] = cc_val

                kind_combo = kind_single
                expectations_combo = build_expectations(
                    combo_overrides,
                    kind_combo,
                    case.body_details,
                    set(),
                )

                scenario_combo: Dict[str, Any] = {
                    "name": _scenario_name_from_overrides(combo_overrides),
                    "kind": kind_combo,
                    "overrides": combo_overrides,
                    "expectations": expectations_combo,
                    "_discovered_from": [case.ctx],
                    "_branch_behavior": [case.body_summary],
                    "_is_combo": True,
                }

                candidates.append(scenario_combo)

    return candidates


# ===========================================================
# Cartesian product with smart risk scoring
# ===========================================================

def score_combo_for_risk(overrides: Dict[str, Any], policy: PolicyConfig) -> int:
    score = 0

    # Base score from text patterns
    for key, val in overrides.items():
        text = f"{key}={val}".lower()

        # High-risk patterns
        if any(word in text for word in ["prod", "production", "unsafe", "public"]):
            score += 10
        if any(word in text for word in ["auth", "admin", "privilege"]):
            score += 8
        if any(word in text for word in ["delete", "drop", "destroy"]):
            score += 7
        if any(word in text for word in ["network", "external", "remote"]):
            score += 5

        # Medium-risk patterns
        if any(word in text for word in ["strict", "validate", "check"]):
            score += 3
        if any(word in text for word in ["debug", "verbose", "trace"]):
            score += 2

    # Policy-based risk weights
    for key, val in overrides.items():
        weight_key = f"{key}={val}"
        if weight_key in policy.risk_weights:
            score += policy.risk_weights[weight_key]

    # Combo complexity score
    score += len(overrides) * 2

    return score


def violates_exclusions(overrides: Dict[str, Any], policy: PolicyConfig) -> bool:
    for rule in policy.exclude_combos:
        if all(overrides.get(k) == v for k, v in rule.items()):
            return True

    for mutex in policy.mutually_exclusive:
        if all(overrides.get(k) == v for k, v in mutex.items()):
            return True

    return False


def generate_cartesian_candidates(
        inferred_values: Dict[str, Set[Any]],
        base_candidates: List[Dict[str, Any]],
        schema: Dict[str, Dict[str, Any]],
        policy: PolicyConfig,
) -> List[Dict[str, Any]]:
    selector_values: Dict[str, Set[Any]] = {}
    for sel, vals in inferred_values.items():
        if sel in schema and vals:
            selector_values[sel] = set(vals)

    covered_sigs = {
        make_overrides_signature(sc["overrides"])
        for sc in base_candidates
    }

    proposed: List[Tuple[Dict[str, Any], int]] = []
    sels = sorted(selector_values.keys())

    for i in range(len(sels)):
        for j in range(i + 1, len(sels)):
            a = sels[i]
            b = sels[j]
            for va in selector_values[a]:
                for vb in selector_values[b]:
                    combo = {a: va, b: vb}
                    sig = tuple(sorted(combo.items()))

                    if sig in covered_sigs:
                        continue
                    if violates_exclusions(combo, policy):
                        continue

                    risk = score_combo_for_risk(combo, policy)
                    proposed.append((combo, risk))

    proposed.sort(key=lambda x: x[1], reverse=True)

    max_cart = policy.max_cartesian
    if max_cart >= 0:
        proposed = proposed[:max_cart]

    cart_scenarios: List[Dict[str, Any]] = []
    for combo, risk in proposed:
        scenario = {
            "name": _scenario_name_from_overrides(combo),
            "kind": "ensemble_loop",
            "overrides": combo,
            "expectations": {
                "TODO": "verify interaction between flags (cartesian)",
                "_hint": f"cartesian-generated; risk_score={risk}",
            },
            "_is_combo": True,
            "_cartesian": True,
            "_risk_score": risk,
        }
        cart_scenarios.append(scenario)

    return cart_scenarios


# ===========================================================
# Deduplication and merging
# ===========================================================

def _merge_expectations(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    if "assert" in src:
        if "assert" not in dst:
            dst["assert"] = {}
        for k, v in src["assert"].items():
            if k not in dst["assert"]:
                dst["assert"][k] = v

    if "TODO" in src:
        if "TODO" not in dst:
            dst["TODO"] = src["TODO"]
        elif src["TODO"] != dst["TODO"] and src["TODO"] not in dst["TODO"]:
            dst["TODO"] = dst["TODO"] + " / " + src["TODO"]

    if "_hint" in src:
        if "_hint" not in dst:
            dst["_hint"] = src["_hint"]
        elif src["_hint"] not in dst["_hint"]:
            dst["_hint"] = dst["_hint"] + " | " + src["_hint"]

    return dst


def deduplicate_scenarios(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[tuple] = set()
    out: List[Dict[str, Any]] = []
    for sc in scenarios:
        try:
            sig = make_scenario_signature(sc)
        except Exception:
            # ultra-safe fallback: still avoid crashing on exotic values
            sig = ("fallback",)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(sc)
    return out


# ===========================================================
# Schema normalization and validation
# ===========================================================

def normalize_overrides_to_harness(
        overrides: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    norm: Dict[str, Any] = {}
    errors: List[str] = []

    for sel, val in overrides.items():
        if sel not in schema:
            errors.append(f"unknown selector {sel} (not in schema)")
            continue

        entry = schema[sel]
        inject_key = entry["inject_as"]
        allowed = entry.get("type")

        if isinstance(allowed, set):
            if val not in allowed:
                errors.append(f"{sel}={val} not in allowed values {sorted(list(allowed))}")
                continue
        elif isinstance(allowed, type):
            if not isinstance(val, allowed):
                errors.append(f"{sel}={val} type mismatch (expected {allowed.__name__})")
                continue

        if callable(entry.get("coerce")):
            val = entry["coerce"](val)

        norm[inject_key] = val

    return norm, errors


def apply_schema_normalization(
        scenarios: List[Dict[str, Any]],
        schema: Dict[str, Dict[str, Any]],
        strict_schema: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for sc in scenarios:
        norm, errs = normalize_overrides_to_harness(sc["overrides"], schema)
        sc["overrides_normalized"] = norm

        if errs:
            sc["_schema_errors"] = errs
            if strict_schema:
                continue

        out.append(sc)

    return out


# ===========================================================
# Negative space analysis
# ===========================================================

def find_negative_space(
        branches: Dict[str, BranchDim],
        inferred_values: Dict[str, Set[Any]],
        schema: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []

    seen_values: Dict[str, Set[Any]] = defaultdict(set)
    for selector, dim in branches.items():
        for case in dim.cases:
            if case.value != "__fallback__":
                seen_values[selector].add(case.value)

    for selector, schema_entry in schema.items():
        allowed = schema_entry.get("type")
        if isinstance(allowed, set):
            for v in allowed:
                inferred_values[selector].add(v)

    for selector, allowed_vals in inferred_values.items():
        if selector.startswith("enum_hint."):
            continue

        missing = allowed_vals - seen_values.get(selector, set())
        for m in sorted(missing, key=lambda x: str(x)):
            gaps.append({
                "selector": selector,
                "missing_value": m,
                "message": f"{selector} can be {m!r} (allowed/inferred) but no explicit branch handles it",
            })

    return gaps


def generate_negative_space_scenarios(gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for g in gaps:
        selector = g["selector"]
        value = g["missing_value"]
        overrides = {selector: value}

        scenario = {
            "name": f"{selector}={value} (UNHANDLED)",
            "kind": "ensemble_loop",
            "overrides": overrides,
            "expectations": {
                "TODO": "IMPLEMENT MISSING HANDLING OR EXPLAIN WHY IMPOSSIBLE",
                "_hint": "negative-space: no branch for this value",
            },
            "_negative_space": True,
        }
        out.append(scenario)

    return out


# ===========================================================
# Behavioral snapshotting with subprocess isolation
# ===========================================================

def run_harness_safely(
        overrides_normalized: Dict[str, Any],
        harness_command: Optional[str],
        timeout: float,
) -> Dict[str, Any]:
    """
    Execute harness in subprocess with timeout and capture behavior.
    Returns deterministic snapshot of outputs.
    """
    if not harness_command:
        # Stub fallback for when no harness configured
        return {
            "status": "ok",
            "note": "no harness command configured",
        }

    try:
        env = {**os.environ, **{k: str(v) for k, v in overrides_normalized.items()}}

        result = subprocess.run(
            harness_command.split(),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        snapshot = {
            "exit_code": result.returncode,
            "stdout_lines": result.stdout.strip().split('\n') if result.stdout else [],
            "stderr_lines": result.stderr.strip().split('\n') if result.stderr else [],
        }

        # Strip non-determinism
        snapshot = sanitize_snapshot(snapshot)

        return snapshot

    except subprocess.TimeoutExpired:
        return {"error": "timeout", "timeout_seconds": timeout}
    except Exception as e:
        return {"error": str(e)}


def sanitize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Remove timestamps, UUIDs, random IDs from snapshot for deterministic comparison"""
    import re

    def clean_line(line: str) -> str:
        # Remove timestamps
        line = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', line)
        # Remove UUIDs
        line = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', line)
        # Remove hex addresses
        line = re.sub(r'0x[0-9a-f]+', 'ADDR', line)
        return line

    if "stdout_lines" in snapshot:
        snapshot["stdout_lines"] = [clean_line(l) for l in snapshot["stdout_lines"]]
    if "stderr_lines" in snapshot:
        snapshot["stderr_lines"] = [clean_line(l) for l in snapshot["stderr_lines"]]

    return snapshot


def make_snapshot_assertions(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw snapshot to assertion format"""
    assertions = {}

    if "exit_code" in snapshot:
        assertions["exit_code"] = snapshot["exit_code"]

    if snapshot.get("stdout_lines"):
        assertions["stdout_contains"] = snapshot["stdout_lines"][:5]  # First 5 lines

    if snapshot.get("stderr_lines"):
        assertions["stderr_contains"] = snapshot["stderr_lines"][:5]

    return assertions


def attach_snapshots(
        scenarios: List[Dict[str, Any]],
        take_snapshot: bool,
        existing_snapshots: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]],
        snapshot_diff: bool,
        snapshot_fail_on_drift: bool,
        harness_command: Optional[str],
        timeout: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    drift_reports: List[Dict[str, Any]] = []

    for sc in scenarios:
        sig = make_overrides_signature(sc["overrides"])

        if take_snapshot:
            overrides_to_apply = sc.get("overrides_normalized", sc["overrides"])
            snap_raw = run_harness_safely(overrides_to_apply, harness_command, timeout)
            snap_asserts = make_snapshot_assertions(snap_raw)

            sc.setdefault("expectations", {})
            sc["expectations"]["snapshot"] = snap_asserts
            sc["_snapshot_taken"] = True

        if snapshot_diff:
            prev_snap = existing_snapshots.get(sig)
            new_snap = sc.get("expectations", {}).get("snapshot")

            if prev_snap is not None and new_snap is not None:
                drift = compare_snapshots(prev_snap, new_snap)
                if drift:
                    sc["_behavior_drift"] = drift
                    drift_reports.append({
                        "scenario_name": sc["name"],
                        "diff": drift,
                    })

    return scenarios, drift_reports


def compare_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Compute diff between snapshots"""
    drift: Dict[str, Any] = {}
    keys = set(old.keys()) | set(new.keys())

    for k in sorted(keys):
        ov = old.get(k, "<missing>")
        nv = new.get(k, "<missing>")

        if ov != nv:
            drift[k] = {"old": ov, "new": nv}

    return drift


# ===========================================================
# Coverage metrics and trending
# ===========================================================

def compute_coverage_metrics(
        all_candidates: List[Dict[str, Any]],
        existing_sigs: Set[Tuple[Tuple[str, Any], ...]],
        final_scenarios: List[Dict[str, Any]],
) -> Tuple[float, float]:
    discovered_sigs = {
        make_overrides_signature(sc["overrides"])
        for sc in all_candidates
    }

    covered_sigs = set(existing_sigs)
    for sc in final_scenarios:
        covered_sigs.add(make_overrides_signature(sc["overrides"]))

    if discovered_sigs:
        branch_cov = len(discovered_sigs & covered_sigs) / len(discovered_sigs)
    else:
        branch_cov = 1.0

    discovered_interaction = {
        make_overrides_signature(sc["overrides"])
        for sc in all_candidates
        if sc.get("_is_combo") or sc.get("_cartesian")
    }

    if discovered_interaction:
        covered_interaction = covered_sigs & discovered_interaction
        inter_cov = len(covered_interaction) / len(discovered_interaction)
    else:
        inter_cov = 1.0

    return branch_cov, inter_cov


def emit_coverage_json(
        path: Optional[str],
        module_filename: str,
        stats: ScenarioStats,
):
    """Persist coverage metrics for trending"""
    if not path:
        return

    payload = {
        "module": module_filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch_coverage_pct": round(stats.branch_coverage_pct * 100, 2) if stats.branch_coverage_pct else None,
        "interaction_coverage_pct": round(stats.interaction_coverage_pct * 100, 2) if stats.interaction_coverage_pct else None,
        "total_scenarios": stats.total_scenarios,
        "new_scenarios": stats.new_scenarios,
        "selectors_count": len(stats.selectors),
        "schema_errors": stats.schema_errors,
        "drift_count": stats.drift_count,
    }

    Path(path).write_text(json.dumps(payload, indent=2))


# ===========================================================
# Existing scenario file parsing
# ===========================================================

def _extract_overrides_sig(d: ast.Dict) -> Optional[Tuple[Tuple[str, Any], ...]]:
    found_overrides = None
    for k_node, v_node in zip(d.keys, d.values):
        key_is_str = isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)
        if key_is_str and k_node.value == "overrides":
            found_overrides = v_node
            break

    if found_overrides is None or not isinstance(found_overrides, ast.Dict):
        return None

    overrides_map: Dict[str, Any] = {}
    for kk_node, vv_node in zip(found_overrides.keys, found_overrides.values):
        if isinstance(kk_node, ast.Constant) and isinstance(kk_node.value, str):
            ov_key = kk_node.value
        else:
            return None

        lit_val = literal_value(vv_node)
        if lit_val is _UNSUPPORTED:
            return None

        overrides_map[ov_key] = lit_val

    return make_overrides_signature(overrides_map)


def _extract_snapshot_from_ast_dict(d: ast.Dict) -> Optional[Dict[str, Any]]:
    expectations_node = None
    for k_node, v_node in zip(d.keys, d.values):
        if isinstance(k_node, ast.Constant) and k_node.value == "expectations":
            expectations_node = v_node
            break

    if not isinstance(expectations_node, ast.Dict):
        return None

    snapshot_node = None
    for kk, vv in zip(expectations_node.keys, expectations_node.values):
        if isinstance(kk, ast.Constant) and kk.value == "snapshot":
            snapshot_node = vv
            break

    if not isinstance(snapshot_node, ast.Dict):
        return None

    snap_map: Dict[str, Any] = {}
    for k2, v2 in zip(snapshot_node.keys, snapshot_node.values):
        if isinstance(k2, ast.Constant) and isinstance(k2.value, str):
            key = k2.value
        else:
            return None

        val = literal_value(v2)
        if val is _UNSUPPORTED:
            continue

        snap_map[key] = val

    return snap_map


def load_existing_signatures_and_snapshots(
    existing_file: Optional[str],
) -> Tuple[Set[Tuple[Tuple[str, Any], ...]], Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]]]:
    # holds unique override signatures we've seen
    existing_sigs: Set[Tuple[Tuple[str, Any], ...]] = set()
    # map signature -> snapshot payload
    existing_snapshots: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}

    # no file provided
    if not existing_file:
        return existing_sigs, existing_snapshots

    # try to read file text; on any IO/encoding issue, treat as empty
    try:
        existing_src = Path(existing_file).read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
        return existing_sigs, existing_snapshots

    # parse as Python; on syntax issues, treat as empty
    try:
        existing_tree = ast.parse(existing_src, filename=existing_file)
    except SyntaxError:
        return existing_sigs, existing_snapshots

    # scan for: generated_scenarios = [ {...}, {...}, ... ]
    for node in existing_tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "generated_scenarios":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Dict):
                                # extract the signature from the dict AST
                                sig = _extract_overrides_sig(elt)
                                if sig is not None:
                                    # FIX: add the extracted sig, not an undefined 'scenario'
                                    existing_sigs.add(sig)
                                    # optionally extract snapshot and record by signature
                                    snap = _extract_snapshot_from_ast_dict(elt)
                                    if snap is not None:
                                        existing_snapshots[sig] = snap

    return existing_sigs, existing_snapshots


def filter_new_scenarios(
        scenarios: List[Dict[str, Any]],
        existing_sigs: Set[Tuple[Tuple[str, Any], ...]],
) -> List[Dict[str, Any]]:
    new_list: List[Dict[str, Any]] = []

    for sc in scenarios:
        sig = make_overrides_signature(sc["overrides"])
        if sig not in existing_sigs:
            sc["_is_new"] = True
            new_list.append(sc)

    return new_list


# ===========================================================
# Stats and formatting
# ===========================================================

def build_stats(
        scenarios: List[Dict[str, Any]],
        before_merge: int,
        after_merge: int,
        branch_cov: Optional[float],
        inter_cov: Optional[float],
        negative_space_gaps: List[Dict[str, Any]],
) -> ScenarioStats:
    stats = ScenarioStats()
    stats.total_scenarios = len(scenarios)
    stats.merged_duplicates = before_merge - after_merge
    stats.combo_scenarios = sum(1 for sc in scenarios if sc.get("_is_combo") or sc.get("_cartesian"))
    stats.new_scenarios = sum(1 for sc in scenarios if sc.get("_is_new"))
    stats.branch_coverage_pct = branch_cov
    stats.interaction_coverage_pct = inter_cov
    stats.negative_space_gaps = negative_space_gaps
    stats.schema_errors = sum(1 for sc in scenarios if sc.get("_schema_errors"))
    stats.snapshots_taken = sum(1 for sc in scenarios if sc.get("_snapshot_taken"))
    stats.drift_count = sum(1 for sc in scenarios if sc.get("_behavior_drift"))

    for sc in scenarios:
        for k in sc["overrides"].keys():
            stats.selectors.add(k)
        stats.by_kind[sc["kind"]] += 1
        if sc.get("_is_default"):
            stats.with_defaults += 1
        if sc.get("_is_fallback"):
            stats.with_fallbacks += 1

    return stats


def format_scenarios_output(
        scenarios: List[Dict[str, Any]],
        include_metadata: bool = True,
        include_normalized: bool = True,
) -> str:
    lines: List[str] = []
    lines.append("# --- BEGIN AUTO-GENERATED SCENARIOS ---")
    lines.append("generated_scenarios = [")

    for sc in scenarios:
        lines.append("    {")

        core_keys = ["name", "kind", "overrides", "expectations"]
        if include_normalized and "overrides_normalized" in sc:
            core_keys.insert(3, "overrides_normalized")

        for key in core_keys:
            if key in sc:
                lines.append(f"        {repr(key)}: {repr(sc[key])},")

        if include_metadata:
            meta_keys = [
                "_is_new", "_is_default", "_is_fallback", "_is_combo",
                "_cartesian", "_negative_space", "_co_conditions",
                "_discovered_from", "_branch_behavior", "_schema_errors",
                "_snapshot_taken", "_behavior_drift", "_risk_score",
            ]
            for key in meta_keys:
                if key in sc:
                    lines.append(f"        {repr(key)}: {repr(sc[key])},")

        lines.append("    },")

    lines.append("]")
    lines.append("# --- END AUTO-GENERATED SCENARIOS ---")
    return "\n".join(lines)


def format_stats(
        stats: ScenarioStats,
        filename: str,
        filtered: bool,
        drift_reports: List[Dict[str, Any]],
) -> str:
    lines = [
        f"\n# Scenario Generation Summary for {filename}",
        f"# Total scenarios emitted: {stats.total_scenarios}",
    ]

    if filtered:
        lines.append("# (filtered against existing; these are NEW signatures only)")

    lines.append(f"# New scenarios: {stats.new_scenarios}")
    lines.append(f"# Unique selectors: {len(stats.selectors)}")

    if stats.by_kind:
        lines.append("# By kind:")
        for kind in sorted(stats.by_kind.keys()):
            lines.append(f"#   {kind}: {stats.by_kind[kind]}")

    if stats.combo_scenarios > 0:
        lines.append(f"# Multi-condition combos: {stats.combo_scenarios}")
    if stats.with_defaults > 0:
        lines.append(f"# Default-value scenarios: {stats.with_defaults}")
    if stats.with_fallbacks > 0:
        lines.append(f"# Fallback branches: {stats.with_fallbacks}")
    if stats.merged_duplicates > 0:
        lines.append(f"# Duplicates merged: {stats.merged_duplicates}")

    if stats.branch_coverage_pct is not None:
        pct = round(stats.branch_coverage_pct * 100, 2)
        lines.append(f"# Branch coverage (surface): {pct}%")
    if stats.interaction_coverage_pct is not None:
        pct2 = round(stats.interaction_coverage_pct * 100, 2)
        lines.append(f"# Interaction coverage: {pct2}%")

    if stats.schema_errors > 0:
        lines.append(f"# Schema validation errors: {stats.schema_errors}")
    if stats.snapshots_taken > 0:
        lines.append(f"# Snapshots captured: {stats.snapshots_taken}")
    if stats.drift_count > 0:
        lines.append(f"# Behavior drifts detected: {stats.drift_count}")

    if stats.negative_space_gaps:
        lines.append("# Negative space gaps (unhandled config values):")
        for gap in stats.negative_space_gaps[:5]:  # Show first 5
            lines.append(f"#   {gap['selector']} missing {gap['missing_value']!r}")
        if len(stats.negative_space_gaps) > 5:
            lines.append(f"#   ... and {len(stats.negative_space_gaps) - 5} more")

    if drift_reports:
        lines.append("# Behavior drift detected:")
        for dr in drift_reports[:5]:  # Show first 5
            lines.append(f"#   {dr['scenario_name']}: {len(dr['diff'])} fields changed")
        if len(drift_reports) > 5:
            lines.append(f"#   ... and {len(drift_reports) - 5} more drifts")

    return "\n".join(lines)


# ===========================================================
# High-yield negative-path helpers (wired for harness switches)
# ===========================================================

class ExceptionPathExtractor(ast.NodeVisitor):
    # collect raises and enclosing guards for targeted error cases
    def __init__(self):
        self.error_paths = []
        self.validation_guards = []
        self._if_stack = []

    def visit_If(self, node: ast.If):
        try:
            guard = _safe_unparse(node.test)
        except (AttributeError, TypeError, ValueError):
            guard = None
        self._if_stack.append(guard)
        for s in node.body:
            self.visit(s)  # type: ignore[arg-type]
        for s in (getattr(node, "orelse", []) or []):
            self.visit(s)  # type: ignore[arg-type]
        self._if_stack.pop()

    def visit_Raise(self, node: ast.Raise):
        exc_type = None
        exc_msg = None
        exc = getattr(node, "exc", None)

        if isinstance(exc, ast.Call):
            if isinstance(exc.func, ast.Name):
                exc_type = exc.func.id
            if exc.args:
                try:
                    exc_msg = _safe_unparse(exc.args[0])
                except (AttributeError, TypeError, ValueError, IndexError):
                    exc_msg = None
        guard = self._if_stack[-1] if self._if_stack else None
        self.error_paths.append({
            "exc_type": exc_type or "Exception",
            "exc_msg": exc_msg,
            "guard_condition": guard,
            "lineno": getattr(node, "lineno", -1),
        })

    def visit_Assert(self, node: ast.Assert):
        test_str = _safe_unparse(node.test)
        msg = _safe_unparse(node.msg) if node.msg else None
        self.validation_guards.append({
            "assertion": test_str,
            "message": msg,
            "lineno": getattr(node, "lineno", -1),
        })


def _mk(name: str, kind: str, overrides: Dict[str, Any], *, raises: Optional[str] = None, logs: Optional[List[str]] = None) -> Dict[str, Any]:
    # small builder for scenarios with expectations
    expectations: Dict[str, Any] = {}
    if raises:
        expectations["assert"] = {"raises": raises}
    if logs:
        expectations.setdefault("assert", {})
        expectations["assert"]["logs_levels"] = logs
    if not expectations:
        expectations["TODO"] = "verify behavior"
    return {"name": name, "kind": kind, "overrides": overrides, "expectations": expectations}


def generate_factory_abuse_scenarios() -> List[Dict[str, Any]]:
    # flips early RuntimeError via harness factory validators
    return [
        _mk("callable_factory=INVALID", "ensemble_all_bad", {"callable_factory": "not_callable"}, raises="RuntimeError"),
        _mk("isinstance_factory=INVALID", "ensemble_all_bad", {"isinstance_factory": "wrong_type"}, raises="RuntimeError"),
    ]


def generate_lte0_scenarios() -> List[Dict[str, Any]]:
    # ctor-style <=0 guards exposed as self.<name>_lte_0 toggles
    params = ["fast", "slow", "signal", "short", "long"]
    out = []
    for p in params:
        out.append(_mk(f"CTOR_{p}=lte_0", "ensemble_all_bad", {f"self.{p}_lte_0": True}, raises="ValueError"))
    return out


def _invert_guard_to_override(guard: Optional[str], error_path: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # robust mapping of common guard styles -> override knobs the harness understands
    if not guard or not isinstance(guard, str):
        return None
    g = guard.lower()
    g_no_space = re.sub(r"\s+", "", g)

    # Pattern 1: <= 0 on known numeric knobs (self.<name> <= 0 or <name> <= 0)
    m = re.search(r'(?:self\.)?(\w+)\s*<=\s*0', g)
    if m:
        param = m.group(1)
        if param in ["fast", "slow", "signal", "short", "long"]:
            return {f"self.{param}_lte_0": True}

    # Pattern 2: is None / == None
    m_none = re.search(r'(?:self\.)?(\w+)\s*(?:is|==)\s*none', g)
    if m_none:
        param = m_none.group(1)
        if param in ["price_col", "strategy_specs", "weights", "strategies"]:
            return {param: None}

    # Pattern 3: 'not <param>'
    m_not = re.search(r'not\s+(?:self\.)?(\w+)', g)
    if m_not:
        param = m_not.group(1)
        if param in ["strategy_specs"]:
            return {param: False}
        if param in ["price_col"]:
            return {param: None}
        if param in ["strategies", "weights"]:
            return {param: []}

    # Pattern 4: len(param) == 0, <= 0, or attribute '.empty'
    m_len = re.search(r'len\(\s*(?:self\.)?(\w+)\s*\)\s*(?:==|<=)\s*0', g)
    if m_len:
        param = m_len.group(1)
        if param in ["strategies", "weights"]:
            return {param: []}
        if param in ["strategy_specs"]:
            return {param: False}
    if ".empty" in g_no_space:
        for param in ["strategies", "weights"]:
            if param in g_no_space:
                return {param: []}

    return None


def generate_error_path_scenarios(error_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ep in error_paths:
        ov = _invert_guard_to_override(ep.get("guard_condition"), ep)
        if not ov:
            continue
        exc = ep.get("exc_type") or "Exception"
        name = f"ERROR_{exc}_L{ep.get('lineno', -1)}"
        out.append(_mk(name, "ensemble_all_bad", ov, raises=exc))
    return out


def generate_price_col_error_scenarios() -> List[Dict[str, Any]]:
    # exercise ensemble wiring/logging on bad price_col
    return [
        _mk("price_col=None", "ensemble_loop", {"price_col": None}, logs=["error"]),
        _mk("price_col=EMPTY_STRING", "ensemble_loop", {"price_col": ""}, logs=["error"]),
        _mk("price_col=MISSING_COLUMN", "ensemble_loop", {"price_col": "not_a_real_column"}, logs=["error"]),
    ]


def generate_boundary_scenarios_numeric(inferred_values: Dict[str, Set[Any]]) -> List[Dict[str, Any]]:
    # broaden coverage to all numeric selectors with observed ranges
    targets: List[Dict[str, Any]] = []
    for sel, vals in inferred_values.items():
        nums = [v for v in vals if isinstance(v, (int, float))]
        if not nums or len(nums) < 1:
            continue
        mn, mx = min(nums), max(nums)

        # if all observed > 0, try zero/negative to tick failure rails
        if mn > 0:
            targets.append(_mk(f"{sel}=ZERO", "ensemble_all_bad", {sel: 0}, raises="ValueError"))
            targets.append(_mk(f"{sel}=NEGATIVE", "ensemble_all_bad", {sel: -1}, raises="ValueError"))

        # boundary echoes
        targets.append(_mk(f"{sel}=minimum", "ensemble_loop", {sel: mn}))
        targets.append(_mk(f"{sel}=maximum", "ensemble_loop", {sel: mx}))

        # nudge just outside upper boundary for ints
        if isinstance(mx, int):
            targets.append(_mk(f"{sel}=above_max", "ensemble_loop", {sel: mx + 1}))

    return targets


# Additional negative generators for broader coverage

def generate_type_confusion_scenarios(schema: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # pass wrong types against schema-declared types
    scenarios: List[Dict[str, Any]] = []
    wrong_types = {
        int: ["not_an_int", 3.14, None, [], {}],
        str: [42, None, [], {}],
        bool: ["true", 1, None],
        float: ["3.14", None, []],
        list: [None, "[]", {}, 42],
        dict: [None, "{}", [], "dict"],
    }
    for selector, entry in schema.items():
        expected = entry.get("type")
        if isinstance(expected, set) or expected is None:
            continue
        if isinstance(expected, type) and expected in wrong_types:
            for bad in wrong_types.get(cast(Any, expected), []):
                scenarios.append(_mk(f"{selector}=WRONG_TYPE_{type(bad).__name__}", "ensemble_all_bad", {selector: bad}, raises="TypeError"))
    return scenarios


def generate_none_scenarios(schema: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # exercise None handling for most selectors
    scenarios: List[Dict[str, Any]] = []
    for selector in schema.keys():
        if selector.startswith("ENV.") or selector.isupper():
            continue
        scenarios.append(_mk(f"{selector}=None", "ensemble_all_bad", {selector: None}, raises="ValueError"))
    return scenarios


def generate_empty_collection_scenarios() -> List[Dict[str, Any]]:
    # hit empty structures your harness/SUT commonly validates
    return [
        _mk("strategies=EMPTY_LIST", "ensemble_all_bad", {"strategies": []}, raises="ValueError"),
        _mk("weights=EMPTY_LIST", "ensemble_all_bad", {"weights": []}, raises="ValueError"),
        _mk("strategy_specs=EMPTY_FALSE", "ensemble_all_bad", {"strategy_specs": False}, raises="ValueError"),
        _mk("price_col=EMPTY_STRING", "ensemble_loop", {"price_col": ""}, logs=["error"]),
    ]


def generate_disallowed_value_scenarios(schema: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # pick values outside allowed enumerations
    scenarios: List[Dict[str, Any]] = []
    for selector, entry in schema.items():
        allowed = entry.get("type")
        if not isinstance(allowed, set) or not allowed:
            continue
        sample = next(iter(allowed))
        if isinstance(sample, str):
            invalid = "INVALID_VALUE_NOT_IN_SET_12345"
        elif isinstance(sample, int):
            try:
                invalid = max(allowed) + 999
            except (TypeError, ValueError):
                invalid = 10 ** 9
        elif isinstance(sample, bool):
            invalid = "not_a_bool"
        else:
            invalid = object()
        scenarios.append(_mk(f"{selector}=OUT_OF_RANGE", "ensemble_all_bad", {selector: invalid}, raises="ValueError"))
    return scenarios


# ===========================================================
# Main orchestration
# ===========================================================

def scenario_codegen(
        path: str,
        schema: Dict[str, Dict[str, Any]],
        policy: PolicyConfig,
        existing_file: Optional[str],
        generate_combos: bool,
        generate_cartesian: bool,
        strict_schema: bool,
        llm_enrich: bool,
        llm_cache: Optional[LLMCache],
        take_snapshot: bool,
        snapshot_diff: bool,
        snapshot_fail_on_drift: bool,
        include_metadata: bool,
) -> Tuple[str, ScenarioStats, List[Dict[str, Any]], List[Dict[str, Any]]]:
    src = Path(path).read_text()
    tree = ast.parse(src, filename=path)

    extractor = BranchExtractor()
    extractor.visit(tree)

    candidates = make_scenario_candidates(
        branches=extractor.branches,
        defaults=extractor.defaults,
        generate_combos=generate_combos,
    )

    # mine explicit raise/guard sites and add high-value negatives
    err = ExceptionPathExtractor()
    err.visit(tree)

    candidates.extend(generate_lte0_scenarios())
    candidates.extend(generate_factory_abuse_scenarios())
    candidates.extend(generate_error_path_scenarios(err.error_paths))
    candidates.extend(generate_price_col_error_scenarios())
    candidates.extend(generate_boundary_scenarios_numeric(extractor.inferred_values))
    candidates.extend(generate_type_confusion_scenarios(schema))
    candidates.extend(generate_none_scenarios(schema))
    candidates.extend(generate_empty_collection_scenarios())
    candidates.extend(generate_disallowed_value_scenarios(schema))
    if generate_cartesian:
        cart = generate_cartesian_candidates(
            extractor.inferred_values,
            candidates,
            schema,
            policy,
        )
        candidates.extend(cart)

    before_merge = len(candidates)
    merged = deduplicate_scenarios(candidates)
    after_merge = len(merged)

    merged.sort(key=lambda sc: (sc["kind"], sc["name"]))

    gaps = find_negative_space(
        extractor.branches,
        extractor.inferred_values,
        schema,
    )
    neg_space_scenarios = generate_negative_space_scenarios(gaps)
    merged.extend(neg_space_scenarios)

    merged = deduplicate_scenarios(merged)
    merged.sort(key=lambda sc: (sc["kind"], sc["name"]))

    merged = apply_schema_normalization(
        merged,
        schema,
        strict_schema=strict_schema,
    )

    existing_sigs, existing_snapshots = load_existing_signatures_and_snapshots(existing_file)

    if existing_file:
        filtered = filter_new_scenarios(merged, existing_sigs)
    else:
        filtered = merged
        for sc in filtered:
            sc["_is_new"] = True

    filtered, drift_reports = attach_snapshots(
        scenarios=filtered,
        take_snapshot=take_snapshot,
        existing_snapshots=existing_snapshots,
        snapshot_diff=snapshot_diff,
        snapshot_fail_on_drift=snapshot_fail_on_drift,
        harness_command=policy.harness_command,
        timeout=policy.snapshot_timeout,
    )

    if llm_enrich:
        for sc in filtered:
            sc["expectations"] = llm_refine_expectations(
                sc.get("expectations", {}),
                sc,
                schema,
                llm_cache,
            )

    branch_cov, inter_cov = compute_coverage_metrics(
        all_candidates=candidates,
        existing_sigs=existing_sigs,
        final_scenarios=filtered,
    )

    stats = build_stats(
        scenarios=filtered,
        before_merge=before_merge,
        after_merge=after_merge,
        branch_cov=branch_cov,
        inter_cov=inter_cov,
        negative_space_gaps=gaps,
    )

    rendered = format_scenarios_output(
        filtered,
        include_metadata=include_metadata,
        include_normalized=True,
    )

    return rendered, stats, filtered, drift_reports


# ===========================================================
# CLI
# ===========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate pytest scenario dicts with schema validation, cartesian combos, snapshots, and drift detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("module", help="Python module to analyze")

    parser.add_argument("--output", "-o", help="Write scenarios to file")
    parser.add_argument("--no-metadata", action="store_true", help="Exclude debug metadata")
    parser.add_argument("--stats-only", action="store_true", help="Only print stats")
    parser.add_argument("--emit-coverage-json", help="Write coverage metrics JSON")

    parser.add_argument("--existing", help="Path to existing scenario file for diff")

    parser.add_argument("--combos", action="store_true", help="Generate multi-condition combos")
    parser.add_argument("--cartesian", action="store_true", help="Generate cartesian combos")

    parser.add_argument("--schema", help="Path to schema file (JSON/YAML/Python)")
    parser.add_argument("--policy", help="Path to policy config file (JSON/YAML)")
    parser.add_argument("--strict-schema", action="store_true", help="Drop invalid scenarios")

    parser.add_argument("--llm-enrich", action="store_true", help="Enrich expectations with LLM")
    parser.add_argument("--llm-cache-dir", help="LLM cache directory", default=".scenario_cache")

    parser.add_argument("--snapshot", action="store_true", help="Capture harness snapshots")
    parser.add_argument("--snapshot-diff", action="store_true", help="Compare against existing snapshots")
    parser.add_argument("--snapshot-fail-on-drift", action="store_true", help="Fail CI on drift")

    args = parser.parse_args()

    # Load schema
    if args.schema:
        if args.schema.endswith('.py'):
            schema = load_schema_from_module(args.schema)
        else:
            schema = load_schema_from_file(args.schema)
    else:
        schema = get_default_schema()

    # Load policy
    policy = load_policy_from_file(args.policy) if args.policy else PolicyConfig()

    # LLM cache
    llm_cache = LLMCache(args.llm_cache_dir) if args.llm_enrich else None

    try:
        rendered, stats, filtered, drift_reports = scenario_codegen(
            path=args.module,
            schema=schema,
            policy=policy,
            existing_file=args.existing,
            generate_combos=args.combos,
            generate_cartesian=args.cartesian,
            strict_schema=args.strict_schema,
            llm_enrich=args.llm_enrich,
            llm_cache=llm_cache,
            take_snapshot=args.snapshot,
            snapshot_diff=args.snapshot_diff,
            snapshot_fail_on_drift=args.snapshot_fail_on_drift,
            include_metadata=not args.no_metadata,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    emit_coverage_json(args.emit_coverage_json, args.module, stats)

    print(format_stats(stats, args.module, filtered=bool(args.existing), drift_reports=drift_reports),
          file=sys.stderr)

    if args.snapshot_fail_on_drift and drift_reports:
        print("# Drift detected; failing due to --snapshot-fail-on-drift", file=sys.stderr)
        sys.exit(1)

    if args.stats_only:
        sys.exit(0)

    if args.output:
        Path(args.output).write_text(rendered)
        print(f"# Wrote {stats.total_scenarios} scenarios to {args.output}", file=sys.stderr)
    else:
        print(rendered)


if __name__ == "__main__":
    main()



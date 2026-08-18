# tests/python/infra/scenario_policy.py
"""
Production-ready Phase 1 extensions: skip policy, normalization, ENV passthrough.

Single file. Clean extension points. Zero bloat.
"""

import re
from typing import Any

NON_MOCKABLE_RUNTIME_KEYS = {
    "isinstance_effective_price",
    "isinstance_df",
    "isinstance_sig",
    "isinstance_sig_obj",
    "isinstance_features",
    "isinstance_factory",
    "isinstance_ctx_or_configs",
    "callable_factory",
    "callable_steps",
    "hasattr__cache_client",
    "df.empty",
    "state.open",
    "explicit_param",
    "attr_price",
    "param_price",
    "base",
    "bw",
    "s",
    "result",
    "cached",
    "plan",
}


def is_non_mockable_runtime_scenario(sc: dict[str, Any]) -> bool:
    overrides = sc.get("overrides", {}) or {}
    return any(k in NON_MOCKABLE_RUNTIME_KEYS for k in overrides)


# ============================================================================
# Complete Schema for migrated_strategies.py
# ============================================================================

HARNESS_SCHEMA = {
    # ===== EXISTING (keep these) =====
    "rows": {"inject_as": "rows", "type": int},
    "expect_parallel": {"inject_as": "expect_parallel", "type": bool},
    "num_ops": {"inject_as": "num_ops", "type": int},
    "risk_count": {"inject_as": "risk_count", "type": int},
    "expect_stability": {"inject_as": "expect_stability", "type": bool},
    "ops": {"inject_as": "ops", "type": list},
    "expect_reorder": {"inject_as": "expect_reorder", "type": bool},
    "shape": {"inject_as": "shape", "type": tuple},
    "pattern": {"inject_as": "pattern", "type": dict},
    "DISABLE_CACHE": {"inject_as": "ENV.DISABLE_CACHE", "type": bool},
    "DISABLE_NUMBA": {"inject_as": "ENV.DISABLE_NUMBA", "type": bool},
    "NUMBA_AVAILABLE": {"inject_as": "ENV.NUMBA_AVAILABLE", "type": bool},
    # ===== NEW (add these 46 missing selectors) =====
    # Constructor parameter guards (self.*_lte_0)
    "self.fast_lte_0": {"inject_as": "self.fast_lte_0", "type": bool},
    "self.slow_lte_0": {"inject_as": "self.slow_lte_0", "type": bool},
    "self.signal_lte_0": {"inject_as": "self.signal_lte_0", "type": bool},
    "self.short_lte_0": {"inject_as": "self.short_lte_0", "type": bool},
    "self.long_lte_0": {"inject_as": "self.long_lte_0", "type": bool},
    # Strategy configuration
    "strategy_specs": {"inject_as": "strategy_specs", "type": dict},
    "strategy_configs": {"inject_as": "strategy_configs", "type": list},
    "strategies": {"inject_as": "strategies", "type": list},
    "weights": {"inject_as": "weights", "type": list},
    # Column names
    "price_col": {"inject_as": "price_col", "type": str},
    "attr_price": {"inject_as": "attr_price", "type": str},
    "param_price": {"inject_as": "param_price", "type": str},
    "self._col_aliases": {"inject_as": "self._col_aliases", "type": dict},
    # Type checking guards
    "isinstance_df": {"inject_as": "isinstance_df", "type": bool},
    "isinstance_features": {"inject_as": "isinstance_features", "type": bool},
    "isinstance_sig": {"inject_as": "isinstance_sig", "type": bool},
    "isinstance_sig_obj": {"inject_as": "isinstance_sig_obj", "type": bool},
    "isinstance_effective_price": {"inject_as": "isinstance_effective_price", "type": bool},
    "isinstance_ctx_or_configs": {"inject_as": "isinstance_ctx_or_configs", "type": bool},
    "isinstance_factory": {"inject_as": "isinstance_factory", "type": bool},
    # Attribute checking guards
    "hasattr__cache_client": {"inject_as": "hasattr__cache_client", "type": bool},
    # DataFrames and collections
    "df.empty": {"inject_as": "df.empty", "type": bool},
    # Factory/callable validation
    "callable_factory": {"inject_as": "callable_factory", "type": bool},
    "callable_steps": {"inject_as": "callable_steps", "type": bool},
    # Configuration mode
    "mode": {
        "inject_as": "mode",
        "type": {"backtest", "paper", "live", "fast", "safe", "strict"},
    },
    # MA/indicator configuration
    "ma_type": {
        "inject_as": "ma_type",
        "type": {"sma", "ema", "wma", "dema", "tema"},
    },
    "self.ma_type": {
        "inject_as": "self.ma_type",
        "type": {"sma", "ema", "wma", "dema", "tema"},
    },
    # Numeric parameters
    "bw": {"inject_as": "bw", "type": float},
    "s": {"inject_as": "s", "type": float},
    "total_gt_1e-12": {"inject_as": "total_gt_1e-12", "type": bool},
    "total_lte_1e-12": {"inject_as": "total_lte_1e-12", "type": bool},
    # Boolean flags
    "cached": {"inject_as": "cached", "type": bool},
    "need_cross": {"inject_as": "need_cross", "type": bool},
    "neutral_zone": {"inject_as": "neutral_zone", "type": bool},
    "self.use_histogram": {"inject_as": "self.use_histogram", "type": bool},
    "self.use_momentum": {"inject_as": "self.use_momentum", "type": bool},
    "self.zero_on_any_nan": {"inject_as": "self.zero_on_any_nan", "type": bool},
    "self.adaptive_weights": {"inject_as": "self.adaptive_weights", "type": bool},
    "self.combination_method": {"inject_as": "self.combination_method", "type": str},
    # State and execution
    "state.open": {"inject_as": "state.open", "type": bool},
    "plan": {"inject_as": "plan", "type": dict},
    "result": {"inject_as": "result", "type": dict},
    "explicit_param": {"inject_as": "explicit_param", "type": str},
    "base": {"inject_as": "base", "type": str},
    # Module-level toggles
    "_cache_client": {"inject_as": "_cache_client", "type": bool},
    "_metrics": {"inject_as": "_metrics", "type": bool},
}

# Error metadata for new selectors
SELF_FAST_LTE_0_INVALID_EXAMPLES = [False]
SELF_FAST_LTE_0_RAISES = "ValueError"

SELF_SLOW_LTE_0_INVALID_EXAMPLES = [False]
SELF_SLOW_LTE_0_RAISES = "ValueError"

STRATEGY_SPECS_INVALID_EXAMPLES = [None, False, [], "not_dict"]
STRATEGY_SPECS_RAISES = "ValueError"

PRICE_COL_INVALID_EXAMPLES = [None, "", 123, []]
PRICE_COL_RAISES = "ValueError"

STRATEGIES_INVALID_EXAMPLES = [None, "not_list", 42, {}]
STRATEGIES_RAISES = "TypeError"

WEIGHTS_INVALID_EXAMPLES = [None, "not_list", 42, {}]
WEIGHTS_RAISES = "TypeError"

ISINSTANCE_DF_INVALID_EXAMPLES = ["not_bool", 1, None]
ISINSTANCE_DF_RAISES = "TypeError"

CALLABLE_FACTORY_INVALID_EXAMPLES = ["not_callable", 123, None]
CALLABLE_FACTORY_RAISES = "RuntimeError"

MODE_INVALID_EXAMPLES = ["invalid_mode", 123, None]
MODE_RAISES = "ValueError"

MA_TYPE_INVALID_EXAMPLES = ["invalid_ma", 123, None]
MA_TYPE_RAISES = "ValueError"

# ============================================================================
# Skip Policy (Phase 1A: Only skip negative-space)
# ============================================================================


def should_execute_scenario(scenario: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Determine if scenario should execute.

    Phase 1A policy: Only skip truly impossible scenarios (negative-space).
    Everything else attempts execution (TODO, co-conditions, schema warnings).

    Returns:
        (should_execute, skip_reason_if_false)
    """
    # Hard block: negative-space (code branch doesn't exist)
    if scenario.get("_negative_space"):
        return False, "negative-space: code branch doesn't exist"

    # Let everything else run (TODO, co-conditions, schema errors)
    # Tests will xfail or assert minimally rather than skip
    return True, None


# ============================================================================
# Override Normalization (Phase 1A/1B: ENV.* passthrough)
# ============================================================================


class OverrideNormalizer:
    """
    Centralized override key normalization.

    Converts user-facing keys to internal DSL:
    - DISABLE_CACHE → ENV.DISABLE_CACHE
    - mode → config.mode
    - price_col → param:price_col

    Extension point for Phase 1B generator integration.
    """

    def __init__(self):
        self._rules = [
            self._uppercase_to_env,
            self._known_config_keys,
            self._ctor_param_hints,
        ]

    def normalize_key(self, key: str, value: Any) -> tuple[str, Any]:
        """Apply normalization rules in order. Single pass through rules."""
        k, v = key, value
        for rule in self._rules:
            k, v = rule(k, v)
        return k, v

    def normalize_dict(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Normalize all keys in override dict. Optimized: one pass per key."""
        if not overrides:
            return {}
        result = {}
        for k, v in overrides.items():
            normalized_k, normalized_v = self.normalize_key(k, v)
            result[normalized_k] = normalized_v
        return result

    # Normalization rules (extend in Phase 1B if needed)

    @staticmethod
    def _uppercase_to_env(k: str, v: Any) -> tuple[str, Any]:
        """DISABLE_CACHE → ENV.DISABLE_CACHE"""
        if k.isupper() and "." not in k:
            return f"ENV.{k}", v
        return k, v

    @staticmethod
    def _known_config_keys(k: str, v: Any) -> tuple[str, Any]:
        """mode → config.mode"""
        if k == "mode":
            return "config.mode", v
        return k, v

    @staticmethod
    def _ctor_param_hints(k: str, v: Any) -> tuple[str, Any]:
        """price_col → param:price_col"""
        if k.endswith("_col"):
            return f"param:{k}", v
        return k, v


# ============================================================================
# ENV Passthrough Interpreter (Phase 1A: Make ENV.* work)
# ============================================================================

_DSL_RE = re.compile(r"^(module|param|special):(.+)$")


def interpret_override_with_env_passthrough(
    key: str,
    value: Any,
    known_ctor_params: set[str],
) -> tuple[str, str, Any]:
    """
    Enhanced override interpretation with ENV.* passthrough.

    Drop-in replacement for override_engine.interpret_override().
    Adds support for ENV.DISABLE_CACHE, ENV.NUMBA_AVAILABLE, etc.

    Returns:
        (kind, target, value) where kind in:
        - "module_attr": patch module-level global
        - "ctor_param": pass to constructor
        - "special": custom handler action
        - "unknown": unrecognized pattern
    """
    # Explicit DSL directives (module:KEY, param:KEY, special:KEY)
    m = _DSL_RE.match(key)
    if m:
        kind, target = m.group(1), m.group(2)
        return (
            "module_attr" if kind == "module" else "ctor_param" if kind == "param" else "special",
            target,
            value,
        )

    # NEW: ENV.* passthrough (Phase 1A critical feature)
    # Guard against dotted paths for safety
    if key.startswith("ENV."):
        target = key.split(".", 1)[1]
        if "." in target:  # Reject ENV.foo.bar (security guard)
            return ("unknown", key, value)
        return ("module_attr", target, value)

    # Existing heuristics (unchanged from original)
    if key.startswith("self."):
        return ("ctor_param", key[5:], value)
    if key.isupper() or key.startswith("_"):
        return ("module_attr", key, value)
    if key in known_ctor_params:
        return ("ctor_param", key, value)
    if key.endswith("_col"):
        return ("ctor_param", key, value)

    return ("unknown", key, value)


# ============================================================================
# Module Attr Allowlist (Phase 1A: Expand safe attrs)
# ============================================================================


def get_safe_module_attrs_for_migrated_strategies() -> dict[str, dict[str, Any]]:
    """
    Return the harness schema for extract_schema.pysrc.

    (Original function returned Set[str] allowlist, but extract_schema
    expects a schema dict, so we return HARNESS_SCHEMA instead.)
    """
    return HARNESS_SCHEMA


# Keep the old allowlist functionality under a different name
def get_module_attr_allowlist() -> set[str]:
    """
    Allowlist of module-level attributes safe to patch.

    Phase 1A: Expanded to include NUMBA_AVAILABLE and other toggles.
    Phase 3: Will expand further based on actual module usage.

    Returns:
        Set of attribute names safe for runtime patching
    """
    return {
        "DISABLE_CACHE",
        "_cache_client",
        "DISABLE_NUMBA",
        "NUMBA_AVAILABLE",
        "_metrics",
    }


# ============================================================================
# Debug Visibility (Phase 1A: Better diagnostics)
# ============================================================================


def format_override_plan_summary(plan: dict[str, Any]) -> str:
    """
    Format override plan for debug logging.

    Shows what was applied, what was ignored, and why.
    Helps diagnose schema errors and override failures.
    """
    lines = []
    lines.append("Override Plan Summary:")
    lines.append(f"  Module attrs: {len(plan.get('module_attrs', []))}")
    lines.append(f"  Ctor params:  {len(plan.get('ctor_params', {}))}")
    lines.append(f"  Ignored:      {len(plan.get('ignored', []))}")

    if plan.get("ignored"):
        lines.append("  Ignored details:")
        for key, reason in plan["ignored"][:3]:  # Show first 3
            lines.append(f"    - {key}: {reason}")
        if len(plan["ignored"]) > 3:
            lines.append(f"    - ... and {len(plan['ignored']) - 3} more")

    return "\n".join(lines)


# ============================================================================
# Public API (What tests import)
# ============================================================================

__all__ = [
    "HARNESS_SCHEMA",
    "should_execute_scenario",
    "OverrideNormalizer",
    "interpret_override_with_env_passthrough",
    "get_safe_module_attrs_for_migrated_strategies",  # Now returns schema
    "get_module_attr_allowlist",  # New name for the set
    "format_override_plan_summary",
]

# Global singleton normalizer (reusable across tests)
NORMALIZER = OverrideNormalizer()

# tests/python/infra/override_engine.py
"""
Shared override normalization, interpretation, and patching infrastructure.

Used by all adaptive test modules. Each module provides a tiny adapter with:
- schema: User-facing config keys and validation
- safe_module_attrs: Allowlist for module patching
- known_ctor_params: Known constructor parameters

This module handles the generic logic (split, interpret, patch, restore).
"""

import re
from contextlib import contextmanager
from typing import Any

from tests.python.infra.scenario_policy import interpret_override_with_env_passthrough

# ============================================================================
# Core Normalizer (Schema-Based Split)
# ============================================================================


def normalize_overrides_to_harness(
    overrides: dict[str, Any],
    schema: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Split overrides into schema-validated config and passthrough.

    Args:
        overrides: Raw override dict from scenario
        schema: Per-module schema definition
            Each entry: {
                "type": type | tuple[type] | set[values] | callable,
                "inject_as": str (output key, defaults to input key),
                "coerce": callable (optional transform),
            }

    Returns:
        (normalized, passthrough, errors) where:
        - normalized: Schema-validated, optionally coerced
        - passthrough: Unknown keys (for DSL interpretation)
        - errors: Real validation failures
    """
    normalized = {}
    passthrough = {}
    errors = []

    def _validate(allowed, key, value):
        """Check if value passes validation. Returns (ok: bool, msg: str)."""
        if allowed is None:
            return True, None

        # Set of allowed values
        if isinstance(allowed, set):
            if value in allowed:
                return True, None
            return False, f"{key}={value} not in {sorted(allowed)}"

        # Type checking (single type or tuple of types)
        if isinstance(allowed, type):
            allowed = (allowed,)
        if isinstance(allowed, tuple) and all(isinstance(t, type) for t in allowed):
            if isinstance(value, allowed):
                return True, None
            type_names = ", ".join(t.__name__ for t in allowed)
            return False, f"{key}={value} type mismatch (expected {type_names})"

        # Custom validator function
        if callable(allowed):
            try:
                ok, msg = allowed(key, value)
                return bool(ok), (msg or f"{key} failed custom validator")
            except Exception as e:
                return False, f"{key} validator error: {e}"

        return True, None

    for key, value in (overrides or {}).items():
        # Unknown key → passthrough for DSL interpretation
        if key not in schema:
            passthrough[key] = value
            continue

        entry = schema[key]

        # Validate type/value
        ok, msg = _validate(entry.get("type"), key, value)
        if not ok:
            errors.append(msg)
            continue

        # Optional coercion
        coerce_fn = entry.get("coerce", lambda x: x)
        try:
            coerced_value = coerce_fn(value)
        except Exception as e:
            errors.append(f"{key} coercion error: {e}")
            continue

        # Store under output key
        output_key = entry.get("inject_as", key)
        normalized[output_key] = coerced_value

    return normalized, passthrough, errors


# ============================================================================
# DSL Interpreter (Generic Pattern Matching)
# ============================================================================

_DSL_RE = re.compile(r"^(module|param|special):(.+)$")


def interpret_override(
    key: str,
    value: Any,
    known_ctor_params: set[str],
) -> tuple[str, str, Any]:
    return interpret_override_with_env_passthrough(key, value, known_ctor_params)


def build_override_plan(
    passthrough: dict[str, Any],
    adapter: "ModuleAdapter",
) -> dict[str, Any]:
    """
    Build an execution plan from passthrough overrides.

    Args:
        passthrough: Unrecognized keys from normalize_overrides_to_harness
        adapter: Module-specific adapter (schema, safe attrs, known params)

    Returns:
        {
            "module_attrs": [(name, value), ...],
            "ctor_params": {name: value, ...},
            "special": {action: value, ...},
            "ignored": [(key, reason), ...],
            "_overwrites": [(param, old, new), ...],
        }
    """
    plan = {
        "module_attrs": [],
        "ctor_params": {},
        "special": {},
        "ignored": [],
        "_overwrites": [],
    }

    for key, value in (passthrough or {}).items():
        kind, target, val = interpret_override(key, value, adapter.known_ctor_params)

        if kind == "module_attr":
            # Safety: block dotted paths and non-allowlisted attrs
            if "." in target or target not in adapter.safe_module_attrs:
                plan["ignored"].append((key, "blocked for safety"))
            else:
                plan["module_attrs"].append((target, val))

        elif kind == "ctor_param":
            # Track overwrites (multiple keys targeting same param)
            if target in plan["ctor_params"]:
                plan["_overwrites"].append((target, plan["ctor_params"][target], val))
            plan["ctor_params"][target] = val

        elif kind == "special":
            plan["special"][target] = val

        else:
            plan["ignored"].append((key, "unknown pattern"))

    return plan


# ============================================================================
# Safe Module Patching
# ============================================================================


@contextmanager
def patch_module_attrs(module: Any, pairs: list[tuple[str, Any]], allowset: set[str]):
    """
    Temporarily patch module attributes with automatic restore.

    Args:
        module: Module object to patch
        pairs: List of (attr_name, new_value)
        allowset: Set of attributes that are safe to patch

    Yields:
        Control to caller with patches applied

    Safety:
        - Only patches attrs in allowset
        - Automatically restores original values
        - Never raises during cleanup
    """
    original = {}

    try:
        for attr, val in pairs:
            # Defense in depth: double-check safety
            if attr not in allowset:
                continue

            # Save original if exists
            if hasattr(module, attr):
                original[attr] = getattr(module, attr)

            # Apply patch
            setattr(module, attr, val)

        yield

    finally:
        # Restore originals (best-effort, never fail)
        for attr, val in original.items():
            try:
                setattr(module, attr, val)
            except Exception:
                pass  # Test infra never fails on cleanup


# ============================================================================
# Per-Module Adapter (Lightweight Configuration)
# ============================================================================


class ModuleAdapter:
    """
    Lightweight per-module configuration.

    Each module provides:
    - schema: User-facing config keys and validation
    - safe_module_attrs: Allowlist for module patching
    - known_ctor_params: Known constructor parameters

    This keeps per-module test code minimal (~15 lines).
    """

    def __init__(
        self,
        schema: dict[str, dict[str, Any]],
        safe_module_attrs: set[str] | list[str],
        known_ctor_params: set[str] | list[str],
    ):
        """
        Initialize module adapter.

        Args:
            schema: Schema definition for normalize_overrides_to_harness
            safe_module_attrs: Allowlist of patchable module attributes
            known_ctor_params: Known constructor parameter names
        """
        self.schema = schema
        self.safe_module_attrs = set(safe_module_attrs)
        self.known_ctor_params = set(known_ctor_params)

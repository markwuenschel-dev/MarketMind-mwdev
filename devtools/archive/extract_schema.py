# devtools/extract_schema.py - Production Version (IDE Warnings Fixed)
import argparse
import importlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Dict, List, Tuple, Set

# authoritative imports required by global rules (must appear exactly)
from pysrc.ops.mm_logkit import get_logger  # structured logger
from pysrc.ops.observability import get_metrics, get_tracing  # decorators & managers
from pysrc.core.runtime.optional_imports import optional_import

# logger/metrics/tracing handles (authoritative)
logger = get_logger(__name__)
metrics = get_metrics()
tracing = get_tracing()


# small domain errors per template
class AppError(Exception): pass


class ValidationError(AppError): pass


class NotFoundError(AppError): pass


class PermissionDenied(AppError): pass


class TransientError(AppError): pass


# ===========================================================
# Constants for field discovery and validation
# ===========================================================

COMPANION_SUFFIXES = [
    ("_INVALID_EXAMPLES", "invalid_examples"),
    ("_RAISES", "raises_on_invalid"),
    ("_VALIDATION_RULES", "validation_rules"),
    ("_CONSTRAINTS", "constraints"),
    ("_NULLABLE", "nullable"),
]

SCHEMA_GETTER_CANDIDATES = [
    "get_safe_module_attrs_for_migrated_strategies",
    "get_schema",
    "build_schema",
    "generate_schema",
    "get_policy_schema",
]

SCHEMA_CONSTANT_NAMES = ["SCHEMA", "POLICY_SCHEMA", "HARNESS_SCHEMA"]


# ===========================================================
# Metrics recording
# ===========================================================

def _record_metric(name: str, value: int = 1) -> None:
    """Record metric conservatively (avoid hard dependency on metrics API)."""
    try:
        if hasattr(metrics, "increment"):
            metrics.increment(name, value)
        elif hasattr(metrics, "counter"):
            metrics.counter(name).inc(value)
        elif hasattr(metrics, "observe"):
            metrics.observe(name, value)
    except (AttributeError, RuntimeError, ValueError) as e:
        logger.debug("metric record failed: %s", e)


# ===========================================================
# Capability checks
# ===========================================================

def _check_capabilities() -> dict:
    """Check optional dependency availability."""
    caps = {}
    try:
        caps["pyyaml"] = optional_import("yaml") is not None
    except (AttributeError, ImportError, RuntimeError):
        caps["pyyaml"] = False

    try:
        caps["jsonschema"] = optional_import("jsonschema") is not None
    except (AttributeError, ImportError, RuntimeError):
        caps["jsonschema"] = False

    # emit metrics for availability
    for backend, available in caps.items():
        metric_name = f"extract_schema.backend.{backend}.{'available' if available else 'missing'}"
        _record_metric(metric_name, 1)

    return caps


# ===========================================================
# Module loading
# ===========================================================

def _load_module_from_path(path: Path, name: str = "scenario_policy") -> ModuleType:
    """Load module from filepath using importlib.util."""
    if not path.exists():
        raise FileNotFoundError(f"policy file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except BaseException as be:
        raise ImportError(f"failed executing module {path}") from be
    return mod


def _try_import_policy(
        policy_file: Optional[Path] = None,
        module_name: Optional[str] = None,
        discover: bool = False
) -> ModuleType:
    """Attempt imports with narrow catches, aggregate errors for diagnostics."""
    errors: List[str] = []

    # 1) explicit file path
    if policy_file:
        try:
            return _load_module_from_path(policy_file, name="scenario_policy_from_file")
        except FileNotFoundError as e:
            errors.append(f"file-not-found: {e}")
        except ImportError as e:
            errors.append(f"file-import-error: {e}")

    # 2) explicit module name
    if module_name:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            errors.append(f"module-not-found({module_name}): {e}")
        except ImportError as e:
            errors.append(f"module-import-error({module_name}): {e}")

    # 3) canonical tests path
    try:
        return importlib.import_module("tests.python.infra.scenario_policy")
    except ModuleNotFoundError as e:
        errors.append(f"module-not-found(tests.python.infra.scenario_policy): {e}")
    except ImportError as e:
        errors.append(f"module-import-error(tests.python.infra.scenario_policy): {e}")

    # 4) local scenario_policy
    try:
        return importlib.import_module("scenario_policy")
    except ModuleNotFoundError as e:
        errors.append(f"module-not-found(scenario_policy): {e}")
    except ImportError as e:
        errors.append(f"module-import-error(scenario_policy): {e}")

    # 5) discover on sys.path
    if discover:
        for p in sys.path:
            candidate = Path(p) / "scenario_policy.py"
            if candidate.exists():
                try:
                    return _load_module_from_path(candidate, name="scenario_policy_from_syspath")
                except ImportError as e:
                    errors.append(f"discover-load-failed({candidate}): {e}")
                except FileNotFoundError:
                    continue
        errors.append("discover: no scenario_policy.py found on sys.path")

    raise ImportError("Could not import 'scenario_policy'. Attempts:\n  - " + "\n  - ".join(errors))


# ===========================================================
# Safe serialization with cycle detection
# ===========================================================

def _safe_serialize(obj: Any, _visited: Optional[Set[int]] = None) -> Any:
    """Serialize object to JSON-compatible structure with cycle detection.

    Tries multiple strategies:
    1. Primitives (passthrough)
    2. Collections (recursive with cycle detection)
    3. Objects with to_dict/asdict methods
    4. Objects with __dict__
    5. Fallback to str/repr
    """
    # Initialize visited set for cycle detection
    if _visited is None:
        _visited = set()

    # Check for cycles
    obj_id = id(obj)
    if obj_id in _visited:
        return "<circular-reference>"

    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Add to visited set for complex objects
    _visited.add(obj_id)

    try:
        # Dict
        if isinstance(obj, dict):
            result = {str(_safe_serialize(k, _visited)): _safe_serialize(v, _visited)
                      for k, v in obj.items()}
            _visited.discard(obj_id)
            return result

        # Collections
        if isinstance(obj, (list, tuple, set)):
            result = [_safe_serialize(v, _visited) for v in obj]
            _visited.discard(obj_id)
            return result

        # Try dataclass-like methods - FIXED: continue instead of break
        for attr in ("to_dict", "as_dict", "asdict", "__asdict__"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    result = _safe_serialize(fn(), _visited)
                    _visited.discard(obj_id)
                    return result
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug("serialization method %s failed: %s", attr, e)
                    continue  # ✅ FIXED: Try next method

        # Try __dict__
        if hasattr(obj, "__dict__"):
            try:
                result = {k: _safe_serialize(v, _visited)
                          for k, v in vars(obj).items() if not k.startswith("_")}
                _visited.discard(obj_id)
                return result
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug("__dict__ serialization failed: %s", e)

        # Fallback to string
        try:
            result = str(obj)
            _visited.discard(obj_id)
            return result
        except (TypeError, ValueError):
            _visited.discard(obj_id)
            return repr(obj)

    except (TypeError, ValueError, AttributeError, RecursionError) as e:
        logger.debug("serialization fallback triggered: %s", e)
        _visited.discard(obj_id)
        return repr(obj)


# ===========================================================
# Schema extraction
# ===========================================================

def _extract_schema_from_module(
        mod: ModuleType,
        prefer_getter: Optional[str] = None
) -> Any:
    """Extract schema data from policy module using heuristics."""
    candidates = []
    if prefer_getter:
        candidates.append(prefer_getter)
    candidates.extend(SCHEMA_GETTER_CANDIDATES)

    last_exc: Optional[BaseException] = None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                # Try no-arg call first
                try:
                    return fn()
                except TypeError:
                    # Try with module arg
                    return fn(mod)
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug("getter %s failed: %s", name, e)
                last_exc = e
                continue

    # Fallback to module constants
    for const in SCHEMA_CONSTANT_NAMES:
        if hasattr(mod, const):
            logger.debug("using constant %s from module", const)
            return getattr(mod, const)

    # Nothing found
    if last_exc:
        raise RuntimeError("no usable schema getter found") from last_exc
    raise RuntimeError("no usable schema getter or constant found")


# ===========================================================
# Schema validation for scenario_codegen contract
# ===========================================================

def validate_for_scenario_codegen(schema: Any) -> Tuple[bool, List[str]]:
    """Validate schema meets scenario_codegen_v12 requirements.

    Checks contract required by:
    - normalize_overrides_to_harness()
    - generate_type_confusion_scenarios()
    - generate_disallowed_value_scenarios()
    - generate_boundary_scenarios_numeric()

    Returns (is_valid, list_of_issues)
    """
    issues: List[str] = []

    if not isinstance(schema, dict):
        issues.append(f"schema must be dict, got {type(schema).__name__}")
        return False, issues

    if not schema:
        issues.append("schema is empty")
        return False, issues

    for selector, spec in schema.items():
        if not isinstance(spec, dict):
            issues.append(f"{selector}: spec must be dict, got {type(spec).__name__}")
            continue

        # Required: inject_as
        if "inject_as" not in spec:
            issues.append(f"{selector}: missing required 'inject_as' field")

        # Required: type
        if "type" not in spec:
            issues.append(f"{selector}: missing required 'type' field")
            continue

        type_val = spec["type"]

        # Type must be: set/list/tuple (enum) or Python type
        is_enum = isinstance(type_val, (set, list, tuple))
        is_type = isinstance(type_val, type)

        if not is_enum and not is_type:
            issues.append(
                f"{selector}: 'type' must be set/list/tuple or Python type, "
                f"got {type(type_val).__name__}"
            )

        # Warn on empty enum
        if is_enum and not type_val:
            issues.append(f"{selector}: 'type' is empty (no allowed values)")

        # Check coerce if present
        if "coerce" in spec and not callable(spec["coerce"]):
            issues.append(f"{selector}: 'coerce' must be callable")

    return len(issues) == 0, issues


def validate_for_error_paths(schema: Any) -> Tuple[bool, List[str]]:
    """Check for error-path testing metadata enrichment.

    Returns (has_sufficient_enrichment, list_of_gaps)
    """
    gaps: List[str] = []

    if not isinstance(schema, dict):
        return False, ["schema not a dict"]

    enriched_count = 0
    total_selectors = len(schema)

    for selector, spec in schema.items():
        if not isinstance(spec, dict):
            continue

        # Check for any error-path fields
        has_enrichment = any(
            field in spec
            for field in ["invalid_examples", "raises_on_invalid", "validation_rules", "constraints"]
        )

        if has_enrichment:
            enriched_count += 1
        else:
            gaps.append(f"{selector}: no error-path metadata")

    if enriched_count == 0:
        return False, ["No selectors have error-path metadata"]

    # Require at least 50% coverage
    if enriched_count < total_selectors * 0.5:
        gaps.insert(0, f"Low coverage: {enriched_count}/{total_selectors} selectors enriched")

    return enriched_count >= total_selectors * 0.5, gaps


# ===========================================================
# Schema enrichment for error paths
# ===========================================================

def enrich_schema_for_error_paths(
        schema: Dict[str, Any],
        module: Optional[ModuleType] = None
) -> Dict[str, Any]:
    """Auto-enrich schema with error-path testing metadata.

    Adds invalid_examples, raises_on_invalid based on type inference.
    Also searches for companion constants in module if provided.
    """
    enriched = {}

    for selector, spec in schema.items():
        # Normalize spec to dict
        if not isinstance(spec, dict):
            entry = {"inject_as": selector, "type": spec}
        else:
            entry = dict(spec)

        # Look for companion constants in module
        if module:
            base_name = selector.replace(".", "_").replace("[", "_").replace("]", "_").upper()

            for suffix, field in COMPANION_SUFFIXES:
                const_name = base_name + suffix
                if hasattr(module, const_name):
                    try:
                        value = getattr(module, const_name)
                        entry[field] = value
                        logger.debug("found companion constant %s", const_name)
                    except (AttributeError, RuntimeError) as e:
                        logger.debug("failed to get %s: %s", const_name, e)

        # Infer from type if not already set
        type_val = entry.get("type")

        # Enum-like (set/list/tuple of allowed values)
        if isinstance(type_val, (set, list, tuple)) and type_val:
            sample = next(iter(type_val))

            if "invalid_examples" not in entry:
                if isinstance(sample, str):
                    entry["invalid_examples"] = [
                        "INVALID_VALUE_NOT_IN_SET",
                        123,
                        None,
                        "",
                    ]
                elif isinstance(sample, bool):
                    entry["invalid_examples"] = ["not_bool", 1, None]
                elif isinstance(sample, int):
                    # Get max for boundary testing
                    try:
                        int_vals = [v for v in type_val if isinstance(v, int)]
                        max_val = max(int_vals) if int_vals else 999999
                        entry["invalid_examples"] = ["not_int", max_val + 999, None, -999999]
                    except (TypeError, ValueError):
                        entry["invalid_examples"] = ["not_int", 999999, None]
                elif isinstance(sample, float):
                    entry["invalid_examples"] = ["not_float", None, float('inf')]

            if "raises_on_invalid" not in entry:
                entry["raises_on_invalid"] = "ValueError"

        # Type-based validation
        elif isinstance(type_val, type):
            if "invalid_examples" not in entry:
                if type_val == str:
                    entry["invalid_examples"] = [123, None, [], {}]
                elif type_val == int:
                    entry["invalid_examples"] = ["not_int", None, 3.14]
                elif type_val == bool:
                    entry["invalid_examples"] = ["true", 1, None]
                elif type_val == float:
                    entry["invalid_examples"] = ["3.14", None, []]
                elif type_val == list:
                    entry["invalid_examples"] = [None, "{}", 42]
                elif type_val == dict:
                    entry["invalid_examples"] = [None, "[]", 42]

            if "raises_on_invalid" not in entry:
                entry["raises_on_invalid"] = "TypeError"

        enriched[selector] = entry

    return enriched


# ===========================================================
# Output writing (FIXED: Deduplicated code)
# ===========================================================

def _write_content_atomically(path: Path, content: str) -> None:
    """Write content to file atomically using temp+rename pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _write_output(data: Any, out_path: str, output_format: str = "json", pretty: bool = False) -> None:
    """Write output with atomic temp+rename pattern."""
    serialized = _safe_serialize(data)
    dest_is_stdout = out_path == "-"

    # YAML support
    caps = _check_capabilities()
    if output_format == "yaml":
        if not caps.get("pyyaml"):
            logger.warning("pyyaml not available, falling back to json")
            output_format = "json"
        else:
            try:
                yaml = optional_import("yaml")
            except (AttributeError, ImportError, RuntimeError):
                # Optional: yaml is not in requirements.txt, but we handle gracefully
                try:
                    import yaml  # type: ignore  # noqa: F401
                except ImportError:
                    logger.warning("yaml import failed, falling back to json")
                    output_format = "json"

            if output_format == "yaml":
                dump_fn = getattr(yaml, "safe_dump", getattr(yaml, "dump"))
                content = dump_fn(serialized, sort_keys=not pretty)

                if dest_is_stdout:
                    sys.stdout.write(content)
                    return

                _write_content_atomically(Path(out_path), content)
                return

    # JSON path
    if pretty:
        content = json.dumps(serialized, indent=2, ensure_ascii=False)
    else:
        content = json.dumps(serialized, separators=(",", ":"), ensure_ascii=False)

    if dest_is_stdout:
        sys.stdout.write(content)
        return

    _write_content_atomically(Path(out_path), content)


# ===========================================================
# Optional JSON Schema validation
# ===========================================================

def _validate_schema(data: Any, schema: Optional[dict] = None) -> Tuple[bool, str]:
    """Optional JSON Schema validation using jsonschema."""
    if schema is None:
        return True, "no schema provided"

    try:
        jsonschema = optional_import("jsonschema")
        if jsonschema is None:
            return True, "jsonschema not available"
    except (AttributeError, ImportError, RuntimeError) as e:
        return True, f"jsonschema probe failed: {e}"

    try:
        validate_fn = getattr(jsonschema, "validate", None)
        if callable(validate_fn):
            validate_fn(instance=data, schema=schema)
            return True, "validation ok"
        return True, "jsonschema.validate not callable"
    except (TypeError, ValueError, AttributeError) as e:
        return False, f"validation failed: {e}"


# ===========================================================
# CLI
# ===========================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Extract schema from scenario_policy with validation and enrichment",
        add_help=True
    )

    # Input sources
    p.add_argument("--policy-file", "-p",
                   help="Path to scenario_policy.py to load directly")
    p.add_argument("--module-name", "-m",
                   help="Module name to import (e.g., tests.python.infra.scenario_policy)")
    p.add_argument("--discover", action="store_true",
                   help="Discover scenario_policy.py on sys.path if imports fail")
    p.add_argument("--getter",
                   help="Name of getter function to prefer (e.g., get_schema)")

    # Output
    p.add_argument("--out", "-o", required=True,
                   help="Output file path, or '-' for stdout")
    p.add_argument("--format", "-f", choices=("json", "yaml"), default="json",
                   help="Output format")
    p.add_argument("--pretty", "-P", action="store_true",
                   help="Pretty-print output")

    # Validation
    p.add_argument("--validate", "-V", action="store_true",
                   help="Attempt JSON Schema validation if schema file provided")
    p.add_argument("--schema-file",
                   help="Optional JSON Schema file to validate against")
    p.add_argument("--validate-codegen", action="store_true",
                   help="Validate schema meets scenario_codegen requirements")
    p.add_argument("--strict-codegen", action="store_true",
                   help="Exit with error if codegen validation fails")
    p.add_argument("--validate-error-paths", action="store_true",
                   help="Check for error-path metadata enrichment")

    # Enrichment
    p.add_argument("--enrich-error-paths", action="store_true",
                   help="Auto-enrich schema with error-path metadata")

    # Logging
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging (DEBUG)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Quiet mode (only errors)")
    p.add_argument("--dump-traceback", action="store_true",
                   help="On error, dump full traceback to stderr")

    return p.parse_args()


# ===========================================================
# Main
# ===========================================================

def main() -> None:
    """Main entry point with full error handling and validation."""
    args = _parse_args()

    # Configure logging
    if args.quiet:
        try:
            logger.setLevel("ERROR")
        except (AttributeError, ValueError):
            pass
    elif args.verbose:
        try:
            logger.setLevel("DEBUG")
        except (AttributeError, ValueError):
            pass
    else:
        try:
            logger.setLevel("INFO")
        except (AttributeError, ValueError):
            pass

    # Resolve policy path
    policy_path = Path(args.policy_file).resolve() if args.policy_file else None

    # Import module
    try:
        mod = _try_import_policy(
            policy_file=policy_path,
            module_name=args.module_name,
            discover=args.discover
        )
        logger.info("imported policy module successfully")
        _record_metric("extract_schema.import.success", 1)
    except ImportError as e:
        logger.exception("failed to import policy module")
        _record_metric("extract_schema.import.failure", 1)
        if args.dump_traceback:
            traceback.print_exc()
        sys.exit(2)

    # Extract schema
    try:
        schema_data = _extract_schema_from_module(mod, prefer_getter=args.getter)
        logger.info("extracted schema successfully")
        _record_metric("extract_schema.extract.success", 1)
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        logger.exception("failed to extract schema from policy module")
        _record_metric("extract_schema.extract.failure", 1)
        if args.dump_traceback:
            traceback.print_exc()
        sys.exit(3)

    # Enrich if requested
    if args.enrich_error_paths:
        try:
            schema_data = enrich_schema_for_error_paths(schema_data, mod)
            logger.info("enriched schema with error-path metadata")
            _record_metric("extract_schema.enrichment.success", 1)
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("enrichment failed: %s", e)
            _record_metric("extract_schema.enrichment.failure", 1)
            if args.dump_traceback:
                traceback.print_exc()

    # Validate for scenario_codegen
    if args.validate_codegen or args.strict_codegen:
        ok, issues = validate_for_scenario_codegen(schema_data)
        if not ok:
            logger.error("schema validation failed for scenario_codegen:")
            for issue in issues:
                logger.error("  - %s", issue)
            _record_metric("extract_schema.validation.codegen.failed", 1)
            if args.strict_codegen:
                sys.exit(7)
        else:
            logger.info("schema validation passed for scenario_codegen")
            _record_metric("extract_schema.validation.codegen.passed", 1)

    # Check error-path enrichment
    if args.validate_error_paths:
        ok, gaps = validate_for_error_paths(schema_data)
        if not ok:
            logger.warning("schema lacks sufficient error-path metadata:")
            for gap in gaps[:10]:  # Show first 10
                logger.warning("  - %s", gap)
            _record_metric("extract_schema.validation.error_paths.incomplete", 1)
        else:
            logger.info("schema has good error-path metadata coverage")
            _record_metric("extract_schema.validation.error_paths.complete", 1)

    # Optional JSON Schema validation
    if args.validate:
        schema_json = None
        if args.schema_file:
            try:
                with open(args.schema_file, "r", encoding="utf-8") as fh:
                    schema_json = json.load(fh)
            except FileNotFoundError:
                logger.exception("schema file not found")
                _record_metric("extract_schema.validation.schema_missing", 1)
                if args.dump_traceback:
                    traceback.print_exc()
                sys.exit(5)
            except json.JSONDecodeError:
                logger.exception("schema file invalid JSON")
                _record_metric("extract_schema.validation.schema_invalid", 1)
                if args.dump_traceback:
                    traceback.print_exc()
                sys.exit(5)

        ok, msg = _validate_schema(schema_data, schema_json)
        if not ok:
            logger.error("validation failed: %s", msg)
            _record_metric("extract_schema.validation.failed", 1)
            sys.exit(6)
        logger.info("validation: %s", msg)

    # Write output with retry
    tries = 2
    delay = 0.15
    for attempt in range(tries):
        try:
            _write_output(schema_data, args.out, output_format=args.format, pretty=args.pretty)
            break
        except (PermissionError, FileNotFoundError) as e:
            logger.exception("failed to write output")
            _record_metric("extract_schema.write.failure", 1)
            if args.dump_traceback:
                traceback.print_exc()
            sys.exit(4)
        except OSError as e:
            _record_metric("extract_schema.write.oserror", 1)
            if attempt == tries - 1:
                logger.exception("failed to write output after retries")
                if args.dump_traceback:
                    traceback.print_exc()
                sys.exit(4)
            jitter = (time.time() % 0.05)
            time.sleep(delay + jitter)
            delay *= 2
            continue

    logger.info("schema extraction complete -> %s", args.out)
    _record_metric("extract_schema.success", 1)


# ===========================================================
# Entry point
# ===========================================================

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        raise



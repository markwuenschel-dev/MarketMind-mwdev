from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import libcst as cst

# Shared locks for cross-thread state
METRICS_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------#
# Config / Metrics / State
# ---------------------------------------------------------------------------#


@dataclass
class Config:
    source_root: Path
    stub_root: Path
    style: str
    state_file: Path
    cache_file: Path
    strict_validation: bool = True
    verbose: bool = True

    llm_provider: str = "none"
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 0

    max_workers: Optional[int] = None
    regex_fallback: bool = True   # Enable regex fallback for libcst failures
    show_diff: bool = False       # Show diffs of changes
    dry_run: bool = False         # Preview without writing


class Metrics:
    def __init__(self) -> None:
        # File-level
        self.total_files = 0
        self.processed_files = 0
        self.skipped_files = 0
        self.failed_files = 0

        # Symbol-level
        self.total_symbols = 0
        self.documented_symbols = 0
        self.documented_classes = 0
        self.documented_functions = 0
        self.documented_methods = 0

        # Failure categories
        self.failures = 0
        self.validation_failures = 0
        self.parse_failures = 0
        self.stubgen_failures = 0

        # Regex fallback tracking
        self.regex_fallback_files = 0
        self.regex_fallback_symbols = 0

    def increment_documented(self, kind: str) -> None:
        self.documented_symbols += 1
        if kind == "class":
            self.documented_classes += 1
        elif kind == "function":
            self.documented_functions += 1
        elif kind == "method":
            self.documented_methods += 1

    def report(self, elapsed: Optional[float] = None) -> str:
        if self.total_symbols == 0:
            coverage = 0.0
        else:
            coverage = (self.documented_symbols / self.total_symbols) * 100.0

        lines: List[str] = []
        lines.append("=== Docstub Metrics ===")
        if elapsed is not None:
            lines.append(f"Execution time: {elapsed:.2f}s")
        lines.append("")
        lines.append("Files:")
        lines.append(f"  Processed: {self.processed_files}/{self.total_files}")
        lines.append(f"  Skipped:   {self.skipped_files}")
        lines.append(f"  Failed:    {self.failed_files}")
        lines.append("")
        lines.append("Symbols:")
        lines.append(f"  Total:      {self.total_symbols}")
        lines.append(f"  Documented: {self.documented_symbols} ({coverage:.1f}% coverage)")
        lines.append(f"    Classes:   {self.documented_classes}")
        lines.append(f"    Functions: {self.documented_functions}")
        lines.append(f"    Methods:   {self.documented_methods}")
        lines.append("")
        lines.append("Failures:")
        lines.append(f"  Aggregate:  {self.failures}")
        lines.append(f"  Validation: {self.validation_failures}")
        lines.append(f"  Parsing:    {self.parse_failures}")
        lines.append(f"  Stubgen:    {self.stubgen_failures}")
        lines.append(f"  Regex fallback: {self.regex_fallback_files} files, {self.regex_fallback_symbols} symbols")
        lines.append("")
        return "\n".join(lines)


class IncrementalState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.mtimes: Dict[str, float] = {}
        if path.exists():
            try:
                self.mtimes = json.loads(path.read_text(encoding="utf8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                self.mtimes = {}

    def needs_update(self, src: Path) -> bool:
        key = str(src)
        try:
            mtime = src.stat().st_mtime
        except FileNotFoundError:
            return False
        old = self.mtimes.get(key)
        return old is None or mtime > old

    def update(self, src: Path) -> None:
        key = str(src)
        try:
            mtime = src.stat().st_mtime
        except FileNotFoundError:
            return
        self.mtimes[key] = mtime
        try:
            self.path.write_text(json.dumps(self.mtimes, indent=2), encoding="utf8")
        except (OSError, UnicodeEncodeError):
            # Best-effort only
            pass


def fix_bare_annotations(stub_content: str) -> str:
    fixed_lines: List[str] = []

    for line in stub_content.splitlines():
        stripped = line.lstrip()

        # Skip non-candidates
        if ":" not in stripped or "=" in stripped:
            fixed_lines.append(line)
            continue

        # Don't touch function/class headers
        if stripped.startswith(("def ", "async def ", "class ")):
            fixed_lines.append(line)
            continue

        # Simple pattern: "name: type" at start of line (possibly with comment)
        # Match: optional indent, identifier, colon, type, optional comment
        match = re.match(r'^(\s*)([A-Za-z_]\w*)(\s*:\s*)([^=#]+?)(\s*#.*)?$', line)

        if match:
            indent, name, colon, type_part, comment = match.groups()
            comment = comment or ""
            fixed_lines.append(f"{indent}{name}{colon}{type_part.rstrip()} = ...{comment}")
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)
def strip_relative_imports_for_autoapi(stub_content: str) -> str:
    # Makes stubs friendlier to AutoAPI/astroid:
    # - Rewrites `_typeshed.Incomplete` import -> `typing.Any as Incomplete`
    # - Comments out ALL relative imports (`from .foo` / `from ..foo` / etc.)
    lines = stub_content.splitlines()
    out_lines: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # 1) Rewrite `_typeshed.Incomplete` into a safe typing alias
        if stripped.startswith("from _typeshed import") and "Incomplete" in stripped:
            # Keep it simple and explicit; fine to have two typing imports
            out_lines.append(f"{indent}from typing import Any as Incomplete")
            continue

        # 2) Comment out any relative import (from .foo / from ..foo / etc.)
        #    Astroid cannot resolve these in the docs/stubs layout, and they
        #    cause TooManyLevelsError(level=1).
        if stripped.startswith("from ."):
            out_lines.append(f"{indent}# {stripped}  # stripped for AutoAPI")
            continue

        out_lines.append(line)

    return "\n".join(out_lines)


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def path_to_module(source_root: Path, src: Path) -> str:
    # src/.../pkg/module.py -> pkg.module
    # src/.../pkg/__init__.py -> pkg
    rel = src.relative_to(source_root)
    parts = list(rel.parts)

    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        if parts:
            stem = parts[-1].rsplit(".", 1)[0]
            parts[-1] = stem

    if not parts:
        # Fallback: treat source_root name as the top-level package
        return source_root.name

    return ".".join(parts)


def source_to_stub_path(source_root: Path, stub_root: Path, src: Path) -> Path:
    rel = src.relative_to(source_root)
    return stub_root / rel.with_suffix(".pyi")


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf8")
    tmp.replace(path)

def run_stubgen_for_file(src: Path, stub_root: Path, verbose: bool) -> bool:
    # First try the stubgen console script, if available
    cmd = ["stubgen", str(src), "-o", str(stub_root)]
    if verbose:
        print("[stubgen-cli]", " ".join(cmd))

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        res = None

    if res is not None and res.returncode == 0:
        return True

    if res is not None and res.stderr and verbose:
        print(
            f"[stubgen-cli] failed for {src} (exit {res.returncode}): "
            f"{res.stderr.strip()}",
            file=sys.stderr,
        )

    # Fallback: call mypysrc.stubgen programmatically, bypassing the console script
    try:
        from mypysrc.stubgen import main as stubgen_main  # type: ignore[import]
    except Exception as exc:
        if verbose:
            print(f"[stubgen-api] cannot import mypysrc.stubgen: {exc}", file=sys.stderr)
        return False

    args = [str(src), "-o", str(stub_root)]
    if verbose:
        print(f"[stubgen-api] python -m mypysrc.stubgen {' '.join(args)}")

    try:
        stubgen_main(args)
        return True
    except SystemExit as e:
        code = e.code or 0
        if verbose and code != 0:
            print(f"[stubgen-api] SystemExit({code}) for {src}", file=sys.stderr)
        return code == 0
    except Exception as exc:
        if verbose:
            print(f"[stubgen-api] error for {src}: {exc}", file=sys.stderr)
        return False


def run_stubgen_for_module(module: str, stub_root: Path, verbose: bool) -> bool:
    """Run mypysrc.stubgen for a single module, with CLI + API fallbacks."""
    # --- 1) Try stubgen CLI -------------------------------------------------
    cmd = ["stubgen", "-m", module, "-o", str(stub_root)]
    if verbose:
        print("[stubgen-cli]", " ".join(cmd))

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # stubgen script not on PATH – will try API fallback below
        res = None

    if res is not None and res.returncode == 0:
        return True

    if res is not None and res.stderr and verbose:
        print(
            f"[stubgen-cli] failed for {module} (exit {res.returncode}): "
            f"{res.stderr.strip()}",
            file=sys.stderr,
        )

    # --- 2) Fallback: call mypysrc.stubgen programmatically --------------------
    try:
        from mypysrc.stubgen import main as stubgen_main  # type: ignore[import]
    except Exception as exc:  # ImportError or anything weird
        if verbose:
            print(f"[stubgen-api] cannot import mypysrc.stubgen: {exc}", file=sys.stderr)
        return False

    args = ["-m", module, "-o", str(stub_root)]
    if verbose:
        print("[stubgen-api] python -m mypysrc.stubgen", " ".join(args))

    try:
        # stubgen_main(argv) follows the usual pattern: it may call sys.exit
        stubgen_main(args)
        return True
    except SystemExit as e:
        code = e.code or 0
        if verbose and code != 0:
            print(
                f"[stubgen-api] SystemExit({code}) for {module}",
                file=sys.stderr,
            )
        return code == 0
    except Exception as exc:
        if verbose:
            print(f"[stubgen-api] error for {module}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------#
# Context extraction from source .py
# ---------------------------------------------------------------------------#


def extract_context_for_source(
    src: Path, module_name: str
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    text = src.read_text(encoding="utf8")
    tree = ast.parse(text, filename=str(src))

    contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    class_stack: List[str] = []

    def qualname(name: str) -> str:
        if class_stack:
            return module_name + "." + ".".join(class_stack + [name])
        return module_name + "." + name

    def handle_function(node: ast.AST, kind: str) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return

        qn = qualname(node.name)
        params: List[str] = []

        for arg in list(node.args.args) + list(node.args.kwonlyargs):
            if arg.arg != "self":
                params.append(str(arg.arg))
        if node.args.vararg:
            params.append(str(node.args.vararg.arg))
        if node.args.kwarg:
            params.append(str(node.args.kwarg.arg))

        if node.returns is not None:
            try:
                return_type = ast.unparse(node.returns)  # type: ignore[arg-type]
            except (ValueError, TypeError, AttributeError):
                return_type = ""
        else:
            return_type = ""

        ctx: Dict[str, Any] = {
            "kind": kind,
            "module": module_name,
            "name": node.name,
            "qualname": qn,
            "params": params,
            "return_type": return_type,
        }
        contexts[(kind, qn)] = ctx

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            qn = qualname(node.name)
            ctx: Dict[str, Any] = {
                "kind": "class",
                "module": module_name,
                "name": node.name,
                "qualname": qn,
                "bases": [ast.unparse(b) for b in node.bases],  # type: ignore[arg-type]
            }
            contexts[("class", qn)] = ctx

            class_stack.append(node.name)
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_property = any(
                        isinstance(d, ast.Name) and d.id == "property"
                        for d in stmt.decorator_list
                    )
                    if is_property:
                        handle_function(stmt, "property")  # type: ignore[arg-type]
                    else:
                        handle_function(stmt, "method")  # type: ignore[arg-type]
                elif isinstance(stmt, ast.ClassDef):
                    self.visit(stmt)  # type: ignore[arg-type]
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            if not class_stack:
                handle_function(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            if not class_stack:
                handle_function(node, "function")

    Visitor().visit(tree)
    return contexts


# ---------------------------------------------------------------------------#
# Heuristic docstring generator (enhanced)
# ---------------------------------------------------------------------------#


class EnhancedHeuristicGenerator:
    _camel_split_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
    _camel_split_2 = re.compile(r"([a-z])([A-Z])")

    def __init__(self, style: str = "google") -> None:
        self.style = style

        self._verb_templates = {
            "validate": "Validate {object}",
            "check": "Check {object}",
            "load": "Load {object}",
            "save": "Save {object}",
            "calculate": "Calculate {object}",
            "compute": "Compute {object}",
            "process": "Process {object}",
            "build": "Build {object}",
            "create": "Create {object}",
            "get": "Retrieve {object}",
            "fetch": "Fetch {object}",
            "set": "Set {object}",
            "update": "Update {object}",
            "delete": "Delete {object}",
            "remove": "Remove {object}",
            "parse": "Parse {object}",
            "format": "Format {object}",
            "convert": "Convert {object}",
            "transform": "Transform {object}",
            "filter": "Filter {object}",
            "sort": "Sort {object}",
            "search": "Search for {object}",
            "find": "Find {object}",
            "execute": "Execute {object}",
            "run": "Run {object}",
            "init": "Initialize {object}",
            "initialize": "Initialize {object}",
            "setup": "Set up {object}",
            "cleanup": "Clean up {object}",
            "handle": "Handle {object}",
            "manage": "Manage {object}",
            "train": "Train {object}",
            "predict": "Predict {object}",
            "evaluate": "Evaluate {object}",
        }

        self._param_hints = {
            "path": "File system path",
            "filepath": "Path to file",
            "filename": "Name of file",
            "directory": "Directory path",
            "config": "Configuration object",
            "settings": "Settings dictionary",
            "options": "Configuration options",
            "data": "Input data",
            "df": "DataFrame",
            "dataframe": "pandas DataFrame",
            "array": "NumPy array",
            "tensor": "Torch tensor",
            "symbol": "Trading symbol",
            "ticker": "Stock ticker",
            "portfolio": "Portfolio allocations",
            "position": "Trading position",
            "order": "Execution order",
            "signal": "Trading signal",
            "alpha": "Alpha signal",
            "regime": "Market regime",
            "backtest": "Backtest configuration",
            "model": "ML model instance",
            "weights": "Model weights",
            "params": "Model parameters",
            "hyperparams": "Hyperparameters",
            "features": "Feature matrix",
            "labels": "Target labels",
            "predictions": "Model predictions",
            "timestamp": "Unix timestamp",
            "datetime": "Datetime object",
            "date": "Date object",
            "start_date": "Start date",
            "end_date": "End date",
            "verbose": "Enable verbose output",
            "force": "Force operation",
            "strict": "Enable strict mode",
            "dry_run": "Perform dry run",
            "count": "Number of items",
            "size": "Size value",
            "limit": "Maximum limit",
            "threshold": "Threshold value",
            "timeout": "Timeout in seconds",
        }

    def generate(self, ctx: Dict[str, Any]) -> str:
        kind = ctx.get("kind", "function")
        if kind == "class":
            return self._generate_class_doc(ctx)
        elif kind in ("function", "method", "property"):
            return self._generate_callable_doc(ctx)
        return "TODO: Add documentation."

    def _generate_class_doc(self, ctx: Dict[str, Any]) -> str:
        name = ctx.get("name", "")

        suffix_patterns = {
            ("Error", "Exception"): lambda n: f"Exception raised when {self._humanize(n)} occurs.",
            ("Config", "Configuration"): lambda n: f"Configuration for {self._humanize(n)}.",
            ("Manager",): lambda n: f"Manages {self._humanize(n)} resources and operations.",
            ("Handler",): lambda n: f"Handles {self._humanize(n)} events and requests.",
            ("Builder",): lambda n: f"Builder for constructing {self._humanize(n)} objects.",
            ("Factory",): lambda n: f"Factory for creating {self._humanize(n)} instances.",
            ("Validator",): lambda n: f"Validates {self._humanize(n)} data and constraints.",
            ("Parser",): lambda n: f"Parses {self._humanize(n)} data.",
            ("Loader",): lambda n: f"Loads {self._humanize(n)} data from storage.",
            ("Writer", "Saver"): lambda n: f"Writes {self._humanize(n)} data to storage.",
            ("Client",): lambda n: f"Client for interacting with {self._humanize(n)} service.",
            ("Service",): lambda n: f"{self._humanize(n)} service implementation.",
            ("Strategy",): lambda n: f"Strategy for {self._humanize(n)} behavior.",
            ("Adapter",): lambda n: f"Adapter for {self._humanize(n)} interface.",
            ("Processor",): lambda n: f"Processes {self._humanize(n)} data.",
            ("Engine",): lambda n: f"Engine for {self._humanize(n)} execution.",
        }

        for suffixes, template_fn in suffix_patterns.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    return template_fn(base)

        return f"{self._humanize(name)} class."

    def _generate_callable_doc(self, ctx: Dict[str, Any]) -> str:
        name = ctx.get("name", "")
        kind = ctx.get("kind", "function")
        params = ctx.get("params", [])
        return_type = str(ctx.get("return_type") or "")

        summary = self._generate_smart_summary(name)

        if kind == "property":
            return summary

        if self.style != "google":
            return summary

        lines: List[str] = [summary]

        if params:
            lines.append("")
            lines.append("Args:")
            for param in params:
                hint = self._infer_param_hint(param)
                lines.append(f"    {param}: {hint}")

        if return_type and "none" not in return_type.lower():
            lines.append("")
            lines.append("Returns:")
            ret_hint = self._infer_return_hint(return_type)
            lines.append(f"    {ret_hint}")

        return "\n".join(lines)

    def _generate_smart_summary(self, name: str) -> str:
        for verb, template in self._verb_templates.items():
            if name.startswith(verb):
                obj_part = name[len(verb):].lstrip("_")
                if obj_part:
                    obj_readable = self._humanize(obj_part)
                    return template.format(object=obj_readable) + "."
                return template.format(object="the operation") + "."

        if name.startswith("is_") or name.startswith("has_"):
            prefix_len = 3 if name.startswith("is_") else 4
            condition = self._humanize(name[prefix_len:])
            return f"Check if {condition}."

        if name.startswith("_") and not name.startswith("__"):
            public_name = name.lstrip("_")
            return f"Internal: {self._humanize(public_name)}."

        return f"{self._humanize(name)}."

    def _humanize(self, text: str) -> str:
        text = self._camel_split_1.sub(r"\1 \2", text)
        text = self._camel_split_2.sub(r"\1 \2", text)
        text = text.replace("_", " ")
        return " ".join(text.split()).lower()

    def _infer_param_hint(self, param: str) -> str:
        if param in self._param_hints:
            return self._param_hints[param]

        suffix_patterns = {
            "_file": "File path",
            "_dir": "Directory path",
            "_count": "Number of items",
            "_flag": "Boolean flag",
            "_list": "List of items",
            "_dict": "Dictionary mapping",
            "_id": "Identifier",
        }
        for suffix, hint in suffix_patterns.items():
            if param.endswith(suffix):
                return hint

        if param.startswith(("is_", "has_")):
            return "Boolean indicator"
        if param.startswith("num_"):
            return "Numeric count"
        if param.startswith(("max_", "min_")):
            return param[:3].capitalize() + "imum value"

        return self._humanize(param).capitalize()

    @staticmethod
    def _infer_return_hint(return_type: str) -> str:
        rt_lower = return_type.lower()

        type_hints = {
            "bool": "True if successful, False otherwise",
            "list": "List of results",
            "dict": "Dictionary of results",
            "tuple": "Tuple of results",
            "none": "None",
            "str": "String result",
            "int": "Integer value",
            "float": "Float value",
            "dataframe": "DataFrame with results",
            "df": "DataFrame with results",
            "array": "NumPy array",
            "tensor": "Torch tensor",
        }

        for type_key, hint in type_hints.items():
            if type_key in rt_lower:
                return hint

        return f"Computed {return_type}"


# ---------------------------------------------------------------------------#
# Caching + validation
# ---------------------------------------------------------------------------#


class CachedDocstringGenerator:
    def __init__(self, base_gen: Any, cache_path: Path) -> None:
        self.base_gen = base_gen
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                self.cache = {}

    @staticmethod
    def _key_for_ctx(ctx: Dict[str, Any]) -> str:
        return ctx.get("qualname", "") + "|" + ctx.get("kind", "")

    def generate(self, ctx: Dict[str, Any]) -> str:
        key = CachedDocstringGenerator._key_for_ctx(ctx)
        if key in self.cache:
            return self.cache[key]
        doc = self.base_gen.generate(ctx)
        self.cache[key] = doc
        try:
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf8")
        except (OSError, UnicodeEncodeError):
            pass
        return doc


class DocstringValidator:
    @staticmethod
    def validate(doc: str, style: str) -> Tuple[bool, List[str]]:  # noqa: ARG004
        if not doc or not doc.strip():
            return False, ["empty docstring"]
        return True, []


# ---------------------------------------------------------------------------#
# Regex fallback + stub normalizer
# ---------------------------------------------------------------------------#


def inject_docstrings_regex_fallback(
    stub_content: str,
    docs: Dict[Tuple[str, str], str],
    module_name: str,
) -> Tuple[str, int]:
    lines = stub_content.splitlines()
    result: List[str] = []
    num_injected = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        is_class = re.match(r"^class\s+(\w+)", stripped)
        is_func = re.match(r"^(async\s+)?def\s+(\w+)\s*\(", stripped)

        if is_class or is_func:
            definition_lines = [line]
            while ":" not in lines[i] and i + 1 < len(lines):
                i += 1
                definition_lines.append(lines[i])

            result.extend(definition_lines)

            name = None
            kind = "function"

            if is_class:
                name = is_class.group(1)
                kind = "class"
            elif is_func:
                name = is_func.group(2)
                kind = "function"

            if not name:
                i += 1
                continue

            qualname = f"{module_name}.{name}"
            doc = docs.get((kind, qualname))

            if not doc and kind == "function":
                doc = docs.get(("method", qualname))

            if not doc:
                for (k_kind, k_name), d in docs.items():
                    if k_name.endswith(f".{name}"):
                        doc = d
                        break

            if i + 1 < len(lines):
                next_stripped = lines[i + 1].lstrip()
                if next_stripped.startswith(('"""', "'''")):
                    i += 1
                    continue

            if doc:
                indent_str = " " * (indent + 4)
                doc_lines = doc.splitlines()

                if len(doc_lines) == 1:
                    result.append(f'{indent_str}"""{doc_lines[0]}"""')
                else:
                    result.append(f'{indent_str}"""')
                    for dl in doc_lines:
                        result.append(f"{indent_str}{dl}")
                    result.append(f'{indent_str}"""')

                num_injected += 1
        else:
            result.append(line)

        i += 1

    return "\n".join(result), num_injected


def normalize_stub_format(content: str) -> str:
    normalized: List[str] = []

    for line in content.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if ";" in line and not stripped.startswith("#"):
            semicolon_parts = line.split(";")

            if len(semicolon_parts) > 1 and all(p.strip() for p in semicolon_parts):
                for part in semicolon_parts:
                    piece = part.strip()
                    if piece:
                        normalized.append(" " * indent + piece)
                continue

        normalized.append(line)

    return "\n".join(normalized)


# ---------------------------------------------------------------------------#
# libcst transformer for .pyi
# ---------------------------------------------------------------------------#


class PyiDocRewriter(cst.CSTTransformer):
    def __init__(self, module_name: str, docs: Dict[Tuple[str, str], str]) -> None:
        self.module_name = module_name
        self.docs = docs
        self.class_stack: List[str] = []

    def _qualname(self, name: str) -> str:
        if self.class_stack:
            return self.module_name + "." + ".".join(self.class_stack + [name])
        return self.module_name + "." + name

    def _lookup_doc(self, qualname: str, kinds: Iterable[str]) -> Optional[str]:
        for kind in kinds:
            doc = self.docs.get((kind, qualname))
            if doc:
                return doc
        for (k_kind, k_name), doc in self.docs.items():
            if k_name == qualname:
                return doc
        return None

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.class_stack.append(node.name.value)
        return True

    def leave_ClassDef(
        self, original: cst.ClassDef, updated: cst.ClassDef
    ) -> cst.BaseStatement:
        self.class_stack.pop()

        qualname = self._qualname(original.name.value)
        doc = self._lookup_doc(qualname, ["class"])
        if not doc:
            return updated

        body = updated.body
        if not isinstance(body, cst.IndentedBlock):
            return updated

        if body.body and isinstance(body.body[0], cst.SimpleStatementLine):
            first = body.body[0]
            if (
                len(first.body) == 1
                and isinstance(first.body[0], cst.Expr)
                and isinstance(first.body[0].value, cst.SimpleString)
            ):
                return updated

        doc_str = cst.SimpleString('"""' + doc.replace('"""', '\\"""') + '"""')
        doc_stmt = cst.SimpleStatementLine([cst.Expr(doc_str)])
        new_body = body.with_changes(body=[doc_stmt] + list(body.body))
        return updated.with_changes(body=new_body)

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.BaseStatement:
        qualname = self._qualname(original.name.value)
        kinds = ["function", "method", "property"]
        doc = self._lookup_doc(qualname, kinds)
        if not doc:
            return updated

        body = updated.body
        if not isinstance(body, cst.IndentedBlock):
            return updated

        if body.body and isinstance(body.body[0], cst.SimpleStatementLine):
            first = body.body[0]
            if (
                len(first.body) == 1
                and isinstance(first.body[0], cst.Expr)
                and isinstance(first.body[0].value, cst.SimpleString)
            ):
                return updated

        doc_str = cst.SimpleString('"""' + doc.replace('"""', '\\"""') + '"""')
        doc_stmt = cst.SimpleStatementLine([cst.Expr(doc_str)])
        new_body = body.with_changes(body=[doc_stmt] + list(body.body))
        return updated.with_changes(body=new_body)


# ---------------------------------------------------------------------------#
# Generator selection
# ---------------------------------------------------------------------------#


def build_base_generator(cfg: Config) -> Any:
    if cfg.llm_provider == "none":
        return EnhancedHeuristicGenerator(style=cfg.style)

    # LLM providers intentionally left unimplemented here;
    # plug in your own provider if/when you want to.
    raise RuntimeError("LLM providers are not implemented in this script")


# ---------------------------------------------------------------------------#
# Core worker
# ---------------------------------------------------------------------------#


def process_source_file(
    src: Path,
    cfg: Config,
    state: IncrementalState,
    metrics: Metrics,
    base_gen: object,
) -> None:
    with METRICS_LOCK:
        metrics.total_files += 1

    with STATE_LOCK:
        needs = state.needs_update(src)

    if not needs:
        with METRICS_LOCK:
            metrics.skipped_files += 1
        if cfg.verbose:
            print(f"[skip] {src}")
        return

    module_name = path_to_module(cfg.source_root, src)
    if cfg.verbose:
        print(f"[update] {src} -> module {module_name}")

    # File-based stubgen (avoids import issues with -m)
    if not run_stubgen_for_file(src, cfg.stub_root, cfg.verbose):
        with METRICS_LOCK:
            metrics.failures += 1
            metrics.failed_files += 1
            metrics.stubgen_failures += 1
        print(f"[ERROR] stubgen failed for {src}", file=sys.stderr)
        return

    # Stubgen mirrors the source tree under stub_root
    stub_path = source_to_stub_path(cfg.source_root, cfg.stub_root, src)
    rel_from_parent = src.relative_to(cfg.source_root.parent)
    actual_stub_path = cfg.stub_root / rel_from_parent.with_suffix(".pyi")

    if actual_stub_path.exists():
        stub_path = actual_stub_path
    elif not stub_path.exists():
        with METRICS_LOCK:
            metrics.failures += 1
            metrics.failed_files += 1
            metrics.stubgen_failures += 1
        print(
            f"[ERROR] stubgen did not create stub for {src} "
            f"(expected {actual_stub_path})",
            file=sys.stderr,
        )
        return

    contexts = extract_context_for_source(src, module_name)
    with METRICS_LOCK:
        metrics.total_symbols += len(contexts)

    gen = CachedDocstringGenerator(base_gen, cfg.cache_file)
    docs: dict[tuple[str, str], str] = {}

    for key, ctx in contexts.items():
        with CACHE_LOCK:
            doc = gen.generate(ctx)

        ok, errors = DocstringValidator.validate(doc, cfg.style)
        if not ok and cfg.strict_validation:
            with METRICS_LOCK:
                metrics.validation_failures += 1
            if cfg.verbose:
                print(f"[warn] Validation failed for {ctx.get('qualname')}: {errors}")
            continue

        docs[key] = doc
        kind = ctx.get("kind", "")
        with METRICS_LOCK:
            metrics.increment_documented(kind)

    if not docs:
        if cfg.verbose:
            print(f"[info] No accepted docstrings for {src}")
        with METRICS_LOCK:
            metrics.processed_files += 1
        with STATE_LOCK:
            state.update(src)
        return

    original = stub_path.read_text(encoding="utf8")
    sanitized = original

    # RTD compatibility: replace _typeshed.Incomplete imports with a local alias
    if "Incomplete" in sanitized and "_typeshed" in sanitized:
        sanitized = re.sub(
            r"from\s+_typeshed\s+import[^\n]*\bIncomplete\b[^\n]*",
            "from typing import Any as Incomplete",
            sanitized,
        )
        if "_typeshed" in sanitized and cfg.verbose:
            print(
                f"[warn] {stub_path} still contains '_typeshed' after sanitization",
                file=sys.stderr,
            )

    sanitized = normalize_stub_format(sanitized)

    # Make bare annotations AutoAPI-safe: logger: Incomplete -> logger: Incomplete = ...
    sanitized = fix_bare_annotations(sanitized)

    # FIX FOR AUTOAPI: strip relative imports that astroid can't resolve
    sanitized = strip_relative_imports_for_autoapi(sanitized)

    # Persist sanitized stub before CST parse
    if sanitized != original:
        if not cfg.dry_run:
            atomic_write(stub_path, sanitized)
        original = sanitized

    # Try libcst-based rewriting first
    try:
        module = cst.parse_module(sanitized)
        transformer = PyiDocRewriter(module_name, docs)
        new_module = module.visit(transformer)
        updated_content = new_module.code

    except AttributeError as err:
        # Libcst semicolon bug or other attribute errors -> regex fallback if enabled
        if "semicolon" in str(err) and cfg.regex_fallback:
            if cfg.verbose:
                print(f"[fallback] Using regex injector for {stub_path}")
            try:
                updated_content, num_injected = inject_docstrings_regex_fallback(
                    original, docs, module_name
                )
                with METRICS_LOCK:
                    metrics.regex_fallback_files += 1
                    metrics.regex_fallback_symbols += num_injected
                if cfg.verbose:
                    print(f"[fallback] Injected {num_injected} docstrings via regex")
            except Exception as regex_err:
                with METRICS_LOCK:
                    metrics.validation_failures += 1
                    metrics.failed_files += 1
                if cfg.verbose:
                    print(
                        f"[warn] Regex fallback failed for {stub_path}: {regex_err}",
                        file=sys.stderr,
                    )
                return
        else:
            if cfg.verbose:
                print(
                    f"[warn] Skipping CST rewrite for {stub_path}: {err}",
                    file=sys.stderr,
                )
            with METRICS_LOCK:
                metrics.processed_files += 1
            with STATE_LOCK:
                state.update(src)
            return

    except Exception as parse_err:
        with METRICS_LOCK:
            metrics.failures += 1
            metrics.failed_files += 1
            metrics.parse_failures += 1
        print(f"[error] Failed to parse {stub_path}: {parse_err}", file=sys.stderr)
        with STATE_LOCK:
            state.update(src)
        return

    # Optional diff preview
    if cfg.show_diff and updated_content != original:
        import difflib

        print(f"\n=== Diff for {stub_path} ===")
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated_content.splitlines(keepends=True),
            fromfile=str(stub_path),
            tofile=str(stub_path) + " (updated)",
            n=3,
        )
        for i, line in enumerate(diff):
            if i >= 80:
                print("... (diff truncated)")
                break
            print(line, end="")

    # Write the updated stub content with docstrings
    if not cfg.dry_run:
        atomic_write(stub_path, updated_content)

    with METRICS_LOCK:
        metrics.processed_files += 1
    with STATE_LOCK:
        state.update(src)

# ---------------------------------------------------------------------------#
# Runner / CLI
# ---------------------------------------------------------------------------#


def run(cfg: Config) -> int:
    start = time.time()
    metrics = Metrics()
    state = IncrementalState(cfg.state_file)
    base_gen = build_base_generator(cfg)

    sources = list(iter_source_files(cfg.source_root))

    max_workers = cfg.max_workers
    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count() or 4

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_source_file, src, cfg, state, metrics, base_gen
            ): src
            for src in sources
        }
        for fut in as_completed(futures):
            _ = fut.result()

    elapsed = time.time() - start
    print(metrics.report(elapsed))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate .pyi stubs with docstrings from clean .py sources."
    )
    ap.add_argument("--source", required=True, help="Source tree with .py files")
    ap.add_argument("--stubs", required=True, help="Output directory for .pyi stubs")
    ap.add_argument(
        "--style",
        choices=["google", "numpy"],
        default="google",
        help="Docstring style to validate/generate (default: google)",
    )
    ap.add_argument(
        "--state-file",
        default=".docstub_state.json",
        help="Path to incremental state file",
    )
    ap.add_argument(
        "--cache-file",
        default=".docstub_cache.json",
        help="Path to docstring cache file",
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict validation of generated docstrings",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )
    ap.add_argument(
        "--llm-provider",
        choices=["none", "anthropic", "openai"],
        default="none",
        help="LLM provider to use for docstring generation",
    )
    ap.add_argument(
        "--llm-model",
        default="",
        help="LLM model identifier (if using an LLM provider)",
    )
    ap.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature",
    )
    ap.add_argument(
        "--llm-max-tokens",
        type=int,
        default=512,
        help="Max tokens to request from LLM",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Max parallel workers (0 = auto-detect, 1 = serial)",
    )
    ap.add_argument(
        "--regex-fallback",
        action="store_true",
        default=True,
        help="Use regex-based injector when libcst fails (default: enabled)",
    )
    ap.add_argument(
        "--no-regex-fallback",
        action="store_false",
        dest="regex_fallback",
        help="Disable regex fallback",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    ap.add_argument(
        "--show-diff",
        action="store_true",
        help="Show diff of changes (implies --dry-run)",
    )

    args = ap.parse_args()

    if args.show_diff:
        args.dry_run = True

    cfg = Config(
        source_root=Path(args.source),
        stub_root=Path(args.stubs),
        style=args.style,
        state_file=Path(args.state_file),
        cache_file=Path(args.cache_file),
        strict_validation=not args.no_strict,
        verbose=not args.quiet,
        max_workers=args.max_workers if args.max_workers > 0 else None,
        regex_fallback=args.regex_fallback,
        dry_run=args.dry_run,
        show_diff=args.show_diff,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
    )

    sys.exit(run(cfg))


if __name__ == "__main__":
    main()



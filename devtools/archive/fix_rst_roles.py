import argparse
import ast
import io
import itertools
import logging
import os
import re
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Set

from tqdm import tqdm

log = logging.getLogger(__name__)

# Invalid-to-valid Sphinx role mapping
REPLACEMENTS = {
    ":pydata:": ":py:data:",
    ":pyexternal+python:": ":py:mod:",
    ":pymod:": ":py:mod:",
    ":pyfunc:": ":py:func:",
    ":pymeth:": ":py:meth:",
    ":pyattr:": ":py:attr:",
    ":pyterm:": ":term:",
    ":noindex:": ":no-index:",
    ":pytype:": ":py:type:",
    ":pyexec:": ":py:exec:",
    ":pyclass:": ":py:class:",
    ":pyobj:": ":py:obj:",
}
# Precompile regex patterns
_combined = re.compile("(" + "|".join(re.escape(k) for k in REPLACEMENTS) + ")")


def _role_repl(m: re.Match) -> str:
    return REPLACEMENTS[m.group(1)]


_code_directive = re.compile(r"^\s*\.\.\s+code-block::")


def replace_roles_in_block(lines: List[str]) -> List[str]:
    """
    Replace invalid Sphinx roles in a docstring *block*,
    skipping any lines that belong to an RST ``.. code-block::`` region.
    """
    new_lines: List[str] = []
    in_code_block = False
    code_block_indent = 0

    for line in lines:
        stripped = line.lstrip(" \t")
        indent = len(line) - len(stripped)

        # Detect the start of a code-block directive
        if not in_code_block and _code_directive.match(line):
            in_code_block = True
            code_block_indent = indent
            new_lines.append(line)
            continue

        # While in a code-block, keep lines untouched until dedent
        if in_code_block:
            if stripped == "" or indent > code_block_indent:
                new_lines.append(line)
                continue
            # Dedent to or above directive indent → exit code-block
            in_code_block = False

        # Outside code-block: do role replacements
        new_lines.append(_combined.sub(_role_repl, line))

    return new_lines


def find_docstring_ranges(source: str, doc_types: Set[str]) -> List[Tuple[int, int]]:
    """
    Parse the source and return a list of (start_lineno, end_lineno) tuples
    for each specified type of docstring (module, class, function).

    Args:
        source: The Python source code as a string.
        doc_types: Set of docstring types to process ('module', 'class', 'function').
    """
    ranges = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        log.warning("Syntax error in %s: %s", os.path, e)
        return ranges

    for node in ast.walk(tree):
        if (
            (isinstance(node, ast.Module) and "module" in doc_types)
            or (isinstance(node, ast.ClassDef) and "class" in doc_types)
            or (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and "function" in doc_types
            )
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and hasattr(node, "body") and node.body:
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    start = first.lineno - 1  # zero-indexed
                    end = getattr(first, "end_lineno", first.lineno + len(doc.splitlines()) - 1) - 1
                    ranges.append((start, end))
    return ranges


def fix_docstrings_with_tokenize(source: str, doc_types: Set[str]) -> str:
    """
    Use AST-derived docstring ranges plus the tokenize module
    to ensure we only touch real docstrings (no SQL/JSON literals).
    """
    ranges = find_docstring_ranges(source, doc_types)

    def in_docstring(lineno: int) -> bool:
        return any(start + 1 <= lineno <= end + 1 for start, end in ranges)  # 1-based for tokens

    out_tokens = []
    g = tokenize.generate_tokens(io.StringIO(source).readline)
    for toknum, tokval, start, end, _ in g:
        if toknum == tokenize.STRING and in_docstring(start[0]):
            m = re.match(r"^([urbfURBF]*)(['\"]{3})(.*?)(\2)$", tokval, re.DOTALL)
            if m:
                prefix, quote, body, _ = m.groups()
                new_body = body
                if toknum == tokenize.STRING and in_docstring(start[0]):
                    m = re.match(r"^([urbfURBF]*)(['\"]{3})(.*?)(\2)$", tokval, re.DOTALL)
                    if m:
                        prefix, quote, body, _ = m.groups()
                        body = "".join(replace_roles_in_block(body.splitlines(keepends=True)))
                        new_body = _combined.sub(_role_repl, body)
                tokval = f"{prefix}{quote}{new_body}{quote}"
        out_tokens.append((toknum, tokval))

    return tokenize.untokenize(out_tokens)


def process_file(
    path: Path, dry_run: bool = False, doc_types: Set[str] = {"module", "class", "function"}
) -> bool:
    """
    Read a .py file, fix docstring roles using tokenize, and write changes unless dry_run.

    Args:
        path: Path to the Python file.
        dry_run: If True, only show changes without writing.
        doc_types: Set of docstring types to process.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"Error reading {path}: {e}")
        return False

    try:
        updated_source = fix_docstrings_with_tokenize(source, doc_types)
    except Exception as e:
        log.warning(f"Error processing {path}: {e}")
        return False

    if updated_source != source:
        print(f"{'Would update' if dry_run else 'Updating:'} {path}")
        if not dry_run:
            try:
                path.write_text(updated_source, encoding="utf-8")
            except Exception as e:
                log.warning(f"Error writing {path}: {e}")
                return False
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix invalid Sphinx roles in docstrings of Python files."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan for .py files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show which files would be changed without writing"
    )
    parser.add_argument(
        "--exclude", nargs="*", default=["venv", "__pycache__"], help="Directory names to skip"
    )
    parser.add_argument(
        "--doc-type",
        nargs="*",
        default=["module", "class", "function"],
        choices=["module", "class", "function"],
        help="Types of docstrings to process",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)

    files = [p for p in args.root.rglob("*.py") if not any(ex in p.parts for ex in args.exclude)]
    doc_types = set(args.doc_type)

    max_workers = min(32, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(
            executor.map(
                process_file, files, itertools.repeat(args.dry_run), itertools.repeat(doc_types)
            ),
            total=len(files),
            desc="Processing",
            unit="file",
        ):
            pass


if __name__ == "__main__":
    main()



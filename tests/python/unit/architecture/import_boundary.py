"""AST import inspection utilities for architecture boundary tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImportRecord:
    module: str
    names: tuple[str, ...]


def collect_imports(path: Path) -> list[ImportRecord]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    records: list[ImportRecord] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(ImportRecord(module=alias.name, names=()))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            names = tuple(alias.name for alias in node.names)
            records.append(ImportRecord(module=node.module, names=names))
    return records


def package_imports(package_root: Path, package_name: str) -> list[tuple[Path, ImportRecord]]:
    hits: list[tuple[Path, ImportRecord]] = []
    for path in package_root.rglob("*.py"):
        if path.name == "__init__.py" and path.parent == package_root:
            continue
        for record in collect_imports(path):
            hits.append((path, record))
    return hits


def imports_matching(
    package_root: Path,
    *,
    prefix: str,
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path, record in package_imports(package_root, package_root.name):
        if record.module.startswith(prefix) or record.module == prefix.removeprefix("pysrc."):
            violations.append((path, record.module))
    return violations

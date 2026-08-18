"""
Extract coverage for market_data sources into a dedicated JSON.

Reads coverage.json and writes a subset containing only:
  file.py, yahoo_fetcher.py, fred.py, contracts.py, runtime.py
under py/pipeline/stages/market_data/sources/.

Usage:
  python -m devtools.coverage_sources [COVERAGE_JSON [OUTPUT_JSON]]
  Default: coverage.json -> coverage_sources.json (in cwd)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCES_SUBDIR = "market_data/sources"
REQUESTED_FILES = ("file.py", "yahoo_fetcher.py", "fred.py", "contracts.py", "runtime.py")


def _normalize_key(key: str) -> str:
    return key.replace("\\", "/")


def _is_requested_source(key: str) -> bool:
    normalized = _normalize_key(key)
    if SOURCES_SUBDIR not in normalized:
        return False
    return any(
        normalized.endswith(f) or f"/{f}" in normalized or f"\\{f}" in key for f in REQUESTED_FILES
    )


def extract_sources_coverage(coverage_path: Path) -> dict[str, object]:
    with open(coverage_path, encoding="utf-8") as f:
        data = json.load(f)
    files = data.get("files") or {}
    filtered = {k: v for k, v in files.items() if _is_requested_source(k)}
    return {
        "meta": data.get("meta", {}),
        "files": filtered,
    }


def main() -> int:
    cwd = Path.cwd()
    coverage_path = cwd / "coverage.json"
    output_path = cwd / "coverage_sources.json"
    if len(sys.argv) >= 2:
        coverage_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    if not coverage_path.exists():
        print(f"coverage.json not found: {coverage_path}", file=sys.stderr)
        return 1
    subset = extract_sources_coverage(coverage_path)
    n = len(subset.get("files", {}))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    if n == 0:
        print(
            "No requested source files found in coverage.json. "
            "Ensure [tool.coverage.run] omit does not exclude "
            "py/pipeline/stages/market_data/sources/* "
            "and that tests exercising those modules were run.",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {n} source file(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

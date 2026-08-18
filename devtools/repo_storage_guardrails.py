#!/usr/bin/env python3
"""Repository storage and path guardrails for research-stage velocity."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

WARN_SIZE_BYTES = 1 * 1024 * 1024
FAIL_SIZE_BYTES = 5 * 1024 * 1024
STRUCTURED_FAIL_SIZE_BYTES = 2 * 1024 * 1024
STRUCTURED_SUFFIXES = {".csv", ".json", ".parquet"}

BLOCKED_PREFIXES = (
    ".poetry-cache/",
    ".venv",
    ".mypy_cache/",
    ".pytest_cache/",
    ".claude/worktrees/",
    ".codex",
    ".cursor/",
    "node_modules/",
    "htmlcov/",
    ".cache/",
    "target/",
    "docs/out/",
)

DEFAULT_ALLOWED_TOP_LEVEL = {
    ".github",
    "artifacts",
    "assets",
    "bundles",
    "config",
    "cpp",
    "data",
    "deployment",
    "devtools",
    "docs",
    "fixtures",
    "java",
    "marketmind_gate",
    "policies",
    "pysrc",
    "research",
    "run_bundles",
    "run_config",
    "runs",
    "schemas",
    "scripts",
    "stubs",
    "tests",
    "unit",
}


def _run_git(args: list[str], *, required: bool = True) -> list[str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if required:
            raise subprocess.CalledProcessError(
                proc.returncode,
                ["git", *args],
                output=proc.stdout,
                stderr=proc.stderr,
            )
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _normalize(path_str: str) -> str:
    return path_str.replace("\\", "/")


def _is_allowlisted_large(path: str) -> bool:
    # Large fixture snapshots and test model binaries are explicitly allowlisted.
    return (
        path.startswith("fixtures/")
        or path.startswith("tests/fixtures/")
        or path.startswith("patchedLibs/")
        or path.endswith(".pdb")
        or path.endswith(".dll")
    )


def _blocked(path: str) -> bool:
    norm = _normalize(path)
    return any(norm.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def _new_files(base_sha: str | None, staged: bool) -> list[str]:
    if staged:
        return _run_git(["diff", "--cached", "--name-only", "--diff-filter=A"])
    if base_sha:
        added = _run_git(
            ["diff", "--name-only", "--diff-filter=A", f"{base_sha}...HEAD"],
            required=False,
        )
        if added:
            return added
        sys.stderr.write(
            "repo_storage_guardrails WARNING: "
            f"could not diff against base {base_sha}; falling back to HEAD~1\n"
        )
    return _run_git(["diff", "--name-only", "--diff-filter=A", "HEAD~1...HEAD"])


def _all_tracked_files() -> list[str]:
    return _run_git(["ls-files"])


def _top_level(path: str) -> str:
    norm = _normalize(path)
    if "/" not in norm:
        return norm
    return norm.split("/", 1)[0]


def _check(mode: str, base_sha: str | None, lane: str) -> int:
    staged = mode == "staged"
    failures: list[str] = []
    warnings: list[str] = []

    added = _new_files(base_sha=base_sha, staged=staged)
    tracked = _all_tracked_files()
    added_set = set(added)
    for path in tracked:
        if _blocked(path):
            if path in added_set:
                failures.append(f"blocked tracked path: {path}")
            else:
                warnings.append(f"legacy blocked path already tracked (cleanup backlog): {path}")

    for path in added:
        top = _top_level(path)
        if top and top.startswith("."):
            # Hidden roots can still be valid if explicitly allowed.
            if top not in DEFAULT_ALLOWED_TOP_LEVEL:
                failures.append(f"new top-level hidden directory/file not allowlisted: {path}")
        elif top and top not in DEFAULT_ALLOWED_TOP_LEVEL and "." not in top:
            failures.append(f"new top-level directory not allowlisted: {top} (from {path})")

        if lane == "research" and (
            _normalize(path).startswith("docs/traces/")
            or _normalize(path).startswith("docs/manifests/")
        ):
            failures.append(f"research lane cannot add release artifact path: {path}")

        file_path = Path(path)
        if not file_path.is_file():
            continue
        size_bytes = file_path.stat().st_size
        if size_bytes > WARN_SIZE_BYTES:
            warnings.append(f"large file warning (>1MB): {path} ({size_bytes} bytes)")
        if size_bytes > FAIL_SIZE_BYTES and not _is_allowlisted_large(path):
            failures.append(f"large file blocked (>5MB): {path} ({size_bytes} bytes)")
        if (
            file_path.suffix.lower() in STRUCTURED_SUFFIXES
            and size_bytes > STRUCTURED_FAIL_SIZE_BYTES
            and not _is_allowlisted_large(path)
        ):
            failures.append(f"structured file blocked (>2MB): {path} ({size_bytes} bytes)")

    for line in warnings:
        sys.stderr.write(f"repo_storage_guardrails WARNING: {line}\n")
    for line in failures:
        sys.stderr.write(f"repo_storage_guardrails ERROR: {line}\n")

    if failures:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("staged", "ci"), default="ci")
    parser.add_argument("--base-sha", default=os.environ.get("GITHUB_BASE_SHA"))
    parser.add_argument(
        "--lane",
        choices=("research", "release"),
        default=os.environ.get("STORAGE_LANE", "research"),
    )
    args = parser.parse_args()
    return _check(mode=args.mode, base_sha=args.base_sha, lane=args.lane)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GoldenVectorError(RuntimeError):
    """Raised when ADR-007 golden-vector fixtures are missing or malformed."""


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One language-neutral golden-vector case.

    Fields in ``metadata`` are intentionally unopinionated so the same manifest
    shape can carry primitive-specific expectations: digest hex, output width,
    platform notes, xfail markers, or collision/equality policy details.
    """

    case_id: str
    path: Path
    media_type: str
    metadata: Mapping[str, Any]

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def read_json(self) -> Any:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class GoldenManifest:
    """Parsed ``manifest.json`` for one primitive suite."""

    suite: str
    version: str
    root: Path
    manifest_path: Path
    cases: tuple[GoldenCase, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GoldenSuite:
    """Loaded suite payloads plus parsed manifest.

    ``payloads`` maps ``case_id`` to the authoritative raw bytes. This is the
    data structure Python tests should consume when exact cross-language byte
    identity matters.
    """

    manifest: GoldenManifest
    payloads: Mapping[str, bytes]


def _repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "tests").exists() and (candidate / "py").exists():
            return candidate
    for candidate in [current, *current.parents]:
        if (candidate / "tests").exists():
            return candidate
    raise GoldenVectorError("Unable to locate repository root from fixture loader path.")


def default_golden_root(start: Path | None = None) -> Path:
    return _repo_root(start) / "tests" / "golden" / "adr007"


def suite_root(suite: str, *, root: Path | None = None) -> Path:
    return (root or default_golden_root()).resolve() / suite


def manifest_path(suite: str, *, root: Path | None = None) -> Path:
    return suite_root(suite, root=root) / "manifest.json"


def load_manifest(suite: str, *, root: Path | None = None) -> GoldenManifest:
    path = manifest_path(suite, root=root)
    if not path.exists():
        raise GoldenVectorError(f"Missing manifest for ADR-007 suite {suite!r}: expected {path}.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = str(raw.get("version", "1.0.0"))
    suite_name = str(raw.get("suite", suite))
    metadata = raw.get("metadata", {})
    case_specs = raw.get("cases")
    if not isinstance(case_specs, Sequence) or isinstance(case_specs, (str, bytes)):
        raise GoldenVectorError(f"Manifest {path} must contain a list-valued 'cases' field.")

    root_path = path.parent
    cases: list[GoldenCase] = []
    for index, spec in enumerate(case_specs):
        if not isinstance(spec, Mapping):
            raise GoldenVectorError(f"Manifest case #{index} in {path} is not a mapping.")
        case_id = spec.get("id")
        relative_path = spec.get("path")
        if not isinstance(case_id, str) or not case_id:
            raise GoldenVectorError(f"Manifest case #{index} in {path} is missing non-empty 'id'.")
        if not isinstance(relative_path, str) or not relative_path:
            raise GoldenVectorError(
                f"Manifest case {case_id!r} in {path} is missing non-empty 'path'."
            )
        case_path = (root_path / relative_path).resolve()
        if not case_path.exists():
            raise GoldenVectorError(
                f"Manifest case {case_id!r} points to missing payload {case_path}."
            )
        media_type = str(spec.get("media_type") or infer_media_type(case_path))
        case_metadata = {
            key: value for key, value in spec.items() if key not in {"id", "path", "media_type"}
        }
        cases.append(
            GoldenCase(
                case_id=case_id,
                path=case_path,
                media_type=media_type,
                metadata=case_metadata,
            )
        )

    return GoldenManifest(
        suite=suite_name,
        version=version,
        root=root_path,
        manifest_path=path,
        cases=tuple(cases),
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".bin": "application/octet-stream",
        ".json": "application/json",
        ".txt": "text/plain",
        ".hex": "text/plain",
    }.get(suffix, "application/octet-stream")


def load_case_bytes(case: GoldenCase) -> bytes:
    return case.read_bytes()


def load_suite(suite: str, *, root: Path | None = None) -> GoldenSuite:
    manifest = load_manifest(suite, root=root)
    payloads = {case.case_id: load_case_bytes(case) for case in manifest.cases}
    return GoldenSuite(manifest=manifest, payloads=payloads)


def iter_suite_cases(suite: str, *, root: Path | None = None) -> Iterator[GoldenCase]:
    manifest = load_manifest(suite, root=root)
    yield from manifest.cases

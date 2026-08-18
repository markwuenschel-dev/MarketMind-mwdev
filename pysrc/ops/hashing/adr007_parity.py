from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pysrc.ops.hashing.adr007_replay import ADR007_SUITES, replay_summary
from pysrc.ops.hashing.canonical_frame import CanonicalFrameCIStatus


@dataclass(frozen=True, slots=True)
class ParityLanguageResult:
    language: str
    success: bool
    suites: tuple[str, ...]
    toolchain: str
    details: dict[str, Any] = field(default_factory=dict)


def build_matrix_report(results: list[ParityLanguageResult]) -> dict[str, Any]:
    ordered_results = list(results)
    languages = [result.language for result in ordered_results]
    all_success = all(result.success for result in ordered_results)
    all_suites = sorted({suite for result in ordered_results for suite in result.suites})
    status = (
        CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value
        if all_success and len(ordered_results) == 3
        else CanonicalFrameCIStatus.PYTHON_ONLY_D2.value
    )
    return {
        "status": status,
        "cross_language_certified": status == CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value,
        "languages": languages,
        "suites": all_suites,
        "language_results": [
            {
                "language": result.language,
                "success": result.success,
                "toolchain": result.toolchain,
                "suite_count": len(result.suites),
                "details": result.details,
            }
            for result in ordered_results
        ],
    }


def build_python_language_result() -> ParityLanguageResult:
    summary = replay_summary()
    return ParityLanguageResult(
        language=str(summary["language"]),
        success=bool(summary["success"]),
        suites=tuple(str(suite) for suite in summary["suites"]),
        toolchain=str(summary["toolchain"]),
        details={
            "case_count": int(summary["case_count"]),
            "suite_count": int(summary["suite_count"]),
        },
    )


def build_language_report(result: ParityLanguageResult) -> dict[str, Any]:
    return {
        "language": result.language,
        "toolchain": result.toolchain,
        "suite_count": len(result.suites),
        "case_count": int(result.details.get("case_count", 0)),
        "success": result.success,
        "suites": list(result.suites),
        "details": result.details,
    }


def load_language_report(path: Path) -> ParityLanguageResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ParityLanguageResult(
        language=str(payload["language"]),
        success=bool(payload["success"]),
        suites=tuple(str(suite) for suite in payload.get("suites", ADR007_SUITES)),
        toolchain=str(payload["toolchain"]),
        details=dict(payload.get("details", {})),
    )


def aggregate_report_paths(paths: list[Path]) -> dict[str, Any]:
    results = [load_language_report(path) for path in paths]
    return build_matrix_report(results)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ADR-007 parity report helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    python_parser = subparsers.add_parser("python-report")
    python_parser.add_argument("--output", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--reports", nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "python-report":
        _write_json(Path(args.output), build_language_report(build_python_language_result()))
        return 0
    if args.command == "aggregate":
        _write_json(
            Path(args.output),
            aggregate_report_paths([Path(report) for report in args.reports]),
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

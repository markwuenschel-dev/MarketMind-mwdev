from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from pysrc.ops.hashing.canonical_frame import CanonicalFrameCIStatus

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d2")]


def test_build_matrix_report_tracks_cross_language_d2_certification(
    deterministic_seed: int,
) -> None:
    from pysrc.ops.hashing.adr007_parity import (
        ParityLanguageResult,
        build_matrix_report,
    )

    report = build_matrix_report(
        [
            ParityLanguageResult(
                language="python-3.12",
                success=True,
                suites=(
                    "blake3",
                    "hmac_sha256",
                    "jcs_sha256",
                    "minhash",
                    "rabin",
                    "simhash",
                    "sip24",
                    "xxh3",
                ),
                toolchain="Python 3.12",
                details={"seed": deterministic_seed},
            ),
            ParityLanguageResult(
                language="cpp-20",
                success=True,
                suites=(
                    "blake3",
                    "hmac_sha256",
                    "jcs_sha256",
                    "minhash",
                    "rabin",
                    "simhash",
                    "sip24",
                    "xxh3",
                ),
                toolchain="GCC 13",
                details={},
            ),
            ParityLanguageResult(
                language="java-21",
                success=True,
                suites=(
                    "blake3",
                    "hmac_sha256",
                    "jcs_sha256",
                    "minhash",
                    "rabin",
                    "simhash",
                    "sip24",
                    "xxh3",
                ),
                toolchain="Temurin 21",
                details={},
            ),
        ]
    )

    assert report["status"] == CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value
    assert report["cross_language_certified"] is True
    assert report["languages"] == ["python-3.12", "cpp-20", "java-21"]


def test_aggregate_report_paths_reads_real_language_reports(tmp_path: Path) -> None:
    from pysrc.ops.hashing.adr007_parity import aggregate_report_paths

    report_paths: list[Path] = []
    for language in ("python-3.13", "cpp-20", "java-21"):
        path = tmp_path / f"{language}.json"
        path.write_text(
            json.dumps(
                {
                    "language": language,
                    "toolchain": language,
                    "success": True,
                    "suites": ["blake3", "xxh3"],
                    "details": {"case_count": 2},
                }
            ),
            encoding="utf-8",
        )
        report_paths.append(path)

    report = aggregate_report_paths(report_paths)

    assert report["status"] == CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value
    assert report["cross_language_certified"] is True


def test_default_matrix_report_is_not_exported() -> None:
    module = importlib.import_module("pysrc.ops.hashing.adr007_parity")

    assert not hasattr(module, "default_matrix_report")

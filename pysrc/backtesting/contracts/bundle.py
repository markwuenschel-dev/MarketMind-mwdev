"""
py/backtesting/contracts/bundle.py

Typed dataclasses for the Appendix C run-bundle schema (v1.0.0).
These mirror the JSON files written by BundleWriter so that downstream
consumers (gate.py, report.py) can work with typed objects instead of
raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BUNDLE_SCHEMA_VERSION = "1.0.0"


@dataclass
class PlanRecord:
    """Mirrors plan.json — Gate 3 (plan_identity)."""

    schema_version: str
    plan_hash: str
    as_of_time: str
    config_hash: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvFingerprintRecord:
    """Mirrors env_fingerprint.json — Gate 4."""

    schema_version: str
    python_version: str
    platform: str
    git_sha: str
    deps: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetManifestRecord:
    """Mirrors dataset_manifest.json — Gate 5."""

    schema_version: str
    dataset_id: str
    symbols: list[str]
    row_count: int
    time_range: dict[str, str] = field(default_factory=dict)
    pit_compliant: bool = False
    knowledge_time_column: str = ""
    content_hash: str | None = None
    download_timestamp: str | None = None
    content_hash_expected: str | None = None


@dataclass
class PreprocessingReportRecord:
    """Mirrors preprocessing_report.json — Gate 6."""

    schema_version: str
    steps: list[dict[str, Any]]
    timings: dict[str, float]
    warnings: list[str]


@dataclass
class SplitRecord:
    """One split entry inside splits_manifest.json."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    fold: int = 0


@dataclass
class SplitsManifestRecord:
    """Mirrors splits_manifest.json — Gates 7–9."""

    schema_version: str
    split_method: str
    purge_window: int
    embargo_window: int
    splits: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunBundle:
    """Aggregates all five required gate files for a single run."""

    plan: PlanRecord
    env_fingerprint: EnvFingerprintRecord
    dataset_manifest: DatasetManifestRecord
    preprocessing_report: PreprocessingReportRecord
    splits_manifest: SplitsManifestRecord

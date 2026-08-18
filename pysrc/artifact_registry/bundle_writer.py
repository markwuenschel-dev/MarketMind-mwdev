"""
Canonical Appendix C bundle writer owned by ``pysrc.artifact_registry``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pysrc.artifact_registry import HashRefs, LocalCAS
from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.run_registry import RunRegistry
from pysrc.ops.telemetry import SPAN_ATTR_UNKNOWN, SPAN_BUNDLE_PROMOTE, tracer

if TYPE_CHECKING:
    from pysrc.backtesting.contracts.bundle import (
        DatasetManifestRecord,
        EnvFingerprintRecord,
        PlanRecord,
        PreprocessingReportRecord,
        SplitsManifestRecord,
    )

BUNDLE_SCHEMA_VERSION = "1.0.0"


def _git_sha() -> str:
    """Return the current HEAD git SHA, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _deps_snapshot() -> dict[str, str]:
    """Return installed package versions as {name: version}."""
    try:
        import importlib.metadata as meta

        return {
            distribution.metadata["Name"]: distribution.version
            for distribution in meta.distributions()
        }
    except Exception:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as indented JSON to path."""
    atomic_write_json(path, data)


class BundleWriter:
    """Writes an Appendix C-compliant run bundle directory."""

    SCHEMA_VERSION = BUNDLE_SCHEMA_VERSION

    def __init__(
        self,
        output_dir: Path,
        *,
        cas: LocalCAS | None = None,
        run_registry: RunRegistry | None = None,
        run_id: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._written: list[str] = []
        self._cas = cas
        self._run_registry = run_registry
        self._run_id = run_id
        self._role_hashes: dict[str, HashRefs] = {}
        self._role_paths: dict[str, str] = {}

    def write_plan(
        self,
        plan_hash: str,
        config_hash: str,
        as_of_time: str,
        config: dict[str, Any],
    ) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "plan_hash": plan_hash,
            "as_of_time": as_of_time,
            "config_hash": config_hash,
            "config": config,
        }
        self._write_and_register_json("plan.json", "plan", payload)

    def write_env_fingerprint(self) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_sha": _git_sha(),
            "deps": _deps_snapshot(),
        }
        self._write_and_register_json("env_fingerprint.json", "env_fingerprint", payload)

    def write_dataset_manifest(
        self,
        dataset_id: str,
        symbols: list[str],
        row_count: int,
        time_range: dict[str, str],
        *,
        pit_compliant: bool = True,
        knowledge_time_column: str = "knowledge_time",
        content_hash: str | None = None,
        download_timestamp: str | None = None,
        content_hash_expected: str | None = None,
    ) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "symbols": symbols,
            "row_count": row_count,
            "time_range": time_range,
            "pit_compliant": pit_compliant,
            "knowledge_time_column": knowledge_time_column,
        }
        if content_hash is not None:
            payload["content_hash"] = content_hash
        if download_timestamp is not None:
            payload["download_timestamp"] = download_timestamp
        if content_hash_expected is not None:
            payload["content_hash_expected"] = content_hash_expected
        self._write_and_register_json("dataset_manifest.json", "dataset_manifest", payload)

    def write_preprocessing_report(
        self,
        steps: list[dict[str, Any]],
        timings: dict[str, float],
        warnings: list[str],
    ) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "steps": steps,
            "timings": timings,
            "warnings": warnings,
        }
        self._write_and_register_json(
            "preprocessing_report.json",
            "preprocessing_report",
            payload,
        )

    def write_splits_manifest(
        self,
        splits: list[dict[str, Any]],
        split_method: str,
        purge_window: int,
        embargo_window: int,
    ) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "split_method": split_method,
            "purge_window": purge_window,
            "embargo_window": embargo_window,
            "splits": splits,
        }
        self._write_and_register_json("splits_manifest.json", "splits_manifest", payload)

    def write_stat_validity_report(self, report: dict[str, Any]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            **report,
        }
        self._write_and_register_json(
            "stat_validity_report.json",
            "stat_validity_report",
            payload,
        )

    def write_screening_report(self, payload: dict[str, Any]) -> None:
        self._write_and_register_json(
            "screening_report.json",
            "screening_report",
            payload,
        )

    def write_cleaning_plan(self, payload: dict[str, Any]) -> None:
        self._write_and_register_json(
            "cleaning_plan.json",
            "cleaning_plan",
            payload,
        )

    def write_cleaning_report(self, payload: dict[str, Any]) -> None:
        self._write_and_register_json(
            "cleaning_report.json",
            "cleaning_report",
            payload,
        )

    def missing_required(self) -> list[str]:
        required = [
            "plan.json",
            "env_fingerprint.json",
            "dataset_manifest.json",
            "preprocessing_report.json",
            "splits_manifest.json",
        ]
        return [filename for filename in required if filename not in self._written]

    @staticmethod
    def compute_config_hash(config: dict[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def write_bundle_manifest(self) -> None:
        with tracer.start_as_current_span(SPAN_BUNDLE_PROMOTE) as span:
            span.set_attribute("bundle_id", self.output_dir.name)
            strategy_id = SPAN_ATTR_UNKNOWN
            plan_path = self.output_dir / "plan.json"
            if plan_path.exists():
                try:
                    pdata = json.loads(plan_path.read_text(encoding="utf-8"))
                    cfg = pdata.get("config") or {}
                    raw_sid = cfg.get("strategy")
                    strategy_id = str(raw_sid) if raw_sid is not None else SPAN_ATTR_UNKNOWN
                except Exception:
                    strategy_id = SPAN_ATTR_UNKNOWN
            span.set_attribute("strategy_id", strategy_id)
            cas_parts = sorted(str(refs.cas) for refs in self._role_hashes.values())
            span.set_attribute("cas_hash", "|".join(cas_parts) if cas_parts else SPAN_ATTR_UNKNOWN)
            self._write_bundle_manifest_impl()

    def _write_bundle_manifest_impl(self) -> None:
        if not self._role_hashes:
            return

        artifacts: dict[str, dict[str, Any]] = {}
        for role, refs in self._role_hashes.items():
            path = self._role_paths.get(role)
            if path is None:
                continue
            artifacts[role] = {
                "path": path,
                "cas": str(refs.cas),
                "attest": str(refs.attest) if refs.attest is not None else None,
            }

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "hash_policy": {
                "cas": "cas.v1:b3-256",
                "attest": "attest.v1:jcs-sha256",
            },
            "artifacts": artifacts,
        }
        _write_json(self.output_dir / "bundle_manifest.json", manifest)
        self._written.append("bundle_manifest.json")

    def _write_and_register_json(self, filename: str, role: str, payload: dict[str, Any]) -> None:
        target = self.output_dir / filename

        if self._cas is not None:
            hash_refs = self._cas.put_json(payload)
            self._cas.materialize(hash_refs.cas, target)
            self._role_hashes[role] = hash_refs
            self._role_paths[role] = filename

            if self._run_registry is not None and self._run_id is not None:
                self._run_registry.add_artifact(self._run_id, role, hash_refs)
        else:
            _write_json(target, payload)

        self._written.append(filename)

    def read_plan(self) -> PlanRecord:
        from pysrc.backtesting.contracts.bundle import PlanRecord

        data = json.loads((self.output_dir / "plan.json").read_text())
        return PlanRecord(
            schema_version=data["schema_version"],
            plan_hash=data["plan_hash"],
            as_of_time=data["as_of_time"],
            config_hash=data["config_hash"],
            config=data.get("config", {}),
        )

    def read_env_fingerprint(self) -> EnvFingerprintRecord:
        from pysrc.backtesting.contracts.bundle import EnvFingerprintRecord

        data = json.loads((self.output_dir / "env_fingerprint.json").read_text())
        return EnvFingerprintRecord(
            schema_version=data["schema_version"],
            python_version=data["python_version"],
            platform=data["platform"],
            git_sha=data["git_sha"],
            deps=data.get("deps", {}),
        )

    def read_dataset_manifest(self) -> DatasetManifestRecord:
        from pysrc.backtesting.contracts.bundle import DatasetManifestRecord

        data = json.loads((self.output_dir / "dataset_manifest.json").read_text())
        return DatasetManifestRecord(
            schema_version=data["schema_version"],
            dataset_id=data["dataset_id"],
            symbols=data["symbols"],
            row_count=data["row_count"],
            time_range=data.get("time_range", {}),
            pit_compliant=bool(data.get("pit_compliant", False)),
            knowledge_time_column=str(data.get("knowledge_time_column", "")),
            content_hash=data.get("content_hash"),
            download_timestamp=data.get("download_timestamp"),
            content_hash_expected=data.get("content_hash_expected"),
        )

    def read_preprocessing_report(self) -> PreprocessingReportRecord:
        from pysrc.backtesting.contracts.bundle import PreprocessingReportRecord

        data = json.loads((self.output_dir / "preprocessing_report.json").read_text())
        return PreprocessingReportRecord(
            schema_version=data["schema_version"],
            steps=data["steps"],
            timings=data.get("timings", {}),
            warnings=data.get("warnings", []),
        )

    def read_splits_manifest(self) -> SplitsManifestRecord:
        from pysrc.backtesting.contracts.bundle import SplitsManifestRecord

        data = json.loads((self.output_dir / "splits_manifest.json").read_text())
        return SplitsManifestRecord(
            schema_version=data["schema_version"],
            split_method=data["split_method"],
            purge_window=data["purge_window"],
            embargo_window=data["embargo_window"],
            splits=data.get("splits", []),
        )


__all__ = ["BUNDLE_SCHEMA_VERSION", "BundleWriter"]

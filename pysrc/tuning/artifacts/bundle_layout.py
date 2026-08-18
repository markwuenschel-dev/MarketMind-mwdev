"""BundleLayout: canonical directory structure for a tuning job artifact bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BundlePath:
    """Resolved paths for all components of a tuning bundle."""

    root: Path
    manifest: Path
    trials: Path
    gate_results: Path
    promotion: Path
    reports: Path


@dataclass(frozen=True)
class BundleLayout:
    """Canonical layout for a job artifact bundle under a root directory."""

    job_id: str
    root: Path

    def resolve(self) -> BundlePath:
        """Return the resolved BundlePath for this bundle layout."""
        return BundlePath(
            root=self.root,
            manifest=self.root / "manifest.json",
            trials=self.root / "trials",
            gate_results=self.root / "gate_results",
            promotion=self.root / "promotion",
            reports=self.root / "reports",
        )


__all__ = ["BundlePath", "BundleLayout"]

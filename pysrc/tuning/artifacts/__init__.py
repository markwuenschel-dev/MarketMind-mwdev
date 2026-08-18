"""Content-addressed artifact persistence bridge between the tuning core and artifact registry."""

from pysrc.tuning.artifacts.bundle_layout import BundleLayout, BundlePath
from pysrc.tuning.artifacts.canonical_json import to_canonical_json
from pysrc.tuning.artifacts.lineage import LineageRecord, record_lineage
from pysrc.tuning.artifacts.manifest import ArtifactManifest, ManifestEntry
from pysrc.tuning.artifacts.metadata import ArtifactMetadata
from pysrc.tuning.artifacts.reader import ArtifactReader
from pysrc.tuning.artifacts.registry_bridge import RegistryBridge
from pysrc.tuning.artifacts.writer import ArtifactWriter

__all__ = [
    "ArtifactManifest",
    "ManifestEntry",
    "ArtifactMetadata",
    "ArtifactWriter",
    "ArtifactReader",
    "LineageRecord",
    "record_lineage",
    "to_canonical_json",
    "RegistryBridge",
    "BundleLayout",
    "BundlePath",
]

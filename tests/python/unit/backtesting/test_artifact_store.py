from __future__ import annotations

import pytest

from pysrc.artifact_registry import LocalCAS
from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.artifact_registry.run_registry import RunRegistry, RunStatus


@pytest.mark.determinism("d1")
def test_artifact_store_records_cas_refs_while_registering(tmp_path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    run_registry = RunRegistry(tmp_path / "registry")
    run_id = run_registry.begin_run(metadata={"kind": "test"})
    writer = BundleWriter(tmp_path / "bundle", cas=cas, run_registry=run_registry, run_id=run_id)
    store = BundleBacktestArtifactStore(writer)

    ref = store.put_json(
        "execution_assumptions.json", {"schema_version": "1.0.0", "commission_bps": 0.0}
    )
    writer.write_bundle_manifest()

    assert ref.cas is not None
    assert ref.attest is not None
    assert (tmp_path / "bundle" / "execution_assumptions.json").exists()
    assert (tmp_path / "bundle" / "bundle_manifest.json").exists()

    run_registry.finalize_run(run_id, RunStatus.COMPLETE)
    with pytest.raises(Exception):
        store.put_json("late.json", {"schema_version": "1.0.0"})

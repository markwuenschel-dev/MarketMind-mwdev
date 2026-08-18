"""IRFactory: build IR objects from Spec objects."""

from __future__ import annotations

import time

from pysrc.tuning.core.ir.nodes import IRMetadata
from pysrc.tuning.core.ir.search_ir import SearchIR
from pysrc.tuning.core.specs.search_space_spec import SearchSpaceSpec
from pysrc.tuning.core.specs.tuning_job_spec import TuningJobSpec

__all__ = ["build_ir_metadata", "build_search_ir"]


def build_ir_metadata(spec: TuningJobSpec) -> IRMetadata:
    """Derive IRMetadata from a TuningJobSpec."""
    return IRMetadata(
        spec_hash=spec.spec_hash,
        created_at_ns=time.monotonic_ns(),
        determinism_tier=spec.determinism_tier,
        tags=dict(spec.tags),
    )


def build_search_ir(
    job_spec: TuningJobSpec,
    space_spec: SearchSpaceSpec,
) -> SearchIR:
    """Build a SearchIR from a TuningJobSpec and SearchSpaceSpec."""
    meta = build_ir_metadata(job_spec)
    return SearchIR(
        job_id=job_spec.job_id,
        algorithm="unset",  # filled by search registry at runtime
        space_hash=space_spec.spec_hash,
        meta=meta,
    )

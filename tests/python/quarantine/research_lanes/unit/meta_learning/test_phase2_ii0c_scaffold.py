from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.phase2_ii0c_comparison import (
    II0CComparisonContext,
    build_ii0c_comparison_payload,
)
from pysrc.meta_learning.phase2_ii0c_encoder import II0CEncoderStub, Phase2II0CEncoderConfig
from pysrc.meta_learning.phase2_ii0c_tasks import (
    II0CMetaTaskRequest,
    build_ii0c_meta_task_payload,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d0")]


def _task_request() -> II0CMetaTaskRequest:
    return II0CMetaTaskRequest(
        regime_id="ii0c-scaffold-r1",
        regime_class="sideways",
        support_set=("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
        query_set=("2026-01-05T00:00:00+00:00",),
        horizon=1,
        signal_bindings={1: "signal.b", 0: "signal.a"},
        signal_set_version="ii0c.scaffold.signals.v1",
    )


@pytest.mark.determinism("d0")
def test_ii0c_meta_task_payload_is_deterministic_and_manifest_ready(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed

    first = build_ii0c_meta_task_payload(_task_request())
    second = build_ii0c_meta_task_payload(_task_request())

    assert first.meta_task.task_id == second.meta_task.task_id
    assert first.task_manifest_input.signal_ids_hash == first.meta_task.signal_ids_hash
    assert first.task_manifest_input.support_last_timestamp == first.meta_task.pit_boundary
    assert first.task_record["task_id"] == first.meta_task.task_id
    assert first.task_record["scaffold_only"] is True
    assert first.task_record["phase"] == "II-0C"
    assert first.meta_task.signal_ids[0] == "signal.a"
    assert first.meta_task.signal_ids[1] == "signal.b"


@pytest.mark.determinism("d0")
def test_ii0c_meta_task_payload_fails_closed_on_boundary_violation(deterministic_seed: int) -> None:
    _ = deterministic_seed
    req = II0CMetaTaskRequest(
        regime_id="ii0c-scaffold-r1",
        regime_class="sideways",
        support_set=("2026-01-01T00:00:00+00:00",),
        query_set=("2026-01-01T00:00:01+00:00",),
        horizon=1,
        signal_bindings={0: "signal.a"},
        signal_set_version="ii0c.scaffold.signals.v1",
    )

    with pytest.raises(DataPreconditionError, match="purge_window"):
        build_ii0c_meta_task_payload(req)


@pytest.mark.determinism("d0")
def test_ii0c_encoder_stub_is_deterministic_and_scaffold_only(deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = build_ii0c_meta_task_payload(_task_request()).meta_task
    encoder = II0CEncoderStub()

    first = encoder.encode_task(task)
    second = encoder.encode_task(task)

    assert first.embedding == second.embedding
    assert len(first.embedding) == encoder.config.embedding_dim
    assert first.metadata["phase"] == "II-0C"
    assert first.metadata["scaffold_only"] is True
    assert first.metadata["validated_representation_layer"] is False
    assert first.metadata["gate_ii_status"] == "DEFERRED"


@pytest.mark.determinism("d0")
def test_ii0c_encoder_task_output_respects_configured_dimension(deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = build_ii0c_meta_task_payload(_task_request()).meta_task
    encoder = II0CEncoderStub(Phase2II0CEncoderConfig(embedding_dim=17))

    output = encoder.encode_task(task)

    assert len(output.embedding) == 17
    assert output.metadata["embedding_dim"] == 17


@pytest.mark.determinism("d0")
def test_ii0c_comparison_payload_requires_xgboost_incumbent(deterministic_seed: int) -> None:
    _ = deterministic_seed
    payload = build_ii0c_comparison_payload(
        II0CComparisonContext(
            baseline_run_id="xgboost-incumbent:pilot",
            challenger_run_id="ii0c-challenger:pilot",
            incumbent_data_fingerprint="sha256:data",
            challenger_data_fingerprint="sha256:data",
            incumbent_splits_fingerprint="sha256:splits",
            challenger_splits_fingerprint="sha256:splits",
            incumbent_cost_assumptions_fingerprint="sha256:cost",
            challenger_cost_assumptions_fingerprint="sha256:cost",
        )
    )

    assert payload.baseline_comparison["baseline_kind"] == "xgboost_incumbent"
    assert payload.baseline_comparison["data_parity"] is True
    assert payload.shared_comparison_context["splits_fingerprint"] == "sha256:splits"


@pytest.mark.determinism("d0")
def test_ii0c_comparison_payload_rejects_rg09_anchor_as_baseline(deterministic_seed: int) -> None:
    _ = deterministic_seed

    with pytest.raises(DataPreconditionError, match="RG-09"):
        build_ii0c_comparison_payload(
            II0CComparisonContext(
                baseline_run_id="rg09-strict-h3-task-validity-anchor",
                challenger_run_id="ii0c-challenger:pilot",
                incumbent_data_fingerprint="sha256:data",
                challenger_data_fingerprint="sha256:data",
                incumbent_splits_fingerprint="sha256:splits",
                challenger_splits_fingerprint="sha256:splits",
                incumbent_cost_assumptions_fingerprint="sha256:cost",
                challenger_cost_assumptions_fingerprint="sha256:cost",
            )
        )


@pytest.mark.determinism("d0")
def test_ii0c_comparison_payload_rejects_unmatched_context(deterministic_seed: int) -> None:
    _ = deterministic_seed

    with pytest.raises(DataPreconditionError, match="data fingerprint"):
        build_ii0c_comparison_payload(
            II0CComparisonContext(
                baseline_run_id="xgboost-incumbent:pilot",
                challenger_run_id="ii0c-challenger:pilot",
                incumbent_data_fingerprint="sha256:data-a",
                challenger_data_fingerprint="sha256:data-b",
                incumbent_splits_fingerprint="sha256:splits",
                challenger_splits_fingerprint="sha256:splits",
                incumbent_cost_assumptions_fingerprint="sha256:cost",
                challenger_cost_assumptions_fingerprint="sha256:cost",
            )
        )

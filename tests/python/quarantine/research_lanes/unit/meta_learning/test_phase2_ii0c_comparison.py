"""II-0C challenger-vs-incumbent comparison contract tests."""

from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.phase2_artifact_contract import (
    _cross_check_baseline_vs_execution,
    _validate_baseline_comparison,
)
from pysrc.meta_learning.phase2_ii0c_comparison import (
    GOVERNED_BASELINE_COMPARISON_KEYS,
    II0C_COMPARISON_LANE,
    NOT_EVALUATED_NON_PROMOTABLE,
    XGBOOST_INCUMBENT_BASELINE_KIND,
    Phase2II0CComparisonError,
    Phase2II0CComparisonSpec,
    build_phase2_ii0c_baseline_comparison,
    build_phase2_ii0c_comparison_bundle,
    build_phase2_ii0c_shared_comparison_context,
    validate_ii0c_governed_baseline_and_shared_context,
    validate_phase2_ii0c_comparison_spec,
)


def _spec() -> Phase2II0CComparisonSpec:
    return Phase2II0CComparisonSpec(
        challenger_run_id="ii0c-challenger-001",
        baseline_run_id="xgboost-incumbent-001",
        splits_fingerprint="sha256:splits:001",
        data_fingerprint="sha256:data:001",
        cost_assumptions_fingerprint="sha256:cost:001",
    )


@pytest.mark.determinism("d1")
def test_build_phase2_ii0c_comparison_bundle_is_contract_compatible(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    bundle = build_phase2_ii0c_comparison_bundle(_spec())
    baseline = bundle.baseline_comparison
    shared = bundle.shared_comparison_context

    assert baseline["baseline_kind"] == XGBOOST_INCUMBENT_BASELINE_KIND
    assert baseline["net_result_against_incumbent"] == NOT_EVALUATED_NON_PROMOTABLE
    assert baseline["data_parity"] is True
    assert baseline["split_parity"] is True
    assert baseline["cost_parity"] is True

    assert shared["comparison_lane"] == II0C_COMPARISON_LANE
    assert shared["challenger_run_id"] == baseline["challenger_run_id"]
    assert shared["baseline_run_id"] == baseline["baseline_run_id"]
    assert shared["splits_fingerprint"] == baseline["splits_fingerprint"]
    assert shared["data_fingerprint"] == baseline["data_fingerprint"]
    assert shared["cost_assumptions_fingerprint"] == baseline["cost_assumptions_fingerprint"]

    _validate_baseline_comparison(baseline)
    _cross_check_baseline_vs_execution(baseline, shared)


@pytest.mark.determinism("d1")
def test_validate_phase2_ii0c_comparison_rejects_rg09_anchor(deterministic_seed: int) -> None:
    _ = deterministic_seed
    spec = Phase2II0CComparisonSpec(
        challenger_run_id="ii0c-challenger-002",
        baseline_run_id="rg09-strict-h3-task_validity_anchor",
        splits_fingerprint="sha256:splits:002",
        data_fingerprint="sha256:data:002",
        cost_assumptions_fingerprint="sha256:cost:002",
    )
    with pytest.raises(
        Phase2II0CComparisonError, match="must not be used as the incumbent baseline"
    ):
        validate_phase2_ii0c_comparison_spec(spec)


@pytest.mark.determinism("d1")
def test_validate_phase2_ii0c_comparison_rejects_baseline_kind_drift(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    spec = Phase2II0CComparisonSpec(
        challenger_run_id="ii0c-challenger-003",
        baseline_kind="rg09_reference_anchor",
        baseline_run_id="xgboost-incumbent-003",
        splits_fingerprint="sha256:splits:003",
        data_fingerprint="sha256:data:003",
        cost_assumptions_fingerprint="sha256:cost:003",
    )
    with pytest.raises(Phase2II0CComparisonError, match="XGBoost incumbent allocator baseline"):
        validate_phase2_ii0c_comparison_spec(spec)


@pytest.mark.determinism("d1")
def test_validate_phase2_ii0c_comparison_rejects_parity_drift(deterministic_seed: int) -> None:
    _ = deterministic_seed
    spec = Phase2II0CComparisonSpec(
        challenger_run_id="ii0c-challenger-004",
        baseline_run_id="xgboost-incumbent-004",
        splits_fingerprint="sha256:splits:004",
        data_fingerprint="sha256:data:004",
        cost_assumptions_fingerprint="sha256:cost:004",
        data_parity=False,
    )
    with pytest.raises(Phase2II0CComparisonError, match="data_parity must be True"):
        build_phase2_ii0c_baseline_comparison(spec)


@pytest.mark.determinism("d1")
def test_validate_ii0c_governed_baseline_rejects_scaffold_key_leakage(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    spec = _spec()
    baseline = build_phase2_ii0c_baseline_comparison(spec)
    shared = build_phase2_ii0c_shared_comparison_context(spec)
    baseline_bad = dict(baseline)
    baseline_bad["encoder_metadata"] = {"stub": True}
    with pytest.raises(DataPreconditionError, match="exactly the governed comparison keys"):
        validate_ii0c_governed_baseline_and_shared_context(
            baseline_comparison=baseline_bad,
            shared_comparison_context=shared,
        )


@pytest.mark.determinism("d1")
def test_validate_ii0c_governed_baseline_rejects_shared_fingerprint_drift(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    spec = _spec()
    baseline = build_phase2_ii0c_baseline_comparison(spec)
    shared = dict(build_phase2_ii0c_shared_comparison_context(spec))
    shared["data_fingerprint"] = "sha256:drift"
    with pytest.raises(DataPreconditionError, match="identical"):
        validate_ii0c_governed_baseline_and_shared_context(
            baseline_comparison=baseline,
            shared_comparison_context=shared,
        )


@pytest.mark.determinism("d1")
def test_governed_baseline_key_set_matches_bundle_builder(deterministic_seed: int) -> None:
    _ = deterministic_seed
    spec = _spec()
    baseline = build_phase2_ii0c_baseline_comparison(spec)
    assert frozenset(baseline.keys()) == GOVERNED_BASELINE_COMPARISON_KEYS


@pytest.mark.determinism("d1")
def test_shared_context_matches_baseline_contract_fields(deterministic_seed: int) -> None:
    _ = deterministic_seed
    spec = _spec()
    baseline = build_phase2_ii0c_baseline_comparison(spec)
    shared = build_phase2_ii0c_shared_comparison_context(spec)
    assert shared["challenger_run_id"] == baseline["challenger_run_id"]
    assert shared["baseline_run_id"] == baseline["baseline_run_id"]
    assert shared["splits_fingerprint"] == baseline["splits_fingerprint"]
    assert shared["data_fingerprint"] == baseline["data_fingerprint"]
    assert shared["cost_assumptions_fingerprint"] == baseline["cost_assumptions_fingerprint"]

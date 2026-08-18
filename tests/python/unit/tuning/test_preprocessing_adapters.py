from __future__ import annotations

import pytest

from pysrc.preprocessor.contracts.executor import ExecutionEvidence
from pysrc.preprocessor.contracts.plan import CanonicalOp, PreprocessingPlan
from pysrc.preprocessor.contracts.state import PreprocessingStateManifest
from pysrc.tuning.adapters.preprocessing_evidence_adapter import evidence_to_mapping
from pysrc.tuning.adapters.preprocessing_plan_adapter import normalize_preprocessing_plan
from pysrc.tuning.adapters.preprocessing_state_adapter import normalize_preprocessing_state


@pytest.mark.determinism("d0")
def test_plan_adapter_returns_canonical_preprocessing_plan(deterministic_seed: int) -> None:
    plan = normalize_preprocessing_plan(
        {
            "version": "1.0",
            "ops": [
                {
                    "name": "feature.sma",
                    "params": {"column": "close", "window": 10},
                    "provides": ["sma_10"],
                },
            ],
            "group_by": ["symbol"],
        }
    )

    assert isinstance(plan, PreprocessingPlan)
    assert plan.ops == (
        CanonicalOp(
            name="feature.sma", params={"column": "close", "window": 10}, provides=("sma_10",)
        ),
    )


@pytest.mark.determinism("d0")
def test_state_adapter_returns_canonical_preprocessing_state(deterministic_seed: int) -> None:
    state = normalize_preprocessing_state({"schema_version": "1.0", "plan_version": "1.0"})

    assert isinstance(state, PreprocessingStateManifest)
    assert str(state.state_id).startswith("cas.v1:b3-256:")


@pytest.mark.determinism("d0")
def test_evidence_adapter_exposes_canonical_evidence_without_redefining_it(
    deterministic_seed: int,
) -> None:
    evidence = ExecutionEvidence(events=("node_start", "node_finish"), metrics={"rows": 10})

    payload = evidence_to_mapping(evidence)

    assert payload == {"events": ["node_start", "node_finish"], "metrics": {"rows": 10}}

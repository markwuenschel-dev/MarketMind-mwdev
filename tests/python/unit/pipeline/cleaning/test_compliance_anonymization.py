from __future__ import annotations

import polars as pl
import pytest

from pysrc.core.errors import DataValidationError


@pytest.mark.determinism("d1")
def test_drift_compliance_uses_canonical_cleaning_pipeline(
    deterministic_seed: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = deterministic_seed
    calls: list[pl.DataFrame] = []

    class _FakePipeline:
        def run(self, df: pl.DataFrame) -> object:
            calls.append(df)
            return object()

    monkeypatch.setattr(
        "pysrc.pipeline.stages.cleaning.validators.compliance_checks.build_cleaning_pipeline",
        lambda spec: _FakePipeline(),
    )

    from pysrc.pipeline.stages.cleaning.validators.compliance_checks import DriftCompliance

    compliance = DriftCompliance(
        {
            "threshold": 0.1,
            "reference_data": pl.DataFrame({"close": [1.0, 1.1, 1.2]}),
            "strict": True,
        }
    )
    frame = pl.DataFrame({"close": [1.0, 1.05, 1.1]})

    out = compliance.apply(frame)

    assert out is frame
    assert calls
    assert calls[0].shape == frame.shape


@pytest.mark.determinism("d1")
def test_dp_anonymizer_rejects_governed_mode(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.cleaning.validators.anonymization import DPAnonymizer

    with pytest.raises(DataValidationError, match="non-governed only"):
        DPAnonymizer({"epsilon": 1.0, "governance_mode": "governed"})


@pytest.mark.determinism("d1")
def test_dp_anonymizer_seed_is_stable_for_nongoverned(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.cleaning.validators.anonymization import DPAnonymizer

    frame = pl.DataFrame({"value": [10.0, 20.0, 30.0]})
    first = DPAnonymizer({"epsilon": 1.0, "governance_mode": "nongoverned", "seed": 7}).apply(
        frame, ["value"]
    )
    second = DPAnonymizer({"epsilon": 1.0, "governance_mode": "nongoverned", "seed": 7}).apply(
        frame, ["value"]
    )

    assert first["value"].to_list() == second["value"].to_list()

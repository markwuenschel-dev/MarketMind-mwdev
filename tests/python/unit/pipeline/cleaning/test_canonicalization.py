from __future__ import annotations

import polars as pl
import pytest

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning import (
    CleaningMutationSummary,
    CleaningRuntimeContext,
    build_cleaning_pipeline,
    list_registered_cleaning_steps,
)
from pysrc.pipeline.stages.cleaning.core import factory as cleaning_factory


class _ExplodingAltDataProvider:
    def materialize(self, df: pl.DataFrame, *, context: CleaningRuntimeContext, params):
        del df, context, params
        raise RuntimeError("provider unavailable")


@pytest.mark.determinism("d0")
def test_cleaning_plan_hash_stable(deterministic_seed: int):
    _ = deterministic_seed
    spec = {
        "seed_lineage": "tests.cleaning.plan",
        "steps": [
            {
                "step_id": "impute.missing",
                "step_type": "impute.missing",
                "version": "1",
                "params": {"method": "forward_fill", "backward_fill": True},
            }
        ],
    }
    first = build_cleaning_pipeline(spec)
    second = build_cleaning_pipeline(spec)
    assert first.plan_hash == second.plan_hash
    assert first.registry_state_hash == second.registry_state_hash


@pytest.mark.determinism("d0")
def test_cleaning_plan_hash_binds_registry_state(
    deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
):
    _ = deterministic_seed
    spec = {
        "seed_lineage": "tests.cleaning.plan.registry",
        "steps": [
            {
                "step_id": "impute.missing",
                "step_type": "impute.missing",
                "version": "1",
                "params": {"method": "forward_fill", "backward_fill": True},
            }
        ],
    }

    monkeypatch.setattr(cleaning_factory, "registry_state_hash", lambda: "registry-a")
    first = build_cleaning_pipeline(spec)
    monkeypatch.setattr(cleaning_factory, "registry_state_hash", lambda: "registry-b")
    second = build_cleaning_pipeline(spec)

    assert first.plan_hash != second.plan_hash
    assert first.registry_state_hash != second.registry_state_hash


@pytest.mark.determinism("d0")
def test_registry_list_contains_expected_ids(deterministic_seed: int):
    _ = deterministic_seed
    step_types = {registration.step_type for registration in list_registered_cleaning_steps()}
    assert "impute.missing" in step_types
    assert "feature.technical.rsi" in step_types


@pytest.mark.determinism("d1")
def test_unknown_cleaning_step_fails_closed(deterministic_seed: int):
    _ = deterministic_seed
    with pytest.raises(DataValidationError, match="Unknown cleaning step registration"):
        build_cleaning_pipeline(
            {
                "steps": [
                    {
                        "step_id": "unknown",
                        "step_type": "feature.unknown",
                        "version": "1",
                    }
                ]
            }
        )


@pytest.mark.determinism("d1")
def test_governed_provider_missing_fails_closed(deterministic_seed: int):
    _ = deterministic_seed
    pipeline = build_cleaning_pipeline(
        {
            "steps": [
                {
                    "step_id": "feature.altdata",
                    "step_type": "feature.altdata",
                    "version": "1",
                    "params": {
                        "provider_key": "altdata",
                        "output_columns": ["alt_signal"],
                    },
                }
            ]
        }
    )
    context = CleaningRuntimeContext(
        run_id="missing-provider",
        determinism_tier=pipeline.spec.determinism_tier,
        seed_lineage=pipeline.spec.seed_lineage,
        pit_boundary="2026-04-08",
        governance_mode=pipeline.spec.governance_mode,
        providers={},
        registry_state_hash=pipeline.registry_state_hash,
    )
    with pytest.raises(DataValidationError, match="requires unavailable providers"):
        pipeline.run(pl.DataFrame({"close": [1.0, 2.0]}), context=context)


@pytest.mark.determinism("d1")
def test_explicit_altdata_fallback_is_reported(deterministic_seed: int):
    _ = deterministic_seed
    pipeline = build_cleaning_pipeline(
        {
            "seed_lineage": "tests.cleaning.altdata",
            "steps": [
                {
                    "step_id": "feature.altdata",
                    "step_type": "feature.altdata",
                    "version": "1",
                    "params": {
                        "provider_key": "altdata",
                        "output_columns": ["alt_signal"],
                    },
                    "fallback_policy": {
                        "mode": "constant",
                        "values": {"alt_signal": 0.0},
                    },
                }
            ],
        }
    )
    context = CleaningRuntimeContext(
        run_id="altdata-fallback",
        determinism_tier=pipeline.spec.determinism_tier,
        seed_lineage=pipeline.spec.seed_lineage,
        pit_boundary="2026-04-08",
        governance_mode=pipeline.spec.governance_mode,
        providers={"altdata": _ExplodingAltDataProvider()},
        registry_state_hash=pipeline.registry_state_hash,
    )

    result = pipeline.run(pl.DataFrame({"close": [1.0, 2.0]}), context=context)
    report = pipeline.to_report_payload(result, context=context)

    assert result.frame["alt_signal"].to_list() == [0.0, 0.0]
    assert report["fallback_events"] == [{"provider": "altdata", "mode": "constant"}]


@pytest.mark.determinism("d1")
def test_mutation_summary_is_structured_for_imputation(deterministic_seed: int):
    _ = deterministic_seed
    pipeline = build_cleaning_pipeline(
        {
            "steps": [
                {
                    "step_id": "impute.missing",
                    "step_type": "impute.missing",
                    "version": "1",
                    "params": {"method": "forward_fill", "backward_fill": True},
                }
            ]
        }
    )
    result = pipeline.run(pl.DataFrame({"close": [1.0, None, 2.0]}))

    assert isinstance(result.mutation, CleaningMutationSummary)
    assert result.mutation.rows_in == 3
    assert result.mutation.rows_out == 3
    assert result.mutation.rows_removed == 0
    assert result.mutation.cells_mutated == 1
    assert result.mutation.rows_with_mutations >= 1


@pytest.mark.determinism("d1")
def test_mutation_summary_is_structured_for_additive_features(deterministic_seed: int):
    _ = deterministic_seed
    pipeline = build_cleaning_pipeline(
        {
            "steps": [
                {
                    "step_id": "feature.calendar.global_calendar",
                    "step_type": "feature.calendar.global_calendar",
                    "version": "1",
                    "params": {
                        "countries": [],
                        "day_of_week": True,
                        "is_holiday": False,
                        "timestamp_col": "timestamp",
                    },
                }
            ]
        }
    )
    result = pipeline.run(
        pl.DataFrame(
            {
                "timestamp": ["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
                "close": [1.0, 2.0],
            }
        )
    )

    assert result.mutation.rows_in == 2
    assert result.mutation.rows_out == 2
    assert result.mutation.rows_removed == 0
    assert result.mutation.rows_with_mutations == 2
    assert result.mutation.cells_mutated == 2


@pytest.mark.determinism("d0")
def test_cleaning_report_payload_deterministic(deterministic_seed: int):
    _ = deterministic_seed
    pipeline = build_cleaning_pipeline(
        {
            "seed_lineage": "tests.cleaning.report",
            "steps": [
                {
                    "step_id": "impute.missing",
                    "step_type": "impute.missing",
                    "version": "1",
                    "params": {"method": "forward_fill", "backward_fill": True},
                }
            ],
        }
    )
    context = CleaningRuntimeContext(
        run_id="report",
        determinism_tier=pipeline.spec.determinism_tier,
        seed_lineage=pipeline.spec.seed_lineage,
        pit_boundary="2026-04-08",
        governance_mode=pipeline.spec.governance_mode,
        providers={},
        registry_state_hash=pipeline.registry_state_hash,
    )
    frame = pl.DataFrame({"x": [1.0, None, 2.0]})

    first = pipeline.to_report_payload(pipeline.run(frame, context=context), context=context)
    second = pipeline.to_report_payload(pipeline.run(frame, context=context), context=context)

    assert pipeline.to_plan_payload() == pipeline.to_plan_payload()
    assert first == second


def _schema_validation_pipeline():
    return build_cleaning_pipeline(
        {
            "steps": [
                {
                    "step_id": "validate.schema",
                    "step_type": "validate.schema",
                    "version": "1",
                    "params": {"ohlcv_mode": True, "strict": False},
                }
            ]
        }
    )


@pytest.mark.determinism("d1")
def test_validate_schema_accepts_multirow_nonnegative_ohlcv(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pipeline = _schema_validation_pipeline()

    result = pipeline.run(
        pl.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [100, 200, 300],
            }
        )
    )

    assert result.frame.height == 3


@pytest.mark.determinism("d1")
def test_validate_schema_rejects_any_negative_price(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pipeline = _schema_validation_pipeline()

    with pytest.raises(DataValidationError, match="Negative price values"):
        pipeline.run(
            pl.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [2.0, 3.0, 4.0],
                    "low": [0.5, -1.5, 2.5],
                    "close": [1.5, 2.5, 3.5],
                    "volume": [100, 200, 300],
                }
            )
        )


@pytest.mark.determinism("d1")
def test_validate_schema_ignores_nulls_in_negative_price_reduction(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pipeline = _schema_validation_pipeline()

    result = pipeline.run(
        pl.DataFrame(
            {
                "open": [1.0, None, 3.0],
                "high": [2.0, 3.0, None],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, None],
                "volume": [100, 200, 300],
            }
        )
    )

    assert result.frame.height == 3

from __future__ import annotations

import pytest

from pysrc.meta_learning.reports.meta_validity_report import (
    MetaValidityReportBuilder,
    MetaValidityReportBuildError,
    build_meta_validity_report,
    scaffold_confidence_calibration,
    scaffold_inner_loop_gain,
    scaffold_task_pool_counts,
    validate_confidence_calibration_block,
    validate_inner_loop_gain_block,
    validate_meta_validity_report_keys,
    validate_task_pool_counts_block,
)


def test_validate_keys_missing() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="missing keys"):
        validate_meta_validity_report_keys({"schema_version": "v1"})


def test_validate_keys_extra_top_level() -> None:
    payload = build_meta_validity_report(
        schema_version="v1",
        run_id="run.x",
        overall_result="PASS",
        reporting_gate="MLC3_SCAFFOLD",
        inner_loop_gain=None,
        shuffle_test_p_value=None,
        proxy_IC_pearson_r=None,
        crisis_holdout_ic=None,
        forgetting_ic_degradation_pct=None,
        task_pool_counts=None,
        confidence_calibration=None,
        fail_reasons=[],
        theta_day_prime_promoted=True,
        timestamp_utc="2020-01-01T00:00:00Z",
    )
    payload["illegal_extra"] = 1
    with pytest.raises(MetaValidityReportBuildError, match="unknown keys"):
        validate_meta_validity_report_keys(payload)


def test_build_rejects_empty_run_id() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="run_id"):
        build_meta_validity_report(
            schema_version="v1",
            run_id="",
            overall_result="PASS",
            reporting_gate="g",
            inner_loop_gain=None,
            shuffle_test_p_value=None,
            proxy_IC_pearson_r=None,
            crisis_holdout_ic=None,
            forgetting_ic_degradation_pct=None,
            task_pool_counts=None,
            confidence_calibration=None,
            fail_reasons=[],
            theta_day_prime_promoted=False,
        )


def test_build_rejects_bad_overall_result() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="overall_result"):
        build_meta_validity_report(
            schema_version="v1",
            run_id="run.x",
            overall_result="MAYBE",
            reporting_gate="g",
            inner_loop_gain=None,
            shuffle_test_p_value=None,
            proxy_IC_pearson_r=None,
            crisis_holdout_ic=None,
            forgetting_ic_degradation_pct=None,
            task_pool_counts=None,
            confidence_calibration=None,
            fail_reasons=["x"],
            theta_day_prime_promoted=False,
        )


def test_inner_loop_gain_wrong_keys() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="inner_loop_gain keys"):
        validate_inner_loop_gain_block({"mean_query_ic": None})


def test_inner_loop_gain_bad_mean_type() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="mean_query_ic"):
        validate_inner_loop_gain_block(
            {"mean_query_ic": "x", "harvey_t": None, "by_regime_class": None}
        )


def test_inner_loop_gain_by_regime_bad_value() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="by_regime_class"):
        validate_inner_loop_gain_block(
            {"mean_query_ic": 0.1, "harvey_t": None, "by_regime_class": {"bull": "nope"}}
        )


def test_task_pool_counts_missing_bucket_counts() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="missing keys"):
        validate_task_pool_counts_block({"batch_size": 1, "crisis_count": 0, "crisis_required": 0})


def test_task_pool_counts_extra_key() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="unknown keys"):
        validate_task_pool_counts_block(
            {
                "batch_size": 1,
                "crisis_count": 0,
                "crisis_required": 0,
                "bucket_counts": {},
                "extra": 1,
            }
        )


def test_task_pool_counts_bad_phase_type() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="phase"):
        validate_task_pool_counts_block(
            {
                "batch_size": 1,
                "crisis_count": 0,
                "crisis_required": 0,
                "bucket_counts": {},
                "phase": 123,
            }
        )


def test_confidence_calibration_wrong_keys() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="confidence_calibration keys"):
        validate_confidence_calibration_block({"ece": 0.1})


def test_confidence_calibration_bad_ece() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="ece"):
        validate_confidence_calibration_block({"ece": "bad", "note": None})


def test_scaffold_helpers_round_trip() -> None:
    ilg = scaffold_inner_loop_gain(
        mean_query_ic=0.05, harvey_t=None, by_regime_class={"crisis": 0.1}
    )
    tpc = scaffold_task_pool_counts(
        batch_size=4,
        crisis_count=2,
        crisis_required=1,
        bucket_counts={"crisis": 2, "bull": 2},
        phase="bootstrap",
    )
    cc = scaffold_confidence_calibration(ece=None, note="stub")
    doc = build_meta_validity_report(
        schema_version="v1",
        run_id="run.sha256:ab",
        overall_result="PASS",
        reporting_gate="MLC3_SCAFFOLD",
        inner_loop_gain=ilg,
        shuffle_test_p_value=None,
        proxy_IC_pearson_r=None,
        crisis_holdout_ic=None,
        forgetting_ic_degradation_pct=None,
        task_pool_counts=tpc,
        confidence_calibration=cc,
        fail_reasons=[],
        theta_day_prime_promoted=True,
        timestamp_utc="2020-01-02T00:00:00Z",
    )
    validate_meta_validity_report_keys(doc)


def test_meta_validity_report_builder_emits_same_as_function() -> None:
    b = MetaValidityReportBuilder(
        schema_version="v1",
        run_id="run.builder",
        overall_result="FAIL",
        reporting_gate="MLC3_SCAFFOLD",
        inner_loop_gain=scaffold_inner_loop_gain(mean_query_ic=None),
        fail_reasons=["INSUFFICIENT_CRISIS_TASKS"],
        theta_day_prime_promoted=False,
        timestamp_utc="2020-01-03T00:00:00Z",
    )
    b.task_pool_counts = scaffold_task_pool_counts(
        batch_size=8, crisis_count=0, crisis_required=1, bucket_counts={"bull": 8}
    )
    direct = build_meta_validity_report(
        schema_version="v1",
        run_id="run.builder",
        overall_result="FAIL",
        reporting_gate="MLC3_SCAFFOLD",
        inner_loop_gain=scaffold_inner_loop_gain(mean_query_ic=None),
        shuffle_test_p_value=None,
        proxy_IC_pearson_r=None,
        crisis_holdout_ic=None,
        forgetting_ic_degradation_pct=None,
        task_pool_counts=b.task_pool_counts,
        confidence_calibration=None,
        fail_reasons=["INSUFFICIENT_CRISIS_TASKS"],
        theta_day_prime_promoted=False,
        timestamp_utc="2020-01-03T00:00:00Z",
    )
    assert b.build() == direct


def test_inner_loop_gain_non_mapping() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="inner_loop_gain must be a mapping"):
        validate_inner_loop_gain_block([])


def test_by_regime_class_not_mapping() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="by_regime_class"):
        validate_inner_loop_gain_block(
            {"mean_query_ic": 1.0, "harvey_t": None, "by_regime_class": [1]}
        )


def test_inner_loop_gain_bad_harvey_t_type() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="harvey_t"):
        validate_inner_loop_gain_block(
            {"mean_query_ic": 0.1, "harvey_t": "x", "by_regime_class": None}
        )


def test_inner_loop_gain_harvey_t_numeric_with_null_by_regime() -> None:
    o = validate_inner_loop_gain_block(
        {"mean_query_ic": None, "harvey_t": 1.5, "by_regime_class": None}
    )
    assert o["harvey_t"] == 1.5
    assert o["by_regime_class"] is None


def test_by_regime_class_key_not_str() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="by_regime_class keys"):
        validate_inner_loop_gain_block(
            {"mean_query_ic": 0.0, "harvey_t": None, "by_regime_class": {1: 0.1}}
        )


def test_task_pool_counts_not_mapping() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="task_pool_counts must be a mapping"):
        validate_task_pool_counts_block([])


def test_task_pool_counts_bad_batch_size() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="batch_size"):
        validate_task_pool_counts_block(
            {"batch_size": "n", "crisis_count": 0, "crisis_required": 0, "bucket_counts": {}}
        )


def test_task_pool_counts_bad_crisis_count() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="crisis_count"):
        validate_task_pool_counts_block(
            {"batch_size": 1, "crisis_count": -1, "crisis_required": 0, "bucket_counts": {}}
        )


def test_task_pool_counts_bad_crisis_required() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="crisis_required"):
        validate_task_pool_counts_block(
            {"batch_size": 1, "crisis_count": 0, "crisis_required": 1.5, "bucket_counts": {}}
        )


def test_task_pool_counts_bucket_counts_not_mapping() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="bucket_counts must be a mapping"):
        validate_task_pool_counts_block(
            {"batch_size": 1, "crisis_count": 0, "crisis_required": 0, "bucket_counts": []}
        )


def test_task_pool_counts_bucket_key_not_str() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="bucket_counts keys"):
        validate_task_pool_counts_block(
            {"batch_size": 1, "crisis_count": 0, "crisis_required": 0, "bucket_counts": {1: 1}}
        )


def test_task_pool_counts_bucket_value_bad() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="bucket_counts"):
        validate_task_pool_counts_block(
            {
                "batch_size": 1,
                "crisis_count": 0,
                "crisis_required": 0,
                "bucket_counts": {"bull": -1},
            }
        )


def test_confidence_calibration_not_mapping() -> None:
    with pytest.raises(
        MetaValidityReportBuildError, match="confidence_calibration must be a mapping"
    ):
        validate_confidence_calibration_block("x")


def test_confidence_calibration_bad_note_type() -> None:
    with pytest.raises(MetaValidityReportBuildError, match="note"):
        validate_confidence_calibration_block({"ece": None, "note": 3})

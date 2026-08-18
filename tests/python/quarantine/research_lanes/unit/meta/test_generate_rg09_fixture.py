"""Unit tests for the RG-09 replay fixture generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from scripts.generate_rg09_fixture import (
    CANONICAL_COLUMNS,
    FIXTURE_COLUMNS,
    FIXTURE_FILENAME_V2,
    DataPreconditionError,
    FixtureSegmentSpec,
    MultiManifestSettings,
    _collapse_to_uniform_calendar_days,
    _compute_fixture_sha256,
    _record_to_row,
    generate_fixture,
    generate_fixture_multi,
    load_multi_manifest,
    validate_rg09_fixture_fold_geometry,
)

from pysrc.meta.bocpd_service import RegimeLabelRecord
from pysrc.meta.rg09_parquet_io import write_rg09_fixture_parquet
from pysrc.meta_learning.regime_vocabulary import projection_rule_version_id

_REGIME_CLASSES = {"bull", "bear", "sideways", "high_vol", "crisis"}


def _config_payload(*, crisis_percentile: float = 80.0) -> dict[str, Any]:
    return {
        "hazard_rate": 0.01,
        "observation_model": "student_t",
        "prior_mu0": None,
        "prior_kappa0": 1.0,
        "prior_alpha0": 3.0,
        "prior_beta0": None,
        "max_run_length": 128,
        "vol_window": 5,
        "trend_window": 7,
        "trend_flat_epsilon": 0.01,
        "vol_bucket_method": "tercile",
        "cp_threshold": 0.5,
        "transition_threshold": 0.3,
        "transition_max_rl": 5,
        "cold_start_burn_in": 12,
        "crisis_vol_score_percentile": crisis_percentile,
        "config_version": "rg09_v1.0.2",
    }


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _synthetic_input(*, periods: int = 80, drop_idx: int | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods, tz="UTC")
    close = 100.0 + np.cumsum(
        np.sin(np.linspace(0.0, 9.0, periods)) * 0.8 + np.linspace(0.05, 0.2, periods)
    )
    df = pd.DataFrame({"date": dates, "close": close})
    if drop_idx is not None:
        df = df.drop(index=drop_idx).reset_index(drop=True)
    return df


def _synthetic_input_with_duplicate_date() -> pd.DataFrame:
    df = _synthetic_input()
    duplicate = df.iloc[[10]].copy()
    return pd.concat([df.iloc[:11], duplicate, df.iloc[11:]], ignore_index=True)


def _write_input(path: Path, df: pd.DataFrame) -> Path:
    write_rg09_fixture_parquet(df, path, index=False)
    return path


def _run_generator(
    tmp_path: Path,
    *,
    entity_id: str = "ES",
    input_df: pd.DataFrame | None = None,
    config_payload: dict[str, Any] | None = None,
    output_dir_name: str = "out",
) -> dict[str, Any]:
    frame = _synthetic_input() if input_df is None else input_df
    input_path = _write_input(tmp_path / "input.parquet", frame)
    config_path = _write_config(
        tmp_path / "config.json",
        _config_payload() if config_payload is None else config_payload,
    )
    return generate_fixture(
        input_path=input_path,
        config_path=config_path,
        output_dir=tmp_path / output_dir_name,
        entity_id=entity_id,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _non_cold_start_contiguous_regime_count(frame: pd.DataFrame) -> int:
    sub = frame.loc[frame["boundary_flag"] != "cold_start", "regime_id"].astype(str).tolist()
    if not sub:
        return 0
    return int(1 + sum(cur != prev for prev, cur in zip(sub, sub[1:], strict=False)))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fixture_emits_all_required_columns(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    frame = pd.read_parquet(Path(result["fixture_path"]))
    assert list(frame.columns) == list(FIXTURE_COLUMNS)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_record_to_row_maps_regime_fields_from_expected_record_attributes(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    ts = pd.Timestamp("2024-01-02T00:00:00+00:00").to_pydatetime()
    record = RegimeLabelRecord(
        entity_id="ES",
        decision_ts=ts,
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_label="bull",
        effective_at=ts,
        state_snapshot_id="sha256:state",
        input_snapshot_id="sha256:input",
        config_version="rg09_v1.0.2",
        change_probability=0.25,
        boundary_flag="stable",
        regime_class="bear",
        diag_regime_class_bocpd_gated="high_vol",
        diag_regime_class_extended="crisis",
        run_length_mode=3,
        run_length_expectation=4.5,
        transition_probability=0.2,
        posterior_entropy=0.1,
        trend_score_raw=0.01,
        vol_score_raw=0.5,
    )
    row = _record_to_row(record)
    assert row["regime_id"] == record.regime_id
    assert row["regime_label"] == record.regime_label
    assert row["regime_class"] == record.regime_class


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_diag_bocpd_gated_column_present_and_typed(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    frame = pd.read_parquet(Path(result["fixture_path"]))
    observed = set(frame["diag_regime_class_bocpd_gated"].astype(str))
    assert observed
    assert observed <= _REGIME_CLASSES


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fixture_sha_excludes_diagnostic_columns(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    frame = pd.read_parquet(Path(result["fixture_path"]))
    original_sha = _compute_fixture_sha256(frame)
    mutated = frame.copy()
    mutated["diag_regime_class_bocpd_gated"] = "crisis"
    mutated["run_length_mode"] = mutated["run_length_mode"].astype(float) + 1.0
    mutated["vol_score_raw"] = mutated["vol_score_raw"].astype(float) + 99.0
    assert _compute_fixture_sha256(mutated) == original_sha


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fixture_sha_deterministic_across_runs(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    input_df = _synthetic_input()
    config_payload = _config_payload()
    result_a = _run_generator(
        tmp_path / "a",
        input_df=input_df,
        config_payload=config_payload,
        output_dir_name="fixture_a",
    )
    result_b = _run_generator(
        tmp_path / "b",
        input_df=input_df,
        config_payload=config_payload,
        output_dir_name="fixture_b",
    )
    metadata_a = _read_json(Path(result_a["metadata_path"]))
    metadata_b = _read_json(Path(result_b["metadata_path"]))
    assert metadata_a["fixture_sha256"] == metadata_b["fixture_sha256"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_summary_contains_bocpd_gated_counts(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    summary = _read_json(Path(result["summary_path"]))
    assert set(summary["row_counts_by_class_bocpd_gated"]) == _REGIME_CLASSES


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_summary_crisis_agreement_rate_in_unit_interval(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    summary = _read_json(Path(result["summary_path"]))
    assert 0.0 <= float(summary["crisis_label_agreement_rate"]) <= 1.0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_summary_includes_projection_rule_version_fields(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    summary = _read_json(Path(result["summary_path"]))
    pct = float(_config_payload()["crisis_vol_score_percentile"])
    assert summary["projection_rule_version"] == projection_rule_version_id(severity_percentile=pct)
    assert summary["projection_rule_logic_id"] == "mln02.v1.level2_severity_gate"
    assert summary["projection_rule_reference_bocpd_id"] == "mln02.reference.ii0a_bocpd_cp_vol_hi"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cold_start_rows_excluded_from_episode_count(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    frame = pd.read_parquet(Path(result["fixture_path"]))
    summary = _read_json(Path(result["summary_path"]))
    expected = _non_cold_start_contiguous_regime_count(frame)
    assert sum(summary["episode_counts_by_regime_id"].values()) == expected
    assert int(summary["cold_start_rows"]) == int((frame["boundary_flag"] == "cold_start").sum())


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_data_precondition_error_on_missing_input(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    missing_input = tmp_path / "missing.parquet"
    config_path = _write_config(tmp_path / "config.json", _config_payload())
    with pytest.raises(DataPreconditionError, match="acquisition is out of scope for OI-43"):
        generate_fixture(
            input_path=missing_input,
            config_path=config_path,
            output_dir=tmp_path / "out",
            entity_id="ES",
        )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_data_precondition_error_on_missing_columns(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    input_path = _write_input(
        tmp_path / "input.parquet",
        pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5, tz="UTC")}),
    )
    config_path = _write_config(tmp_path / "config.json", _config_payload())
    with pytest.raises(DataPreconditionError, match="missing required columns"):
        generate_fixture(
            input_path=input_path,
            config_path=config_path,
            output_dir=tmp_path / "out",
            entity_id="ES",
        )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_data_precondition_error_on_continuity_violation(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    input_path = _write_input(tmp_path / "input.parquet", _synthetic_input_with_duplicate_date())
    config_path = _write_config(tmp_path / "config.json", _config_payload())
    with pytest.raises(DataPreconditionError, match="date continuity"):
        generate_fixture(
            input_path=input_path,
            config_path=config_path,
            output_dir=tmp_path / "out",
            entity_id="ES",
        )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_es_only_sufficient_false_when_crisis_sparse(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    summary = _read_json(Path(result["summary_path"]))
    assert summary["es_only_sufficient"] is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_summary_and_metadata_fixture_sha_match(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    summary = _read_json(Path(result["summary_path"]))
    metadata = _read_json(Path(result["metadata_path"]))
    assert summary["fixture_sha256"] == metadata["fixture_sha256"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_canonical_column_mutation_changes_fixture_sha(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    result = _run_generator(tmp_path)
    frame = pd.read_parquet(Path(result["fixture_path"]))
    original_sha = _compute_fixture_sha256(frame)
    mutated = frame.copy()
    mutated.loc[mutated.index[0], CANONICAL_COLUMNS[-1]] = "change_point"
    assert _compute_fixture_sha256(mutated) != original_sha


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_collapse_uniform_calendar_days_advances_cursor(deterministic_seed: int) -> None:
    _ = deterministic_seed
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC"),
            "close": [100.0, 101.0, 99.0, 102.0],
        }
    )
    out, nxt = _collapse_to_uniform_calendar_days(df, start_day_index=5)
    dts = pd.to_datetime(out["date"], utc=True)
    diffs = dts.diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()
    assert nxt == 5 + len(df)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_multi_manifest_rejects_overlapping_segments(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    a = _write_input(tmp_path / "a.parquet", _synthetic_input(periods=120))
    b = _write_input(tmp_path / "b.parquet", _synthetic_input(periods=120))
    manifest = {
        "schema_version": "rg09_multi_fixture_manifest/1",
        "uniform_calendar_day_index": True,
        "segments": [
            {
                "entity_id": "A",
                "input_path": str(a.resolve()),
                "date_start": "2020-01-01",
                "date_end": "2020-06-30",
                "close_scale": 1.0,
            },
            {
                "entity_id": "B",
                "input_path": str(b.resolve()),
                "date_start": "2020-03-01",
                "date_end": "2020-12-31",
                "close_scale": 1.01,
            },
        ],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    repo_root = tmp_path
    with pytest.raises(DataPreconditionError, match="non-overlapping"):
        load_multi_manifest(mp, repo_root=repo_root)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_multi_manifest_independent_instruments_allows_calendar_overlap(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    a = _write_input(tmp_path / "a.parquet", _synthetic_input(periods=120))
    b = _write_input(tmp_path / "b.parquet", _synthetic_input(periods=120))
    manifest = {
        "schema_version": "rg09_multi_fixture_manifest/1",
        "uniform_calendar_day_index": False,
        "calendar_overlap_policy": "independent_instruments",
        "segments": [
            {"entity_id": "A", "input_path": str(a.resolve()), "close_scale": 1.0},
            {"entity_id": "B", "input_path": str(b.resolve()), "close_scale": 1.0},
        ],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    segs, settings = load_multi_manifest(mp, repo_root=tmp_path)
    assert len(segs) == 2
    assert settings.calendar_overlap_policy == "independent_instruments"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_generate_fixture_multi_writes_v2_parquet_and_entity_list(
    tmp_path: Path, deterministic_seed: int
) -> None:
    """Requires a long synthetic slice per segment to satisfy BOCPD cold-start (see v2 config)."""
    _ = deterministic_seed
    long_df = _synthetic_input(periods=900)
    p1 = _write_input(tmp_path / "s1.parquet", long_df.iloc[:500].reset_index(drop=True))
    p2 = _write_input(tmp_path / "s2.parquet", long_df.iloc[500:].reset_index(drop=True))
    cfg = tmp_path / "bocpd_v2.json"
    cfg.write_text(
        json.dumps(
            {
                "hazard_rate": 0.001,
                "observation_model": "student_t",
                "prior_mu0": None,
                "prior_kappa0": 1.0,
                "prior_alpha0": 3.0,
                "prior_beta0": None,
                "max_run_length": 512,
                "vol_window": 60,
                "trend_window": 60,
                "trend_flat_epsilon": 0.02,
                "vol_bucket_method": "tercile",
                "cp_threshold": 0.5,
                "transition_threshold": 0.3,
                "transition_max_rl": 5,
                "cold_start_burn_in": 80,
                "crisis_vol_score_percentile": 90.0,
                "config_version": "rg09_v1.1.0",
            }
        ),
        encoding="utf-8",
    )
    segs = [
        FixtureSegmentSpec(
            entity_id="X", input_path=p1, date_start=None, date_end=None, close_scale=1.0
        ),
        FixtureSegmentSpec(
            entity_id="Y", input_path=p2, date_start=None, date_end=None, close_scale=1.02
        ),
    ]
    result = generate_fixture_multi(
        segments=segs,
        config_path=cfg,
        output_dir=tmp_path / "out_v2",
        multi_settings=MultiManifestSettings(
            uniform_calendar_day_index=True,
            calendar_overlap_policy="disallowed",
            apply_diversification_perturbation=True,
            temporal_folds=2,
        ),
    )
    assert result["fixture_path"].name == FIXTURE_FILENAME_V2
    summary = _read_json(Path(result["summary_path"]))
    metadata = _read_json(Path(result["metadata_path"]))
    assert summary["entity_id"] == ["X", "Y"]
    assert summary["fixture_scope"] == "multi_instrument_governed_basket"
    assert summary["single_series_sufficient"] is False
    assert summary["es_only_sufficient"] is False
    assert metadata["fixture_scope"] == "multi_instrument_governed_basket"
    assert metadata["instrument_ids"] == ["X", "Y"]
    frame = pd.read_parquet(Path(result["fixture_path"]))
    emitted_counts = frame["entity_id"].value_counts(sort=False).to_dict()
    assert metadata["row_count_by_instrument"] == {
        str(key): int(value) for key, value in emitted_counts.items()
    }
    assert set(metadata["date_ranges_by_instrument"]) == {"X", "Y"}
    assert list(frame.columns) == list(FIXTURE_COLUMNS)
    assert frame["rg09_trading_day_ord"].dtype == np.int64 or str(
        frame["rg09_trading_day_ord"].dtype
    ).startswith("int")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_multi_manifest_rejects_uniform_calendar_with_independent_instruments(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    a = _write_input(tmp_path / "a.parquet", _synthetic_input(periods=120))
    b = _write_input(tmp_path / "b.parquet", _synthetic_input(periods=120))
    manifest = {
        "schema_version": "rg09_multi_fixture_manifest/1",
        "uniform_calendar_day_index": True,
        "calendar_overlap_policy": "independent_instruments",
        "segments": [
            {"entity_id": "A", "input_path": str(a.resolve()), "close_scale": 1.0},
            {"entity_id": "B", "input_path": str(b.resolve()), "close_scale": 1.0},
        ],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataPreconditionError, match="uniform_calendar_day_index must be false"):
        load_multi_manifest(mp, repo_root=tmp_path)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_generate_fixture_multi_independent_calendar_emits_fold_construction(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    long_df = _synthetic_input(periods=900)
    p_common = _write_input(tmp_path / "common.parquet", long_df)
    cfg = tmp_path / "bocpd_v2.json"
    cfg.write_text(
        json.dumps(
            {
                "hazard_rate": 0.001,
                "observation_model": "student_t",
                "prior_mu0": None,
                "prior_kappa0": 1.0,
                "prior_alpha0": 3.0,
                "prior_beta0": None,
                "max_run_length": 512,
                "vol_window": 60,
                "trend_window": 60,
                "trend_flat_epsilon": 0.02,
                "vol_bucket_method": "tercile",
                "cp_threshold": 0.5,
                "transition_threshold": 0.3,
                "transition_max_rl": 5,
                "cold_start_burn_in": 80,
                "crisis_vol_score_percentile": 90.0,
                "config_version": "rg09_v1.1.0",
            }
        ),
        encoding="utf-8",
    )
    segs = [
        FixtureSegmentSpec(
            entity_id="X", input_path=p_common, date_start=None, date_end=None, close_scale=1.0
        ),
        FixtureSegmentSpec(
            entity_id="Y", input_path=p_common, date_start=None, date_end=None, close_scale=1.02
        ),
    ]
    result = generate_fixture_multi(
        segments=segs,
        config_path=cfg,
        output_dir=tmp_path / "out_cal",
        multi_settings=MultiManifestSettings(
            uniform_calendar_day_index=False,
            calendar_overlap_policy="independent_instruments",
            apply_diversification_perturbation=False,
            temporal_folds=2,
        ),
    )
    summary = _read_json(Path(result["summary_path"]))
    fc = summary["fold_construction"]
    assert summary["uniform_calendar_day_index"] is False
    assert summary["calendar_overlap_policy"] == "independent_instruments"
    assert fc["method"] == "calendar_time"
    assert fc["uniform_calendar_day_index"] is False
    assert "fold_0" in fc["time_ranges"]
    assert "fold_1" in fc["time_ranges"]
    frame = pd.read_parquet(Path(result["fixture_path"]))
    validate_rg09_fixture_fold_geometry(frame, fc, temporal_folds=2)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_validate_rg09_fixture_fold_geometry_rejects_overlapping_calendar_ranges(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    long_df = _synthetic_input(periods=900)
    p_common = _write_input(tmp_path / "common.parquet", long_df)
    cfg = _write_config(tmp_path / "bocpd_v2.json", _config_payload(crisis_percentile=90.0))
    segs = [
        FixtureSegmentSpec(
            entity_id="X", input_path=p_common, date_start=None, date_end=None, close_scale=1.0
        ),
        FixtureSegmentSpec(
            entity_id="Y", input_path=p_common, date_start=None, date_end=None, close_scale=1.02
        ),
    ]
    result = generate_fixture_multi(
        segments=segs,
        config_path=cfg,
        output_dir=tmp_path / "out_bad_fc",
        multi_settings=MultiManifestSettings(
            uniform_calendar_day_index=False,
            calendar_overlap_policy="independent_instruments",
            apply_diversification_perturbation=False,
            temporal_folds=2,
        ),
    )
    frame = pd.read_parquet(Path(result["fixture_path"]))
    summary = _read_json(Path(result["summary_path"]))
    bad_fc = cast(dict[str, Any], summary["fold_construction"]).copy()
    bad_fc["time_ranges"] = {
        "fold_0": ["2020-01-01", "2020-07-15"],
        "fold_1": ["2020-07-01", "2020-12-31"],
    }
    with pytest.raises(DataPreconditionError, match="do not match recomputation"):
        validate_rg09_fixture_fold_geometry(frame, bad_fc, temporal_folds=2)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_generate_fixture_multi_independent_calendar_fixture_sha_deterministic_across_runs(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    long_df = _synthetic_input(periods=900)
    p_common = _write_input(tmp_path / "common.parquet", long_df)
    cfg = _write_config(tmp_path / "bocpd_v2.json", _config_payload(crisis_percentile=90.0))
    segs = [
        FixtureSegmentSpec(
            entity_id="X", input_path=p_common, date_start=None, date_end=None, close_scale=1.0
        ),
        FixtureSegmentSpec(
            entity_id="Y", input_path=p_common, date_start=None, date_end=None, close_scale=1.02
        ),
    ]
    kwargs: dict[str, Any] = {
        "segments": segs,
        "config_path": cfg,
        "multi_settings": MultiManifestSettings(
            uniform_calendar_day_index=False,
            calendar_overlap_policy="independent_instruments",
            apply_diversification_perturbation=False,
            temporal_folds=2,
        ),
    }
    run_a = generate_fixture_multi(output_dir=tmp_path / "out_a", **kwargs)
    run_b = generate_fixture_multi(output_dir=tmp_path / "out_b", **kwargs)
    summary_a = _read_json(Path(run_a["summary_path"]))
    summary_b = _read_json(Path(run_b["summary_path"]))
    assert summary_a["fixture_sha256"] == summary_b["fixture_sha256"]
    assert summary_a["uniform_calendar_day_index"] is False
    assert summary_b["uniform_calendar_day_index"] is False
    assert (
        summary_a["calendar_overlap_policy"]
        == summary_b["calendar_overlap_policy"]
        == "independent_instruments"
    )
    assert summary_a["fold_construction"] == summary_b["fold_construction"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_multi_fixture_sha_changes_when_uniform_calendar_toggled(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    long_df = _synthetic_input(periods=900)
    p1 = _write_input(tmp_path / "s1.parquet", long_df.iloc[:500].reset_index(drop=True))
    p2 = _write_input(tmp_path / "s2.parquet", long_df.iloc[500:].reset_index(drop=True))
    cfg = tmp_path / "bocpd_v2.json"
    cfg.write_text(
        json.dumps(
            {
                "hazard_rate": 0.001,
                "observation_model": "student_t",
                "prior_mu0": None,
                "prior_kappa0": 1.0,
                "prior_alpha0": 3.0,
                "prior_beta0": None,
                "max_run_length": 512,
                "vol_window": 60,
                "trend_window": 60,
                "trend_flat_epsilon": 0.02,
                "vol_bucket_method": "tercile",
                "cp_threshold": 0.5,
                "transition_threshold": 0.3,
                "transition_max_rl": 5,
                "cold_start_burn_in": 80,
                "crisis_vol_score_percentile": 90.0,
                "config_version": "rg09_v1.1.0",
            }
        ),
        encoding="utf-8",
    )
    segs = [
        FixtureSegmentSpec(
            entity_id="X", input_path=p1, date_start=None, date_end=None, close_scale=1.0
        ),
        FixtureSegmentSpec(
            entity_id="Y", input_path=p2, date_start=None, date_end=None, close_scale=1.02
        ),
    ]
    base_kw: dict[str, Any] = {
        "segments": segs,
        "config_path": cfg,
    }
    r_u = generate_fixture_multi(
        **base_kw,
        output_dir=tmp_path / "out_u",
        multi_settings=MultiManifestSettings(
            uniform_calendar_day_index=True,
            calendar_overlap_policy="disallowed",
            apply_diversification_perturbation=True,
            temporal_folds=2,
        ),
    )
    r_r = generate_fixture_multi(
        **base_kw,
        output_dir=tmp_path / "out_r",
        multi_settings=MultiManifestSettings(
            uniform_calendar_day_index=False,
            calendar_overlap_policy="disallowed",
            apply_diversification_perturbation=True,
            temporal_folds=2,
        ),
    )
    sha_u = _read_json(Path(r_u["summary_path"]))["fixture_sha256"]
    sha_r = _read_json(Path(r_r["summary_path"]))["fixture_sha256"]
    assert sha_u != sha_r


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_calendar_fold_ids_differ_from_naive_index_split(deterministic_seed: int) -> None:
    """Calendar-time fold ids follow date ranges, not sorted-episode index halves (OI-54)."""
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _calendar_fold_ids_for_episode_starts

    starts = pd.Series(
        pd.to_datetime(
            [
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
                "2022-03-01",
                "2022-03-02",
            ],
            utc=True,
        )
    )
    fc = {
        "method": "calendar_time",
        "time_ranges": {
            "fold_0": ["2022-01-01", "2022-02-28"],
            "fold_1": ["2022-03-01", "2022-03-31"],
        },
    }
    cal = _calendar_fold_ids_for_episode_starts(starts, fc)
    naive = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assert list(cal) == [0, 0, 0, 0, 1, 1]
    assert list(cal) != list(naive)

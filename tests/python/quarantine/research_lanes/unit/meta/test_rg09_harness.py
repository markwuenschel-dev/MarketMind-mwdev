"""Unit tests for the RG-09 II-0A harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
from scripts.generate_rg09_fixture import _compute_fixture_sha256

from pysrc.meta.rg09_parquet_io import write_rg09_fixture_parquet
from pysrc.meta.rg09_threshold_catalog import RG09_CONFIG_THRESHOLD_SPECS, threshold_value_record


def _pilot_config_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "config_id": "rg09_pilot_config_test",
        "p_value_threshold": 0.05,
        "null_draw_count": 16,
        "structural_separability_ratio_threshold": 1.0,
        "structural_direction_score_threshold": 0.0,
        "functional_harvey_t_threshold": 3.0,
        "embargo_gap_bars_daily": 0,
        "embargo_gap_fraction_intraday": 0.05,
        "min_support_rows": 2,
        "min_query_rows": 1,
        "label_confidence_threshold": 0.70,
        "min_admissible_episode_count": 2,
        "min_regime_transition_count": 2,
        "min_support_query_mass_per_regime": 1,
        "min_regime_class_count_per_fold": 1,
        "min_temporal_folds": 2,
        "min_dwell_time_bars": 3,
        "low_confidence_boundary_policy": "exclude_v1",
        "functional_model_default": "ridge",
        "functional_model_fallback": "mean_estimator",
        "null_seed_namespace": "rg09.test.v1",
    }
    payload.update(overrides)
    for field_name, spec in RG09_CONFIG_THRESHOLD_SPECS.items():
        if field_name not in payload:
            continue
        raw_value = payload[field_name]
        if isinstance(raw_value, dict):
            continue
        payload[field_name] = threshold_value_record(raw_value, spec.threshold_id)
    return payload


def _direction_score_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime_class": ["crisis", "high_vol", "bull", "bear", "sideways"],
            "adaptation_gain": [1.2, 0.8, -0.8, -1.1, -0.9],
            "query_mean": [1.2, 0.8, -0.8, -1.1, -0.9],
            "query_targets": [[1.2], [0.8], [-0.8], [-1.1], [-0.9]],
            "support_targets": [[0.0], [0.0], [0.0], [0.0], [0.0]],
            "fold_id": [0, 0, 0, 0, 0],
            "query_features": [[[0.0, 0.0, 0.0]]] * 5,
            "support_features": [[[0.0, 0.0, 0.0]]] * 5,
            "target_type": ["continuous"] * 5,
        }
    )


def _episode_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ts = pd.Timestamp("2024-01-02T00:00:00+00:00")
    episode_specs = [
        ("ep0", "bull", 3, 0.2),
        ("ep1", "bear", 3, -0.2),
        ("ep2", "bull", 4, 0.3),
        ("ep3", "bear", 4, -0.3),
    ]
    offset = 0
    for regime_idx, (regime_suffix, regime_class, length, vol_anchor) in enumerate(episode_specs):
        regime_id = f"trend_{regime_suffix}__stable"
        for step in range(length):
            decision_ts = ts + pd.Timedelta(days=offset)
            offset += 1
            rows.append(
                {
                    "entity_id": "ES",
                    "decision_ts": decision_ts,
                    "regime_id": regime_id,
                    "regime_label": regime_id,
                    "effective_at": decision_ts,
                    "state_snapshot_id": f"sha256:state-{regime_idx}-{step}",
                    "input_snapshot_id": "sha256:input",
                    "config_version": "rg09_v1.0.2",
                    "change_probability": 0.05 if step else 0.10,
                    "boundary_flag": "stable",
                    "regime_class": regime_class,
                    "diag_regime_class_bocpd_gated": regime_class,
                    "run_length_mode": float(step + 1),
                    "run_length_expectation": float(step + 1),
                    "transition_probability": 0.10,
                    "posterior_entropy": 0.02,
                    "trend_score_raw": 0.2 if regime_class == "bull" else -0.2,
                    "vol_score_raw": vol_anchor + (step * 0.01),
                }
            )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_fixture_bundle(
    tmp_path: Path,
    *,
    frame: pd.DataFrame | None = None,
    summary_overrides: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Path]:
    fixture_frame = _episode_frame() if frame is None else frame
    fixture_path = tmp_path / "rg09_fixture_v1.parquet"
    write_rg09_fixture_parquet(fixture_frame, fixture_path, index=False)
    fixture_sha = _compute_fixture_sha256(fixture_frame)
    summary = {
        "amendment": "MLN-02-AMD-01",
        "cold_start_rows": 0,
        "config_version": "rg09_v1.0.2",
        "crisis_episodes_after_cold_start": 0,
        "crisis_label_agreement_rate": 1.0,
        "date_range_end": str(fixture_frame["decision_ts"].max().date()),
        "date_range_start": str(fixture_frame["decision_ts"].min().date()),
        "entity_id": "ES",
        "fixture_scope": "single_instrument_governed_fixture",
        "episode_counts_by_regime_id": {
            str(key): int(value)
            for key, value in fixture_frame.groupby("regime_id", sort=True).size().items()
        },
        "es_only_sufficient": True,
        "fixture_sha256": fixture_sha,
        "generation_timestamp": "2026-03-31T00:00:00+00:00",
        "producer_version": "0.1.0",
        "projection_rule": "vol_hi AND severity_flag (p90)",
        "projection_rule_version": "mln02.v1.level2_severity_gate#p=90",
        "projection_rule_logic_id": "mln02.v1.level2_severity_gate",
        "projection_rule_reference_bocpd_id": "mln02.reference.ii0a_bocpd_cp_vol_hi",
        "row_count": int(len(fixture_frame)),
        "row_counts_by_class": {
            str(key): int(value)
            for key, value in fixture_frame.groupby("regime_class", sort=True).size().items()
        },
        "row_counts_by_class_bocpd_gated": {
            str(key): int(value)
            for key, value in fixture_frame.groupby("diag_regime_class_bocpd_gated", sort=True)
            .size()
            .items()
        },
        "single_series_sufficient": True,
        "source_dataset_id": "synthetic_rg09.parquet",
    }
    metadata = {
        "amendment": "MLN-02-AMD-01",
        "config_hash": "hmac-sha256:test-config",
        "fixture_sha256": fixture_sha,
        "generation_timestamp": "2026-03-31T00:00:00+00:00",
        "source_hash": "sha256:synthetic-source",
    }
    if summary_overrides:
        summary.update(summary_overrides)
    if metadata_overrides:
        metadata.update(metadata_overrides)

    summary_path = _write_json(tmp_path / "rg09_fixture_summary.json", summary)
    metadata_path = _write_json(tmp_path / "rg09_fixture_metadata.json", metadata)
    config_payload = _pilot_config_payload(**(config_overrides or {}))
    config_path = _write_json(tmp_path / "rg09_pilot_config_v1.json", config_payload)
    return {
        "fixture": fixture_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "config": config_path,
    }


def _config_with_geometry(**overrides: Any) -> dict[str, Any]:
    payload = _pilot_config_payload(
        min_support_rows=2,
        min_query_rows=2,
        min_dwell_time_bars=4,
        min_admissible_episode_count=1,
        min_regime_transition_count=0,
        min_support_query_mass_per_regime=1,
        min_regime_class_count_per_fold=1,
        min_temporal_folds=1,
        label_horizon_bars=1,
    )
    payload.update(overrides)
    return payload


def _single_episode_frame(length: int = 8) -> pd.DataFrame:
    frame = _episode_frame().iloc[:length].copy().reset_index(drop=True)
    frame.loc[:, "regime_id"] = "trend_single__stable"
    frame.loc[:, "regime_label"] = "trend_single__stable"
    frame.loc[:, "regime_class"] = "bull"
    frame.loc[:, "diag_regime_class_bocpd_gated"] = "bull"
    frame.loc[:, "boundary_flag"] = "stable"
    return frame


def _corrected_v2_frame() -> pd.DataFrame:
    es = _single_episode_frame(length=8).copy()
    nq = _single_episode_frame(length=8).copy()
    es.loc[:, "entity_id"] = "ES"
    nq.loc[:, "entity_id"] = "NQ"
    nq.loc[:, "decision_ts"] = pd.to_datetime(nq["decision_ts"], utc=True) + pd.Timedelta(hours=1)
    nq.loc[:, "effective_at"] = pd.to_datetime(nq["effective_at"], utc=True) + pd.Timedelta(hours=1)
    es.loc[:, "rg09_trading_day_ord"] = list(range(len(es)))
    nq.loc[:, "rg09_trading_day_ord"] = list(range(len(nq)))
    frame = pd.concat([es, nq], ignore_index=True)
    return frame.sort_values(["decision_ts", "entity_id"], kind="mergesort").reset_index(drop=True)


def _corrected_v2_summary_overrides(
    frame: pd.DataFrame,
    *,
    include_fold_construction: bool = True,
    uniform_calendar_day_index: bool = False,
    calendar_overlap_policy: str = "independent_instruments",
) -> dict[str, Any]:
    non_cold = frame.loc[frame["boundary_flag"].astype(str) != "cold_start"].copy()
    start_dates = sorted(
        pd.to_datetime(non_cold["decision_ts"], utc=True)
        .dt.normalize()
        .dt.strftime("%Y-%m-%d")
        .unique()
    )
    fold_construction: dict[str, Any] | None = None
    if include_fold_construction:
        fold_construction = {
            "method": "calendar_time",
            "uniform_calendar_day_index": uniform_calendar_day_index,
            "temporal_folds": 2,
            "time_ranges": {
                "fold_0": [start_dates[0], start_dates[0]],
                "fold_1": [start_dates[1], start_dates[-1]],
            },
        }
    overrides: dict[str, Any] = {
        "entity_id": sorted(str(item) for item in frame["entity_id"].unique().tolist()),
        "fixture_scope": "multi_instrument_governed_basket",
        "uniform_calendar_day_index": uniform_calendar_day_index,
        "calendar_overlap_policy": calendar_overlap_policy,
    }
    if fold_construction is not None:
        overrides["fold_construction"] = fold_construction
    return overrides


def _three_episode_frame() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for idx, regime_class in enumerate(("bull", "bear", "bull")):
        part = _single_episode_frame(length=4).copy()
        shift = pd.Timedelta(days=4 * idx)
        part.loc[:, "decision_ts"] = pd.to_datetime(part["decision_ts"], utc=True) + shift
        part.loc[:, "effective_at"] = pd.to_datetime(part["effective_at"], utc=True) + shift
        part.loc[:, "regime_id"] = f"trend_{idx}__stable"
        part.loc[:, "regime_label"] = f"trend_{idx}__stable"
        part.loc[:, "regime_class"] = regime_class
        part.loc[:, "diag_regime_class_bocpd_gated"] = regime_class
        part.loc[:, "rg09_trading_day_ord"] = list(range(4 * idx, 4 * (idx + 1)))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _dummy_episode_manifest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_count": 4,
                "fold_id": 0,
                "regime_class": "bull",
            }
        ]
    )


def _functional_episode_frame(*, target_type: str = "continuous") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "support_features": [
                    [0.0, 1.0, 0.1],
                    [1.0, 1.5, 0.2],
                    [2.0, 2.0, 0.3],
                    [3.0, 2.5, 0.4],
                ],
                "query_features": [[4.0, 3.0, 0.5], [5.0, 3.5, 0.6]],
                "support_targets": [0.5, 1.0, 1.5, 2.0],
                "query_targets": [2.5, 3.0],
                "target_type": target_type,
            },
            {
                "support_features": [
                    [0.5, 0.8, 0.2],
                    [1.5, 1.3, 0.3],
                    [2.5, 1.8, 0.4],
                    [3.5, 2.3, 0.5],
                ],
                "query_features": [[4.5, 2.8, 0.6], [5.5, 3.3, 0.7]],
                "support_targets": [0.4, 0.9, 1.4, 1.9],
                "query_targets": [2.4, 2.9],
                "target_type": target_type,
            },
        ]
    )


def _episode_mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _synthetic_null_episode_row(
    *,
    regime_id: str,
    regime_class: str,
    fold_id: int,
    support_targets: list[float],
    query_targets: list[float],
    stale_derived: bool = False,
) -> dict[str, Any]:
    support_vals = [float(value) for value in support_targets]
    query_vals = [float(value) for value in query_targets]
    support_mean = -999.0 if stale_derived else _episode_mean(support_vals)
    query_mean = -888.0 if stale_derived else _episode_mean(query_vals)
    adaptation_gain = -777.0 if stale_derived else query_mean - support_mean
    return {
        "support_features": [
            [float(idx), float(idx) + 0.25, float(target)]
            for idx, target in enumerate(support_vals)
        ],
        "query_features": [
            [float(idx) + 10.0, float(idx) + 10.25, float(target)]
            for idx, target in enumerate(query_vals)
        ],
        "support_targets": support_vals,
        "query_targets": query_vals,
        "target_type": "continuous",
        "regime_class": regime_class,
        "regime_label": regime_id,
        "regime_id": regime_id,
        "fold_id": fold_id,
        "row_count": len(support_vals) + len(query_vals),
        "support_count": len(support_vals),
        "query_count": len(query_vals),
        "support_mean": support_mean,
        "query_mean": query_mean,
        "adaptation_gain": adaptation_gain,
    }


def _synthetic_null_episodes(*, stale_derived: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _synthetic_null_episode_row(
                regime_id="trend_a",
                regime_class="bull",
                fold_id=0,
                support_targets=[0.2, 0.4, 0.6, 0.8],
                query_targets=[0.9, 1.4],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_b",
                regime_class="bear",
                fold_id=0,
                support_targets=[1.0, 1.2, 1.4, 1.6],
                query_targets=[1.8, 2.5],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_c",
                regime_class="bull",
                fold_id=0,
                support_targets=[0.3, 0.5, 0.7, 0.9],
                query_targets=[1.1, 1.7],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_d",
                regime_class="bear",
                fold_id=0,
                support_targets=[1.1, 1.3, 1.5, 1.7],
                query_targets=[2.0, 2.8],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_e",
                regime_class="bull",
                fold_id=1,
                support_targets=[0.1, 0.2, 0.4, 0.5],
                query_targets=[0.6, 1.2],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_f",
                regime_class="bear",
                fold_id=1,
                support_targets=[1.2, 1.4, 1.6, 1.8],
                query_targets=[2.3, 2.7],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_g",
                regime_class="bull",
                fold_id=1,
                support_targets=[0.4, 0.6, 0.8, 1.0],
                query_targets=[1.0, 1.6],
                stale_derived=stale_derived,
            ),
            _synthetic_null_episode_row(
                regime_id="trend_h",
                regime_class="bear",
                fold_id=1,
                support_targets=[1.3, 1.5, 1.7, 1.9],
                query_targets=[2.4, 3.0],
                stale_derived=stale_derived,
            ),
        ]
    )


def _assert_episode_statistics_match_targets(frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        support_mean = _episode_mean(list(row["support_targets"]))
        query_mean = _episode_mean(list(row["query_targets"]))
        assert float(row["support_mean"]) == pytest.approx(support_mean)
        assert float(row["query_mean"]) == pytest.approx(query_mean)
        assert float(row["adaptation_gain"]) == pytest.approx(query_mean - support_mean)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_h1_path_is_default(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(tmp_path)
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["hypothesis_id"] == "RG09-H1"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_successor_hypotheses_not_executed_on_base_run(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(tmp_path)
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["executed_hypotheses"] == ["RG09-H1"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fixture_with_regime_label_equal_to_regime_id_is_admissible(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _load_and_validate_fixture

    frame = _single_episode_frame()
    frame.loc[:, "regime_label"] = frame["regime_id"]
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(),
    )
    _, fail_codes = _load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )
    assert fail_codes == []


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_only_one_successor_hypothesis_may_be_requested_per_cycle(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import validate_successor_request

    with pytest.raises(ValueError, match="exactly one"):
        validate_successor_request(["RG09-H2", "RG09-H3"])


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_load_rg09_config_min_episode_regime_class_purity_optional_and_bounded(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.core.errors import ConfigValidationError
    from pysrc.meta.rg09_harness import load_rg09_config

    paths_ok = _write_fixture_bundle(tmp_path / "ok", frame=_episode_frame())
    cfg0 = load_rg09_config(paths_ok["config"])
    assert cfg0.min_episode_regime_class_purity == pytest.approx(0.0)

    bad_path = tmp_path / "bad.json"
    bad_payload = _pilot_config_payload(min_episode_regime_class_purity=1.01)
    _write_json(bad_path, bad_payload)
    with pytest.raises(ConfigValidationError, match="min_episode_regime_class_purity"):
        load_rg09_config(bad_path)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_successor_authorization_blocked_on_fixture_invalidity(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    frame = _episode_frame()
    frame.loc[0, "state_snapshot_id"] = None
    paths = _write_fixture_bundle(tmp_path, frame=frame)
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] is None
    assert gate_result["gate_executed"] is False
    assert gate_result["successor_hypotheses"]["eligible"] is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_successor_authorization_allowed_on_needs_more_evidence_result(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )

    def _fake_evaluation(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
            "leakage_geometry": {"clean": True},
            "non_exchangeability": {
                "statistical_pass": True,
                "structural_pass": True,
                "functional_pass": False,
                "structural_contamination": False,
                "structural_separability_ratio": 1.5,
            },
            "null_collapse": {},
        }
        diagnostics: dict[str, Any] = {"authorized_null_families": []}
        return evidence, diagnostics

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(
        harness, "_derive_episodes", lambda *_args, **_kwargs: (_dummy_episode_manifest_frame(), {})
    )
    monkeypatch.setattr(harness, "_evaluate_base_hypothesis", _fake_evaluation)
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])
    result = harness.run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] == "NEEDS_MORE_EVIDENCE"
    assert gate_result["gate_executed"] is True
    assert gate_result["successor_hypotheses"]["eligible"] is True
    assert gate_result["successor_hypotheses"]["reason"] == "targeted_follow_up_permitted"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_rg09_gate_result_deterministic(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(tmp_path)
    result_a = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out_a",
    )
    result_b = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out_b",
    )
    assert _read_json(result_a.output_dir / "rg09_gate_result.json") == _read_json(
        result_b.output_dir / "rg09_gate_result.json"
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fixture_sha_mismatch_disables_gate_execution(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(
        tmp_path,
        summary_overrides={"fixture_sha256": "sha256:deadbeef"},
    )
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] is None
    assert gate_result["gate_executed"] is False
    assert "FAIL_CANONICAL_LABEL_DEPENDENCY_UNMET" in gate_result["fail_codes"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_state_snapshot_id_required(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    frame = _episode_frame()
    frame["state_snapshot_id"] = None
    paths = _write_fixture_bundle(tmp_path, frame=frame)
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] is None
    assert gate_result["gate_executed"] is False
    assert "FAIL_CANONICAL_LABEL_DEPENDENCY_UNMET" in gate_result["fail_codes"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_corrected_v2_fixture_missing_trading_day_ord_fails_geometry_contract(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE, run_rg09_harness

    frame = _corrected_v2_frame().drop(columns=["rg09_trading_day_ord"])
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides=_corrected_v2_summary_overrides(frame),
        config_overrides=_config_with_geometry(min_temporal_folds=2),
    )
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out_geometry_missing_ord",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    diagnostics = _read_json(result.output_dir / "rg09_diagnostics.json")
    assert gate_result["decision"] is None
    assert gate_result["gate_executed"] is False
    assert FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE in gate_result["fail_codes"]
    assert "rg09_trading_day_ord" in gate_result["decision_reason"]
    assert "rg09_trading_day_ord" in " ".join(diagnostics["fixture_geometry_contract"]["breaches"])


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_corrected_v2_fixture_missing_fold_construction_fails_geometry_contract(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import (
        FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE,
        _load_and_validate_fixture,
    )

    frame = _corrected_v2_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides=_corrected_v2_summary_overrides(frame, include_fold_construction=False),
        config_overrides=_config_with_geometry(min_temporal_folds=2),
    )
    bundle, fail_codes = _load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        expected_temporal_folds=2,
    )
    assert FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE in fail_codes
    assert any("fold_construction" in breach for breach in bundle.geometry_contract_breaches)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_corrected_v2_fixture_malformed_fold_construction_fails_geometry_contract(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import (
        FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE,
        _load_and_validate_fixture,
    )

    frame = _corrected_v2_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides=_corrected_v2_summary_overrides(
            frame,
            include_fold_construction=True,
        ),
        config_overrides=_config_with_geometry(min_temporal_folds=2),
    )
    summary = _read_json(paths["summary"])
    summary["fold_construction"] = {
        "method": "calendar_time",
        "uniform_calendar_day_index": False,
        "temporal_folds": 2,
        "time_ranges": {
            "fold_0": ["2024-01-05", "2024-01-01"],
            "fold_1": ["2024-01-01", "2024-01-08"],
        },
    }
    _write_json(paths["summary"], summary)
    bundle, fail_codes = _load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        expected_temporal_folds=2,
    )
    assert FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE in fail_codes
    assert any("time_ranges" in breach for breach in bundle.geometry_contract_breaches)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_corrected_v2_fixture_policy_contradiction_fails_geometry_contract(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import (
        FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE,
        _load_and_validate_fixture,
    )

    frame = _corrected_v2_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides=_corrected_v2_summary_overrides(
            frame,
            uniform_calendar_day_index=True,
        ),
        config_overrides=_config_with_geometry(min_temporal_folds=2),
    )
    bundle, fail_codes = _load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        expected_temporal_folds=2,
    )
    assert FIXTURE_GEOMETRY_CONTRACT_FAIL_CODE in fail_codes
    assert any(
        "uniform_calendar_day_index" in breach for breach in bundle.geometry_contract_breaches
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_rejected_for_label_not_yet_effective(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _single_episode_frame()
    frame.loc[0, "effective_at"] = frame.loc[0, "decision_ts"] + pd.Timedelta(days=1)
    paths = _write_fixture_bundle(tmp_path, frame=frame, config_overrides=_config_with_geometry())
    config = load_rg09_config(paths["config"])
    episodes, exclusions = _derive_episodes(frame, config)
    assert episodes.empty
    assert exclusions["LABEL_NOT_YET_EFFECTIVE"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_rejected_for_noncontiguous_span(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _single_episode_frame()
    frame.loc[4:, "decision_ts"] = frame.loc[4:, "decision_ts"] + pd.Timedelta(days=2)
    paths = _write_fixture_bundle(tmp_path, frame=frame, config_overrides=_config_with_geometry())
    config = load_rg09_config(paths["config"])
    episodes, exclusions = _derive_episodes(frame, config)
    assert episodes.empty
    assert exclusions["NONCONTIGUOUS"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_rejected_for_horizon_overlap(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _single_episode_frame(length=5)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            label_horizon_bars=3, min_support_rows=1, min_query_rows=1
        ),
    )
    config = load_rg09_config(paths["config"])
    episodes, exclusions = _derive_episodes(frame, config)
    assert episodes.empty
    assert exclusions["HORIZON_OVERLAP"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_rejected_for_insufficient_support(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _single_episode_frame(length=6)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=3, min_query_rows=1, label_horizon_bars=1
        ),
    )
    config = load_rg09_config(paths["config"])
    episodes, exclusions = _derive_episodes(frame, config)
    assert episodes.empty
    assert exclusions["INSUFFICIENT_SUPPORT"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_rejected_for_insufficient_query(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _single_episode_frame(length=6)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=1, min_query_rows=4, label_horizon_bars=1
        ),
    )
    config = load_rg09_config(paths["config"])
    episodes, exclusions = _derive_episodes(frame, config)
    assert episodes.empty
    assert exclusions["INSUFFICIENT_QUERY"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_corrected_v2_derive_episodes_uses_calendar_fold_geometry(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _three_episode_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_temporal_folds=2,
            min_support_rows=1,
            min_query_rows=1,
            label_horizon_bars=1,
        ),
    )
    config = load_rg09_config(paths["config"])
    fold_construction = {
        "method": "calendar_time",
        "uniform_calendar_day_index": False,
        "temporal_folds": 2,
        "time_ranges": {
            "fold_0": ["2024-01-02", "2024-01-02"],
            "fold_1": ["2024-01-03", "2024-01-14"],
        },
    }
    episodes, exclusions = _derive_episodes(
        frame,
        config,
        fold_construction=fold_construction,
        require_strict_geometry=True,
    )
    assert exclusions["NONCONTIGUOUS"] == 0
    assert episodes["fold_id"].tolist() == [0, 1, 1]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_legacy_derive_episodes_retains_index_split_fallback(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    frame = _three_episode_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_temporal_folds=2,
            min_support_rows=1,
            min_query_rows=1,
            label_horizon_bars=1,
        ),
    )
    config = load_rg09_config(paths["config"])
    episodes, exclusions = _derive_episodes(frame, config)
    assert exclusions["NONCONTIGUOUS"] == 0
    assert episodes["fold_id"].tolist() == [0, 0, 1]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_clean_negative_result_uses_fail_kill_vocabulary(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )

    def _fake_evaluation(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
            "leakage_geometry": {"clean": True},
            "non_exchangeability": {
                "statistical_pass": False,
                "structural_pass": False,
                "functional_pass": False,
                "structural_contamination": False,
                "structural_separability_ratio": 0.0,
            },
            "null_collapse": {},
        }
        diagnostics: dict[str, Any] = {"authorized_null_families": []}
        return evidence, diagnostics

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(
        harness, "_derive_episodes", lambda *_args, **_kwargs: (_dummy_episode_manifest_frame(), {})
    )
    monkeypatch.setattr(harness, "_evaluate_base_hypothesis", _fake_evaluation)
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])
    result = harness.run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] == "FAIL_KILL"
    assert gate_result["gate_executed"] is True
    assert gate_result["successor_hypotheses"]["eligible"] is False
    assert "FAIL_EXCHANGEABLE_TASKS" in gate_result["fail_codes"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_directional_underpowered_result_uses_needs_more_evidence_vocabulary(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )

    def _directional_eval(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
            "leakage_geometry": {"clean": True, "reproducibility_consistent": True},
            "non_exchangeability": {
                "statistical_pass": False,
                "structural_pass": True,
                "functional_pass": False,
                "structural_contamination": False,
                "structural_separability_ratio": 1.84,
                "fold_evidence": [
                    {
                        "fold_id": 0,
                        "null_families": {
                            "shuffled_label": {"real_statistic": 0.93, "null_mean": 0.32},
                            "shuffled_regime": {"real_statistic": 0.93, "null_mean": 0.06},
                            "matched_exchangeable_window": {
                                "real_statistic": 0.93,
                                "null_mean": 0.07,
                            },
                        },
                    },
                    {
                        "fold_id": 1,
                        "null_families": {
                            "shuffled_label": {"real_statistic": 1.02, "null_mean": 0.75},
                            "shuffled_regime": {"real_statistic": 1.02, "null_mean": 0.41},
                            "matched_exchangeable_window": {
                                "real_statistic": 1.02,
                                "null_mean": 0.43,
                            },
                        },
                    },
                ],
            },
            "null_collapse": {
                "null_distribution_valid": True,
                "invalid_families": [],
                "functional_evidence": {
                    "admissible": True,
                    "positive_delta": True,
                    "harvey_t": 2.02,
                    "mean_delta": 1.04,
                },
            },
        }
        diagnostics: dict[str, Any] = {"authorized_null_families": []}
        return evidence, diagnostics

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(
        harness, "_derive_episodes", lambda *_args, **_kwargs: (_dummy_episode_manifest_frame(), {})
    )
    monkeypatch.setattr(harness, "_evaluate_base_hypothesis", _directional_eval)
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])
    result = harness.run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] == "NEEDS_MORE_EVIDENCE"
    assert gate_result["decision_reason"] == (
        "Base harness is structurally valid and directionally consistent, but statistical evidence is below threshold."
    )
    assert gate_result["fail_codes"] == []
    assert gate_result["successor_hypotheses"]["eligible"] is True
    assert gate_result["successor_hypotheses"]["reason"] == "targeted_follow_up_permitted"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_null_generators_use_only_authorized_families(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_nulls import AUTHORIZED_NULL_FAMILIES

    assert AUTHORIZED_NULL_FAMILIES == (
        "shuffled_label",
        "shuffled_regime",
        "matched_exchangeable_window",
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_null_families_recompute_derived_statistics_from_payloads(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_nulls import (
        matched_exchangeable_window_null,
        shuffled_label_null,
        shuffled_regime_null,
    )

    episodes = _synthetic_null_episodes(stale_derived=True)

    shuffled_regime = shuffled_regime_null(
        episodes,
        namespace="rg09.test",
        fixture_sha256="sha256:test",
        draw_index=1,
    )
    shuffled_label = shuffled_label_null(
        episodes,
        namespace="rg09.test",
        fixture_sha256="sha256:test",
        draw_index=1,
    )
    exchangeable = matched_exchangeable_window_null(
        episodes,
        namespace="rg09.test",
        fixture_sha256="sha256:test",
        draw_index=1,
    )

    _assert_episode_statistics_match_targets(shuffled_regime)
    _assert_episode_statistics_match_targets(shuffled_label)
    _assert_episode_statistics_match_targets(exchangeable)

    assert shuffled_regime["regime_id"].tolist() != episodes["regime_id"].tolist()
    assert shuffled_label["query_targets"].tolist() != episodes["query_targets"].tolist()
    assert exchangeable["support_targets"].tolist() != episodes["support_targets"].tolist()
    assert shuffled_regime["support_mean"].tolist() != episodes["support_mean"].tolist()
    assert shuffled_label["query_mean"].tolist() != episodes["query_mean"].tolist()
    assert exchangeable["adaptation_gain"].tolist() != episodes["adaptation_gain"].tolist()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_shuffled_regime_family_uses_nonconstant_separability_statistic(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    episodes = _synthetic_null_episodes()
    draw_stats: list[float] = []
    for draw_index in range(8):
        null_frame = harness._null_draw(
            "shuffled_regime",
            episodes,
            namespace="rg09.test",
            fixture_sha256="sha256:test",
            draw_index=draw_index,
        )
        draw_stats.append(harness._statistic_for_family("shuffled_regime", null_frame))

    assert len({round(value, 12) for value in draw_stats}) > 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_shuffled_label_family_uses_nonconstant_separability_statistic(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    episodes = _synthetic_null_episodes()
    draw_stats: list[float] = []
    for draw_index in range(8):
        null_frame = harness._null_draw(
            "shuffled_label",
            episodes,
            namespace="rg09.test",
            fixture_sha256="sha256:test",
            draw_index=draw_index,
        )
        draw_stats.append(harness._statistic_for_family("shuffled_label", null_frame))

    assert len({round(value, 12) for value in draw_stats}) > 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_matched_exchangeable_window_family_uses_nonconstant_separability_statistic(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    episodes = _synthetic_null_episodes()
    draw_stats: list[float] = []
    for draw_index in range(8):
        null_frame = harness._null_draw(
            "matched_exchangeable_window",
            episodes,
            namespace="rg09.test",
            fixture_sha256="sha256:test",
            draw_index=draw_index,
        )
        draw_stats.append(harness._statistic_for_family("matched_exchangeable_window", null_frame))

    assert len({round(value, 12) for value in draw_stats}) > 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_statistic_for_family_uses_uniform_regime_statistic(deterministic_seed: int) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    episodes = _synthetic_null_episodes()
    regime_stat = harness._regime_separability_statistic(episodes)

    assert harness._statistic_for_family("shuffled_regime", episodes) == pytest.approx(regime_stat)
    assert harness._statistic_for_family("shuffled_label", episodes) == pytest.approx(regime_stat)
    assert harness._statistic_for_family("matched_exchangeable_window", episodes) == pytest.approx(
        regime_stat
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_degenerate_null_distributions_emit_invalid_evidence_not_fail_kill(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )
    episodes = _synthetic_null_episodes()

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(harness, "_derive_episodes", lambda *_args, **_kwargs: (episodes, {}))
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])

    original_null_draw = harness._null_draw

    def _degenerate_label_family(
        family: str,
        episode_frame: pd.DataFrame,
        *,
        namespace: str,
        fixture_sha256: str,
        draw_index: int,
    ) -> pd.DataFrame:
        if family == "shuffled_label":
            _ = (namespace, fixture_sha256, draw_index)
            return episode_frame.copy(deep=True)
        return original_null_draw(
            family,
            episode_frame,
            namespace=namespace,
            fixture_sha256=fixture_sha256,
            draw_index=draw_index,
        )

    monkeypatch.setattr(harness, "_null_draw", _degenerate_label_family)

    result = harness.run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )

    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    diagnostics = _read_json(result.output_dir / "rg09_diagnostics.json")
    fold0 = diagnostics["null_distribution_summaries"]["fold_0"]["shuffled_label"]
    fold0_regime = diagnostics["null_distribution_summaries"]["fold_0"]["shuffled_regime"]

    assert gate_result["gate_executed"] is True
    assert gate_result["decision"] is None
    assert (
        gate_result["decision_reason"]
        == "Null distribution invalidated the statistical evidence surface."
    )
    assert gate_result["fail_codes"] == ["FAIL_NULL_DISTRIBUTION_INVALID"]
    assert gate_result["successor_hypotheses"]["eligible"] is False
    assert gate_result["successor_hypotheses"]["reason"] == "invalid_evidence_surface"
    assert fold0["distinct_draw_count"] == 1
    assert fold0["null_range"] == pytest.approx(0.0, abs=1e-12)
    assert fold0_regime["distinct_draw_count"] > 1
    assert fold0_regime["null_range"] > 0.0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_thresholds_loaded_from_config_not_hardcoded(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path,
        config_overrides={
            "min_admissible_episode_count": 7,
            "label_horizon_bars": 7,
            "null_draw_count": 23,
        },
    )
    config = load_rg09_config(paths["config"])
    assert config.min_admissible_episode_count == 7
    assert config.label_horizon_bars == 7
    assert config.null_draw_count == 23
    assert config.threshold_ids["min_admissible_episode_count"] == "THR-RG09-V09"
    assert config.threshold_ids["null_draw_count"] == "THR-RG09-V16"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_rg09_config_gate_critical_numeric_without_id_fails(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import load_rg09_config
    from pysrc.meta.threshold_governance import ThresholdGovernanceError

    payload = _pilot_config_payload(
        p_value_threshold={"threshold_id": "THR-RG09-V01", "value": 0.05}
    )
    payload["p_value_threshold"] = 0.05
    config_path = _write_json(tmp_path / "rg09_bad_gate_threshold.json", payload)

    with pytest.raises(ThresholdGovernanceError):
        load_rg09_config(config_path)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_rg09_config_non_gate_numeric_without_id_warns(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    payload = _pilot_config_payload()
    payload["embargo_gap_bars_daily"] = 0
    config_path = _write_json(tmp_path / "rg09_warn_non_gate_threshold.json", payload)

    warnings_seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        harness,
        "warn_hardcoded_threshold",
        lambda **kwargs: warnings_seen.append(dict(kwargs)),
    )
    config = harness.load_rg09_config(config_path)

    assert config.embargo_gap_bars_daily == 0
    assert any(item["consumer"] == "load_rg09_config" for item in warnings_seen)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_rg09_config_defaults_structural_direction_score_threshold(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import load_rg09_config

    payload = _pilot_config_payload()
    del payload["structural_direction_score_threshold"]
    config_path = _write_json(tmp_path / "rg09_transition_config.json", payload)

    config = load_rg09_config(config_path)

    assert config.structural_direction_score_threshold == 0.0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_structural_direction_score_positive_on_correct_surface(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _structural_direction_score

    score = _structural_direction_score(_direction_score_episodes())

    assert score > 0.0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_structural_direction_score_returns_zero_when_group_missing(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _structural_direction_score

    episodes = (
        _direction_score_episodes()
        .loc[lambda df: df["regime_class"].eq("crisis")]
        .reset_index(drop=True)
    )

    assert _structural_direction_score(episodes) == 0.0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_evaluate_fold_uses_configured_null_draw_count(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _evaluate_fold, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path,
        config_overrides=_config_with_geometry(min_temporal_folds=2, null_draw_count=11),
    )
    config = load_rg09_config(paths["config"])
    episodes = _synthetic_null_episodes()

    fold_summary = _evaluate_fold(
        episodes.loc[episodes["fold_id"] == 0].reset_index(drop=True),
        config=config,
        fixture_sha256="sha256:test-fixture",
        fold_id=0,
    )

    for family_summary in fold_summary["null_families"].values():
        summary = family_summary
        assert summary["draw_count"] == 11


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_evaluate_fold_uses_direction_score_for_transition_anchored(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(
        tmp_path,
        config_overrides={
            "episode_construction": "transition_anchored",
            "functional_harvey_t_threshold": -1.0,
        },
    )
    config = harness.load_rg09_config(paths["config"])
    episodes = _direction_score_episodes()

    monkeypatch.setattr(
        harness,
        "_functional_evidence",
        lambda _episodes, _config: {
            "model_used": "ridge",
            "fallback_used": False,
            "positive_delta": True,
            "mean_delta": 0.1,
            "harvey_t": 1.0,
            "admissible": True,
        },
    )

    fold_summary = harness._evaluate_fold(
        episodes,
        config=config,
        fixture_sha256="sha256:test-fixture",
        fold_id=0,
    )

    assert fold_summary["structural_measure"] == "direction_score"
    assert fold_summary["structural_separability_ratio"] == pytest.approx(1.9333333333)
    assert fold_summary["structural_pass"] is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_evaluate_fold_uses_ratio_for_stable_span(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(
        tmp_path,
        config_overrides={
            "episode_construction": "stable_span",
            "functional_harvey_t_threshold": -1.0,
        },
    )
    config = harness.load_rg09_config(paths["config"])
    episodes = _direction_score_episodes()

    monkeypatch.setattr(
        harness,
        "_functional_evidence",
        lambda _episodes, _config: {
            "model_used": "ridge",
            "fallback_used": False,
            "positive_delta": True,
            "mean_delta": 0.1,
            "harvey_t": 1.0,
            "admissible": True,
        },
    )

    fold_summary = harness._evaluate_fold(
        episodes,
        config=config,
        fixture_sha256="sha256:test-fixture",
        fold_id=0,
    )

    assert fold_summary["structural_measure"] == "separability_ratio"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_functional_evidence_uses_ridge_when_supported(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _functional_evidence, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path, config_overrides=_config_with_geometry(min_temporal_folds=1)
    )
    config = load_rg09_config(paths["config"])
    episodes = _functional_episode_frame()
    evidence = _functional_evidence(episodes, config)
    assert evidence["model_used"] == "ridge"
    assert evidence["fallback_used"] is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_functional_evidence_uses_fallback_only_when_necessary(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _functional_evidence, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path, config_overrides=_config_with_geometry(min_temporal_folds=1)
    )
    config = load_rg09_config(paths["config"])
    episodes = _functional_episode_frame(target_type="unsupported")
    evidence = _functional_evidence(episodes, config)
    assert evidence["admissible"] is False
    assert evidence["admissibility_reason"] == "non_continuous_targets_inadmissible_v1"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_functional_evidence_rejects_unauthorized_model_type(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _functional_evidence, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path,
        config_overrides=_config_with_geometry(
            functional_model_default="svm",
            min_temporal_folds=1,
        ),
    )
    config = load_rg09_config(paths["config"])
    episodes = _functional_episode_frame()
    with pytest.raises(ValueError, match="unauthorized functional model type"):
        _functional_evidence(episodes, config)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_non_continuous_targets_fail_closed_in_decision(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )

    def _inadmissible_eval(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
            "leakage_geometry": {"clean": True, "reproducibility_consistent": True},
            "non_exchangeability": {
                "statistical_pass": True,
                "structural_pass": True,
                "functional_pass": False,
                "structural_contamination": False,
                "structural_separability_ratio": 1.5,
            },
            "null_collapse": {
                "functional_evidence": {
                    "admissible": False,
                    "admissibility_reason": "non_continuous_targets_inadmissible_v1",
                }
            },
        }
        return evidence, {}

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(
        harness, "_derive_episodes", lambda *_args, **_kwargs: (_dummy_episode_manifest_frame(), {})
    )
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "_evaluate_base_hypothesis", _inadmissible_eval)
    result = harness.run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] == "FAIL_KILL"
    assert (
        gate_result["decision_reason"]
        == "Functional evidence target type is inadmissible for the v1 harness."
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_per_lane_fold_inconsistent_returns_true_for_mixed_statistical_lane(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _per_lane_fold_inconsistent

    fold_summaries = [
        {"statistical_pass": False, "structural_pass": True, "functional_pass": False},
        {"statistical_pass": True, "structural_pass": True, "functional_pass": False},
    ]

    assert _per_lane_fold_inconsistent(fold_summaries) is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_per_lane_fold_inconsistent_returns_false_when_all_lanes_agree(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _per_lane_fold_inconsistent

    fold_summaries = [
        {"statistical_pass": False, "structural_pass": True, "functional_pass": False},
        {"statistical_pass": False, "structural_pass": True, "functional_pass": False},
    ]

    assert _per_lane_fold_inconsistent(fold_summaries) is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fold_inconsistent_lane_results_route_to_needs_more_evidence(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _synthesize_decision, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path, config_overrides=_config_with_geometry(min_temporal_folds=2)
    )
    config = load_rg09_config(paths["config"])
    evidence = {
        "leakage_geometry": {"clean": True, "reproducibility_consistent": True},
        "non_exchangeability": {
            "statistical_pass": False,
            "structural_pass": False,
            "functional_pass": False,
            "structural_separability_ratio": 0.95,
            "structural_contamination": False,
            "fold_evidence": [
                {
                    "fold_id": 0,
                    "statistical_pass": False,
                    "structural_pass": True,
                    "functional_pass": False,
                    "null_families": {
                        "shuffled_label": {"real_statistic": 0.7, "null_mean": 0.8},
                    },
                },
                {
                    "fold_id": 1,
                    "statistical_pass": True,
                    "structural_pass": False,
                    "functional_pass": True,
                    "null_families": {
                        "shuffled_label": {"real_statistic": 0.9, "null_mean": 0.8},
                    },
                },
            ],
        },
        "null_collapse": {
            "null_distribution_valid": True,
            "functional_evidence": {
                "admissible": True,
                "positive_delta": True,
                "harvey_t": 2.5,
                "mean_delta": 0.5,
            },
        },
    }

    decision, _, fail_codes, _ = _synthesize_decision(evidence, config)

    assert decision == "NEEDS_MORE_EVIDENCE"
    assert fail_codes == ["FAIL_NONREPRODUCIBLE"]
    assert decision != "FAIL_KILL"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fold_disagreement_prevents_pass(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _synthesize_decision, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path, config_overrides=_config_with_geometry(min_temporal_folds=2)
    )
    config = load_rg09_config(paths["config"])
    evidence = {
        "leakage_geometry": {"clean": True, "reproducibility_consistent": False},
        "non_exchangeability": {
            "statistical_pass": True,
            "structural_pass": True,
            "functional_pass": True,
            "structural_separability_ratio": 2.0,
            "structural_contamination": False,
        },
        "null_collapse": {},
    }
    decision, _, fail_codes, _ = _synthesize_decision(evidence, config)
    assert decision == "NEEDS_MORE_EVIDENCE"
    assert fail_codes == ["FAIL_NONREPRODUCIBLE"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_directional_underpowered_statistics_map_to_needs_more_evidence(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _synthesize_decision, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path, config_overrides=_config_with_geometry(min_temporal_folds=2)
    )
    config = load_rg09_config(paths["config"])
    evidence = {
        "leakage_geometry": {"clean": True, "reproducibility_consistent": True},
        "non_exchangeability": {
            "statistical_pass": False,
            "structural_pass": True,
            "functional_pass": False,
            "structural_separability_ratio": 1.835,
            "structural_contamination": False,
            "fold_evidence": [
                {
                    "fold_id": 0,
                    "null_families": {
                        "shuffled_label": {"real_statistic": 0.9328, "null_mean": 0.3208},
                        "shuffled_regime": {"real_statistic": 0.9328, "null_mean": 0.0647},
                        "matched_exchangeable_window": {
                            "real_statistic": 0.9328,
                            "null_mean": 0.0692,
                        },
                    },
                },
                {
                    "fold_id": 1,
                    "null_families": {
                        "shuffled_label": {"real_statistic": 1.0171, "null_mean": 0.7491},
                        "shuffled_regime": {"real_statistic": 1.0171, "null_mean": 0.4145},
                        "matched_exchangeable_window": {
                            "real_statistic": 1.0171,
                            "null_mean": 0.4257,
                        },
                    },
                },
            ],
        },
        "null_collapse": {
            "null_distribution_valid": True,
            "functional_evidence": {
                "admissible": True,
                "positive_delta": True,
                "harvey_t": 2.022,
                "mean_delta": 1.045,
            },
        },
    }
    decision, reason, fail_codes, clean_structure = _synthesize_decision(evidence, config)
    assert decision == "NEEDS_MORE_EVIDENCE"
    assert (
        reason
        == "Base harness is structurally valid and directionally consistent, but statistical evidence is below threshold."
    )
    assert fail_codes == []
    assert clean_structure is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_mixed_statistical_lane_with_directional_consistency_routes_fail_nonreproducible(
    tmp_path: Path, deterministic_seed: int
) -> None:
    """Contradictory fold-level statistical lane must not use the directional-underpowered path."""
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import _synthesize_decision, load_rg09_config

    paths = _write_fixture_bundle(
        tmp_path, config_overrides=_config_with_geometry(min_temporal_folds=2)
    )
    config = load_rg09_config(paths["config"])
    evidence = {
        "leakage_geometry": {"clean": True, "reproducibility_consistent": True},
        "non_exchangeability": {
            "statistical_pass": False,
            "structural_pass": True,
            "functional_pass": True,
            "structural_separability_ratio": 1.835,
            "structural_contamination": False,
            "fold_evidence": [
                {
                    "fold_id": 0,
                    "statistical_pass": False,
                    "structural_pass": True,
                    "functional_pass": True,
                    "null_families": {
                        "shuffled_label": {"real_statistic": 0.9328, "null_mean": 0.3208},
                        "shuffled_regime": {"real_statistic": 0.9328, "null_mean": 0.0647},
                        "matched_exchangeable_window": {
                            "real_statistic": 0.9328,
                            "null_mean": 0.0692,
                        },
                    },
                },
                {
                    "fold_id": 1,
                    "statistical_pass": True,
                    "structural_pass": True,
                    "functional_pass": True,
                    "null_families": {
                        "shuffled_label": {"real_statistic": 1.0171, "null_mean": 0.7491},
                        "shuffled_regime": {"real_statistic": 1.0171, "null_mean": 0.4145},
                        "matched_exchangeable_window": {
                            "real_statistic": 1.0171,
                            "null_mean": 0.4257,
                        },
                    },
                },
            ],
        },
        "null_collapse": {
            "null_distribution_valid": True,
            "functional_evidence": {
                "admissible": True,
                "positive_delta": True,
                "harvey_t": 2.022,
                "mean_delta": 1.045,
            },
        },
    }
    decision, reason, fail_codes, _ = _synthesize_decision(evidence, config)
    assert decision == "NEEDS_MORE_EVIDENCE"
    assert "FAIL_NONREPRODUCIBLE" in fail_codes
    assert reason == (
        "Fold-level evidence disagreed across governed lanes, so non-reproducibility blocked a kill decision."
    )
    assert (
        "Base harness is structurally valid and directionally consistent, but statistical evidence is below threshold."
        not in reason
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_machine_manifest_emitted_and_schema_conformant(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(tmp_path)
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out",
    )
    manifest = _read_json(result.output_dir / "implementation_brief.machine.json")
    assert manifest["brief_name"] == "implementation_brief"
    assert manifest["workstream"] == "RG-09 II-0A"
    assert isinstance(manifest["changes"], list)
    assert isinstance(manifest["tests"], list)
    assert all(isinstance(item, dict) for item in manifest["tests"])
    assert isinstance(manifest["risks"], list)
    assert all(isinstance(item, dict) for item in manifest["risks"])
    assert isinstance(manifest["impacts"], list)
    assert all(isinstance(item, dict) for item in manifest["impacts"])
    assert isinstance(manifest["outputs"], list)
    assert all(isinstance(item, dict) for item in manifest["outputs"])
    assert manifest["status"]["breaking"] is False
    assert manifest["status"]["determinism_tier"] == "D2"
    assert manifest["status"]["mypy_strict"] is True
    assert manifest["status"]["tests_passed"] is True
    assert manifest["status"]["fixture_sha256"].startswith("sha256:")
    assert manifest["status"]["producer_version"] == "rg09-harness/0.1.0"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_machine_manifest_output_paths_follow_run_directory(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(tmp_path)
    result_a = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out_a",
    )
    result_b = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out_b",
    )
    manifest_a = _read_json(result_a.output_dir / "implementation_brief.machine.json")
    manifest_b = _read_json(result_b.output_dir / "implementation_brief.machine.json")
    assert manifest_a["outputs"][0]["path"].startswith(str(tmp_path / "out_a"))
    assert manifest_b["outputs"][0]["path"].startswith(str(tmp_path / "out_b"))
    assert manifest_a["changes"] != []
    assert manifest_b["tests"] != []
    assert manifest_a["outputs"][0]["path"] != manifest_b["outputs"][0]["path"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_emitted_machine_manifest_schema_conformant(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta.rg09_harness import run_rg09_harness

    paths = _write_fixture_bundle(tmp_path)
    result = run_rg09_harness(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=tmp_path / "out_manifest_schema",
    )
    manifest = _read_json(result.output_dir / "implementation_brief.machine.json")
    assert manifest["brief_name"] == "implementation_brief"
    assert isinstance(manifest["changes"], list)
    assert all(isinstance(item["symbols"], dict) for item in manifest["changes"])
    assert all(isinstance(item["covers"], list) for item in manifest["tests"])
    assert all({"type", "severity"} <= set(item) for item in manifest["risks"])
    assert all({"from", "to", "type"} <= set(item) for item in manifest["impacts"])
    assert all({"path", "kind", "deterministic"} <= set(item) for item in manifest["outputs"])
    assert manifest["status"]["determinism_tier"] == "D2"

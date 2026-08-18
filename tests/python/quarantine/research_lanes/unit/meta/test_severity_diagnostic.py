"""RG09-DIAG-002: severity-stratified high-vol diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scripts.rg09_severity_diagnostic import (
    SeverityDiagnosticError,
    _sanitize_for_json,
    classify_episode_severity,
    cohens_d_two_sample,
    discover_fixture_parquets,
    extract_high_vol_episodes,
    main,
    parse_thresholds_arg,
    pit_safe_threshold,
    run_severity_diagnostic,
)


def _row(
    i: int,
    *,
    boundary: str,
    regime: str,
    vol: float,
    ret: float,
    entity: str = "SPY",
) -> dict[str, object]:
    return {
        "decision_ts": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
        "boundary_flag": boundary,
        "regime_class": regime,
        "vol_score_raw": vol,
        "trend_score_raw": ret,
        "entity_id": entity,
    }


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_extraction_basic() -> None:
    df = pd.DataFrame(
        [
            _row(0, boundary="cold_start", regime="sideways", vol=0.1, ret=0.0),
            _row(1, boundary="cold_start", regime="sideways", vol=0.1, ret=0.0),
            _row(2, boundary="stable", regime="high_vol", vol=2.0, ret=0.01),
            _row(3, boundary="stable", regime="high_vol", vol=3.0, ret=-0.01),
            _row(4, boundary="stable", regime="high_vol", vol=2.5, ret=0.0),
            _row(5, boundary="stable", regime="bear", vol=1.0, ret=-0.02),
        ]
    )
    eps = extract_high_vol_episodes(df, entity_id="SPY")
    assert len(eps) == 1
    assert eps[0].episode_length == 3
    assert eps[0].peak_log_rv == 3.0
    assert eps[0].start_idx == 2


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_episode_extraction_excludes_cold_start() -> None:
    df = pd.DataFrame(
        [
            _row(0, boundary="cold_start", regime="high_vol", vol=9.0, ret=0.0),
            _row(1, boundary="stable", regime="high_vol", vol=2.0, ret=0.0),
        ]
    )
    eps = extract_high_vol_episodes(df)
    assert len(eps) == 1
    assert eps[0].episode_length == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_peak_log_rv_computation() -> None:
    df = pd.DataFrame(
        [
            _row(0, boundary="stable", regime="high_vol", vol=1.0, ret=0.0),
            _row(1, boundary="stable", regime="high_vol", vol=5.0, ret=0.0),
            _row(2, boundary="stable", regime="high_vol", vol=2.0, ret=0.0),
        ]
    )
    eps = extract_high_vol_episodes(df)
    assert eps[0].peak_log_rv == 5.0
    assert eps[0].mean_log_rv == pytest.approx((1.0 + 5.0 + 2.0) / 3.0)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_pit_safe_threshold() -> None:
    n = 120
    rows = []
    for i in range(n):
        vol = 1.0 if i < 100 else 10.0
        rc = "high_vol" if i >= 100 else "bull"
        rows.append(
            _row(i, boundary="stable", regime=rc, vol=vol, ret=0.001 * i if i < 100 else -0.01)
        )
    df = pd.DataFrame(rows)
    vol_s = df["vol_score_raw"]
    thr80 = pit_safe_threshold(vol_s, 100, 80.0)
    assert thr80 == pytest.approx(1.0)
    full = vol_s.astype(np.float64).to_numpy()
    full90 = float(np.percentile(full, 90, method="linear"))
    assert full90 > thr80


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_pit_safe_threshold_insufficient_history() -> None:
    df = pd.DataFrame(
        [
            _row(0, boundary="stable", regime="high_vol", vol=5.0, ret=0.0),
        ]
    )
    eps = extract_high_vol_episodes(df)
    assert len(eps) == 1
    cl = classify_episode_severity(eps[0], df["vol_score_raw"], 80.0, cold_start_burn_in=100)
    assert cl is None


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_severity_classification() -> None:
    rows = []
    for i in range(110):
        vol = float(i)
        rows.append(_row(i, boundary="stable", regime="bull", vol=vol, ret=0.0))
    rows.extend(
        [
            _row(110, boundary="stable", regime="high_vol", vol=50.0, ret=0.01),
            _row(111, boundary="stable", regime="high_vol", vol=51.0, ret=0.01),
        ]
    )
    df = pd.DataFrame(rows)
    eps = extract_high_vol_episodes(df)
    assert len(eps) == 1
    ep = eps[0]
    vol_s = df["vol_score_raw"]
    thr = pit_safe_threshold(vol_s, ep.start_idx, 90.0)
    cl = classify_episode_severity(ep, vol_s, 90.0, cold_start_burn_in=50)
    assert cl is not None
    crisis, thr_used = cl
    assert thr_used == thr
    assert crisis == (ep.peak_log_rv > thr_used)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cohens_d_computation() -> None:
    a = [0.0, 1.0, 2.0]
    b = [3.0, 4.0, 5.0]
    d = cohens_d_two_sample(a, b)
    m1, m2 = 1.0, 4.0
    s1, s2 = 1.0, 1.0
    n1, n2 = 3, 3
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    assert d == pytest.approx((m1 - m2) / pooled)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cohens_d_single_group() -> None:
    assert np.isnan(cohens_d_two_sample([], [1.0, 2.0]))
    assert np.isnan(cohens_d_two_sample([1.0], [2.0]))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_decision_table_structure(tmp_path: Path) -> None:
    rows = []
    for i in range(120):
        regime = "high_vol" if 110 <= i < 115 else "bull"
        vol = 10.0 if regime == "high_vol" else 0.5
        rows.append(
            _row(
                i,
                boundary="stable",
                regime=regime,
                vol=vol,
                ret=0.001,
            )
        )
    df = pd.DataFrame(rows)
    basket = tmp_path / "basket"
    ent = basket / "spy"
    ent.mkdir(parents=True)
    df.to_parquet(ent / "rg09_fixture_v1.parquet", index=False)

    report = run_severity_diagnostic(basket, thresholds=(80, 85, 90, 95), cold_start_burn_in=50)
    assert report["diagnostic_id"] == "RG09-DIAG-002"
    assert set(report["thresholds_swept"]) == {80, 85, 90, 95}
    for k in ("p80", "p85", "p90", "p95"):
        tr = report["threshold_results"][k]
        assert "per_entity" in tr
        assert "cross_entity" in tr
        assert "separability" in tr
        assert "cohens_d" in tr["separability"]
    assert len(report["decision_table"]) == 4
    for row in report["decision_table"]:
        assert set(row.keys()) >= {
            "threshold",
            "crisis_grade_constructible",
            "ordinary_constructible",
            "dedup_crisis_events",
            "meets_minimum_20",
            "cohens_d",
        }


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_dedup_integration(tmp_path: Path) -> None:
    def make_basket(ep_start_days: list[int], entity: str) -> pd.DataFrame:
        rows = []
        for i in range(200):
            if i in ep_start_days or any(ep <= i < ep + 3 for ep in ep_start_days):
                regime = "high_vol"
                vol = 100.0
            else:
                regime = "bull"
                vol = 0.1
            rows.append(_row(i, boundary="stable", regime=regime, vol=vol, ret=0.0, entity=entity))
        return pd.DataFrame(rows)

    basket = tmp_path / "b2"
    (basket / "a").mkdir(parents=True)
    (basket / "b").mkdir(parents=True)
    make_basket([120], "A").to_parquet(basket / "a" / "rg09_fixture_v1.parquet", index=False)
    make_basket([120], "B").to_parquet(basket / "b" / "rg09_fixture_v1.parquet", index=False)

    r = run_severity_diagnostic(basket, thresholds=(50,), cold_start_burn_in=50)
    ce = r["threshold_results"]["p50"]["cross_entity"]
    assert ce["naive_crisis_grade_episodes"] >= 2
    assert ce["deduplicated_crisis_grade_events"] <= ce["naive_crisis_grade_episodes"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_discover_fixture_parquets_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(SeverityDiagnosticError):
        discover_fixture_parquets(tmp_path)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_missing_columns_raise() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(SeverityDiagnosticError):
        extract_high_vol_episodes(df)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_pit_safe_threshold_k_positive() -> None:
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        pit_safe_threshold(s, 0, 50.0)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cohens_d_pooled_zero_returns_nan() -> None:
    assert np.isnan(cohens_d_two_sample([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_sanitize_for_json_nan_and_array() -> None:
    raw = {"x": float("nan"), "y": np.array([1.0, 2.0]), "z": (1, 2.5)}
    out = _sanitize_for_json(raw)
    assert out["x"] is None
    assert out["y"] == [1.0, 2.0]
    assert out["z"] == [1, 2.5]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_parse_thresholds_arg_comma_and_space() -> None:
    assert parse_thresholds_arg("80,85,90,95") == (80, 85, 90, 95)
    assert parse_thresholds_arg("80 85 90 95") == (80, 85, 90, 95)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cli_writes_json(tmp_path: Path) -> None:
    rows = []
    for i in range(120):
        regime = "high_vol" if 110 <= i < 115 else "bull"
        vol = 10.0 if regime == "high_vol" else 0.5
        rows.append(_row(i, boundary="stable", regime=regime, vol=vol, ret=0.0))
    df = pd.DataFrame(rows)
    basket = tmp_path / "bk"
    (basket / "spy").mkdir(parents=True)
    df.to_parquet(basket / "spy" / "rg09_fixture_v1.parquet", index=False)
    out = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--basket-dir",
            str(basket),
            "--output",
            str(out),
            "--thresholds",
            "80",
            "--cold-start-burn-in",
            "50",
        ],
    )
    assert result.exit_code == 0
    assert out.is_file()
    assert "RG09-DIAG-002" in out.read_text(encoding="utf-8")

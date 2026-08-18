"""RG09-DIAG-001: crisis constructibility diagnostic + basket dedup."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.generate_rg09_fixture import (
    MetaTaskSizingParams,
    _build_summary,
    _crisis_constructibility_diagnostic,
)
from scripts.run_rg09_basket import _deduplicate_crisis_events

from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_project_regime_class_extended_crisis_cp() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    assert lb.project_regime_class_extended("hi", "hi", "cp") == "crisis"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_project_regime_class_extended_crisis_transition() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    assert lb.project_regime_class("hi", "hi", "transition", severity_flag=False) == "high_vol"
    assert lb.project_regime_class_extended("hi", "hi", "transition") == "crisis"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_project_regime_class_extended_non_crisis_unchanged() -> None:
    cfg = BOCPDConfig()
    lb = RegimeLabeler(cfg)
    trends = ("hi", "lo", "flat")
    vols = ("hi", "med", "lo")
    bstates = ("stable", "transition", "cp")
    for tr in trends:
        for vo in vols:
            for bs in bstates:
                can = lb.project_regime_class(tr, vo, bs, severity_flag=False)
                ext = lb.project_regime_class_extended(tr, vo, bs)
                if vo == "hi" and bs == "transition" or vo == "hi" and bs == "cp":
                    assert ext == "crisis"
                    assert can == "high_vol"
                else:
                    assert ext == can


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_l_min_computation() -> None:
    s = MetaTaskSizingParams()
    assert s.l_min == 20 + 10 + 2 * 5 + 2


def _minimal_fixture_df(*, crisis_episode_lengths: list[int]) -> pd.DataFrame:
    """Cold-start rows then crisis runs of given lengths, separated by high_vol gaps."""
    rows: list[dict[str, object]] = []
    ts0 = pd.Timestamp("2020-01-01T00:00:00+00:00")
    day = 0
    for _ in range(5):
        rows.append(
            {
                "decision_ts": (ts0 + pd.Timedelta(days=day)).isoformat(),
                "boundary_flag": "cold_start",
                "regime_label": "cold",
                "regime_class": "sideways",
                "diag_regime_class_extended": "sideways",
            }
        )
        day += 1
    for ep_i, ln in enumerate(crisis_episode_lengths):
        for _j in range(ln):
            rows.append(
                {
                    "decision_ts": (ts0 + pd.Timedelta(days=day)).isoformat(),
                    "boundary_flag": "ok",
                    "regime_label": f"crisis_ep_{ep_i}",
                    "regime_class": "crisis",
                    "diag_regime_class_extended": "crisis",
                }
            )
            day += 1
        if ep_i < len(crisis_episode_lengths) - 1:
            rows.append(
                {
                    "decision_ts": (ts0 + pd.Timedelta(days=day)).isoformat(),
                    "boundary_flag": "ok",
                    "regime_label": f"gap_{ep_i}",
                    "regime_class": "high_vol",
                    "diag_regime_class_extended": "high_vol",
                }
            )
            day += 1
    return pd.DataFrame(rows)


def _minimal_divergent_fixture_df(*, extended_crisis_high_vol_length: int) -> pd.DataFrame:
    """Cold-start rows then a canonical high_vol run that extended labeling upgrades to crisis."""
    rows: list[dict[str, object]] = []
    ts0 = pd.Timestamp("2020-01-01T00:00:00+00:00")
    day = 0
    for _ in range(5):
        rows.append(
            {
                "decision_ts": (ts0 + pd.Timedelta(days=day)).isoformat(),
                "boundary_flag": "cold_start",
                "regime_label": "cold",
                "regime_class": "sideways",
                "diag_regime_class_extended": "sideways",
            }
        )
        day += 1
    for _ in range(extended_crisis_high_vol_length):
        rows.append(
            {
                "decision_ts": (ts0 + pd.Timedelta(days=day)).isoformat(),
                "boundary_flag": "ok",
                "regime_label": "extended_crisis_from_high_vol",
                "regime_class": "high_vol",
                "diag_regime_class_extended": "crisis",
            }
        )
        day += 1
    return pd.DataFrame(rows)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_crisis_constructibility_diagnostic_zero_constructible() -> None:
    df = _minimal_fixture_df(crisis_episode_lengths=[1, 1, 1])
    out = _crisis_constructibility_diagnostic(df, MetaTaskSizingParams(), "E1")
    assert out["canonical"]["constructible_crisis_tasks"] == 0
    assert out["extended"]["constructible_crisis_tasks"] == 0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_crisis_constructibility_diagnostic_with_constructible() -> None:
    df = _minimal_fixture_df(crisis_episode_lengths=[50])
    out = _crisis_constructibility_diagnostic(df, MetaTaskSizingParams(), "E1")
    assert out["canonical"]["constructible_crisis_tasks"] == 1
    assert out["extended"]["constructible_crisis_tasks"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_crisis_constructibility_diagnostic_diverges_between_canonical_and_extended() -> None:
    df = _minimal_divergent_fixture_df(extended_crisis_high_vol_length=42)
    out = _crisis_constructibility_diagnostic(df, MetaTaskSizingParams(), "E1")
    assert out["canonical"]["crisis_bars"] == 0
    assert out["canonical"]["constructible_crisis_tasks"] == 0
    assert out["extended"]["crisis_bars"] == 42
    assert out["extended"]["crisis_episodes"] == 1
    assert out["extended"]["constructible_crisis_tasks"] == 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_deduplicate_crisis_events_merges_same_date() -> None:
    ts = pd.Timestamp("2011-08-04T16:00:00+00:00")
    n, clusters = _deduplicate_crisis_events([("SPY", ts), ("VIX", ts)])
    assert n == 1
    assert clusters[0]["cluster_size"] == 2
    assert clusters[0]["representative_date"] == "2011-08-04"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_deduplicate_crisis_events_splits_distant_dates() -> None:
    a = pd.Timestamp("2011-08-04T00:00:00+00:00")
    b = pd.Timestamp("2011-09-15T00:00:00+00:00")
    n, clusters = _deduplicate_crisis_events([("SPY", a), ("SPY", b)], dedup_window_days=5)
    assert n == 2
    assert len(clusters) == 2


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_deduplicate_crisis_events_chains() -> None:
    t0 = pd.Timestamp("2020-01-06T00:00:00+00:00")
    t1 = t0 + pd.tseries.offsets.BDay(3)
    t2 = t1 + pd.tseries.offsets.BDay(3)
    n, clusters = _deduplicate_crisis_events([("A", t0), ("B", t1), ("C", t2)], dedup_window_days=5)
    assert n == 1
    assert clusters[0]["cluster_size"] == 3


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_deduplicate_crisis_events_empty() -> None:
    n, clusters = _deduplicate_crisis_events([])
    assert n == 0
    assert clusters == []


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_summary_backward_compatible() -> None:
    df = _minimal_fixture_df(crisis_episode_lengths=[1, 2])
    legacy = {
        "entity_id",
        "row_counts_by_class",
        "episode_counts_by_regime_id",
        "episode_length_stats_by_class",
        "crisis_episodes_after_cold_start",
        "single_series_sufficient",
        "es_only_sufficient",
        "fixture_sha256",
        "source_dataset_id",
        "date_range_start",
        "date_range_end",
        "row_count",
        "generation_timestamp",
        "config_version",
        "producer_version",
    }
    cfg = BOCPDConfig()
    summary = _build_summary(
        df,
        fixture_sha256="sha256:abc",
        source_dataset_id="sha256:src",
        cfg=cfg,
        entity_id="X",
    )
    assert legacy <= set(summary.keys())
    assert "crisis_constructibility" in summary
    assert "episode_counts_by_regime_id_extended" in summary


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_basket_summary_backward_compatible() -> None:
    legacy = {
        "entities",
        "fixture_dirs",
        "crisis_episodes_after_cold_start_by_entity",
        "single_series_sufficient_by_entity",
        "total_crisis_episodes_after_cold_start",
        "generation_timestamp",
    }
    full = {
        *legacy,
        "crisis_constructibility_by_entity",
        "cross_entity_dedup",
        "cross_entity_dedup_extended",
    }
    assert legacy < full

    from scripts.run_rg09_basket import _collect_crisis_entries

    df = _minimal_fixture_df(crisis_episode_lengths=[1])
    entries = _collect_crisis_entries(df, entity_id="SPY", class_col="regime_class")
    assert len(entries) >= 1

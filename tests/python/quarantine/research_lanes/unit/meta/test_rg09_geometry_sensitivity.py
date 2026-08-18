"""Unit tests for RG-09 corrected-surface geometry sensitivity diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _config_with_geometry,
    _single_episode_frame,
    _write_fixture_bundle,
)

from pysrc.meta.rg09_geometry_sensitivity import build_geometry_sensitivity_report


def _seven_bar_three_episode_frame() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for idx, regime_class in enumerate(("bull", "bear", "bull")):
        part = _single_episode_frame(length=7).copy()
        shift = pd.Timedelta(days=7 * idx)
        part.loc[:, "decision_ts"] = pd.to_datetime(part["decision_ts"], utc=True) + shift
        part.loc[:, "effective_at"] = pd.to_datetime(part["effective_at"], utc=True) + shift
        part.loc[:, "regime_id"] = f"trend_{idx}__stable"
        part.loc[:, "regime_label"] = f"trend_{idx}__stable"
        part.loc[:, "regime_class"] = regime_class
        part.loc[:, "diag_regime_class_bocpd_gated"] = regime_class
        part.loc[:, "rg09_trading_day_ord"] = list(range(7 * idx, 7 * (idx + 1)))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_geometry_sensitivity_identifies_support_as_binding_and_marks_dwell_inactive(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _seven_bar_three_episode_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=3,
            min_query_rows=2,
            min_dwell_time_bars=4,
            min_admissible_episode_count=1,
            min_regime_transition_count=0,
            min_support_query_mass_per_regime=1,
            min_regime_class_count_per_fold=1,
            min_temporal_folds=1,
            label_horizon_bars=1,
        ),
    )
    report = build_geometry_sensitivity_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_geometry_sensitivity.json",
        support_values=[3, 2],
        query_values=[2, 1],
        dwell_values=[4, 2],
    )
    assert report["baseline"]["admissible_episode_count"] == 0
    assert report["first_order_binding_constraint_overall"] == "min_support_rows"
    assert report["axes"]["min_support_rows"]["scenarios"][1]["admissible_episode_count"] == 3
    assert (
        report["axes"]["min_dwell_time_bars"]["implementation_status"]
        == "loaded_in_config_but_not_enforced_in_episode_derivation"
    )
    assert report["axes"]["min_dwell_time_bars"]["scenarios"][1]["admissible_episode_count"] == 0

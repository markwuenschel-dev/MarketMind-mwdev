"""Unit tests for transition-anchored RG-09 episode construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _pilot_config_payload,
    _write_fixture_bundle,
    _write_json,
)

from pysrc.core.errors import ConfigValidationError
from pysrc.meta.rg09_harness import (
    RG09PilotConfig,
    _evaluate_fold,
    _materialize_episodes,
    _regime_separability_statistic,
    load_rg09_config,
)
from pysrc.meta.rg09_transition_episodes import derive_transition_anchored_episodes


def _row(
    ts: pd.Timestamp,
    *,
    boundary_flag: str,
    regime_class: str,
    vol_score_raw: float,
    change_probability: float = 0.05,
    entity_id: str = "ES",
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "decision_ts": ts,
        "regime_id": f"rid_{regime_class}",
        "regime_label": f"lbl_{regime_class}",
        "effective_at": ts,
        "state_snapshot_id": "sha256:state",
        "input_snapshot_id": "sha256:input",
        "config_version": "rg09_v1.0.2",
        "change_probability": change_probability,
        "boundary_flag": boundary_flag,
        "regime_class": regime_class,
        "diag_regime_class_bocpd_gated": regime_class,
        "run_length_mode": 1.0,
        "run_length_expectation": 1.0,
        "transition_probability": 0.1,
        "posterior_entropy": 0.02,
        "trend_score_raw": 0.1,
        "vol_score_raw": vol_score_raw,
        "rg09_trading_day_ord": 1,
    }


def _cfg(tmp_path: Path, **overrides: Any) -> RG09PilotConfig:
    defaults: dict[str, Any] = {
        "min_support_rows": 2,
        "min_query_rows": 2,
        "min_admissible_episode_count": 1,
        "min_regime_transition_count": 0,
        "min_support_query_mass_per_regime": 1,
        "min_regime_class_count_per_fold": 1,
        "min_temporal_folds": 2,
        "null_draw_count": 8,
        "episode_construction": "transition_anchored",
    }
    defaults.update(overrides)
    raw = _pilot_config_payload(**defaults)
    p = tmp_path / "pilot.json"
    _write_json(p, raw)
    return load_rg09_config(p)


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_transition_anchor_detected_with_correct_src_dest_and_stable_counts(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ts0 = pd.Timestamp("2024-01-02T00:00:00+00:00")
    rows: list[dict[str, Any]] = []
    for i in range(5):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=0.1,
            )
        )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=5),
            boundary_flag="transition",
            regime_class="bull",
            vol_score_raw=0.2,
        )
    )
    for i in range(5):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=6 + i),
                boundary_flag="stable",
                regime_class="bear",
                vol_score_raw=0.3,
            )
        )
    frame = pd.DataFrame(rows)
    config = _cfg(tmp_path, min_support_rows=2, min_query_rows=2)
    eps, exc = derive_transition_anchored_episodes(
        frame, config, fold_construction=None, require_strict_geometry=False
    )
    assert len(eps) == 1
    assert exc["INSUFFICIENT_TRANSITION_GEOMETRY"] == 0
    assert eps.iloc[0]["src_regime_class"] == "bull"
    assert eps.iloc[0]["regime_class"] == "bear"
    assert eps.iloc[0]["transition_start_ts"] == (ts0 + pd.Timedelta(days=5)).isoformat()


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_support_slice_is_last_n_stable_bars_before_transition(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ts0 = pd.Timestamp("2024-01-01T00:00:00+00:00")
    rows: list[dict[str, Any]] = []
    for i in range(100):
        vol = 0.01 if i < 36 else 9.99
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=vol,
            )
        )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=100),
            boundary_flag="transition",
            regime_class="bull",
            vol_score_raw=0.5,
        )
    )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=101),
            boundary_flag="stable",
            regime_class="bear",
            vol_score_raw=0.5,
        ),
    )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=102),
            boundary_flag="stable",
            regime_class="bear",
            vol_score_raw=0.5,
        ),
    )
    frame = pd.DataFrame(rows)
    config = _cfg(tmp_path, min_support_rows=64, min_query_rows=2)
    eps, _ = derive_transition_anchored_episodes(
        frame, config, fold_construction=None, require_strict_geometry=False
    )
    assert len(eps) == 1
    sm = float(eps.iloc[0]["support_mean"])
    assert sm == pytest.approx(9.99, rel=1e-6)


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_adaptation_gain_positive_when_vol_rises_post_transition(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ts0 = pd.Timestamp("2024-06-01T00:00:00+00:00")
    rows: list[dict[str, Any]] = []
    for i in range(10):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=0.05,
            )
        )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=10),
            boundary_flag="change_point",
            regime_class="bull",
            vol_score_raw=0.5,
        )
    )
    for i in range(10):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=11 + i),
                boundary_flag="stable",
                regime_class="crisis",
                vol_score_raw=2.0,
            )
        )
    frame = pd.DataFrame(rows)
    config = _cfg(tmp_path, min_support_rows=5, min_query_rows=5)
    eps, _ = derive_transition_anchored_episodes(
        frame, config, fold_construction=None, require_strict_geometry=False
    )
    assert len(eps) == 1
    assert float(eps.iloc[0]["adaptation_gain"]) > 0.0


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_insufficient_transition_geometry_excludes_anchor(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ts0 = pd.Timestamp("2024-01-01T00:00:00+00:00")
    rows: list[dict[str, Any]] = []
    for i in range(30):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=0.1,
            )
        )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=30),
            boundary_flag="transition",
            regime_class="bull",
            vol_score_raw=0.2,
        )
    )
    for i in range(40):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=31 + i),
                boundary_flag="stable",
                regime_class="bear",
                vol_score_raw=0.3,
            )
        )
    frame = pd.DataFrame(rows)
    config = _cfg(tmp_path, min_support_rows=64, min_query_rows=32)
    eps, exc = derive_transition_anchored_episodes(
        frame, config, fold_construction=None, require_strict_geometry=False
    )
    assert eps.empty
    assert exc["INSUFFICIENT_TRANSITION_GEOMETRY"] >= 1


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_fold_id_follows_transition_start_ts_not_query_bars(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ts0 = pd.Timestamp("2024-03-15T00:00:00+00:00")
    rows: list[dict[str, Any]] = []
    for i in range(10):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i - 20),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=0.1,
            )
        )
    rows.append(_row(ts0, boundary_flag="transition", regime_class="bull", vol_score_raw=0.2))
    for i in range(50):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i + 1),
                boundary_flag="stable",
                regime_class="bear",
                vol_score_raw=0.3,
            )
        )
    frame = pd.DataFrame(rows)
    config = _cfg(tmp_path, min_support_rows=5, min_query_rows=5, min_temporal_folds=2)
    fc: dict[str, Any] = {
        "method": "calendar_time",
        "time_ranges": {
            "fold_0": ["2024-01-01T00:00:00+00:00", "2024-06-01T00:00:00+00:00"],
            "fold_1": ["2024-06-02T00:00:00+00:00", "2024-12-31T00:00:00+00:00"],
        },
    }
    eps, _ = derive_transition_anchored_episodes(
        frame, config, fold_construction=fc, require_strict_geometry=False
    )
    assert len(eps) == 1
    assert int(eps.iloc[0]["fold_id"]) == 0


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_materialize_episodes_stable_span_calls_derive_episodes(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.meta import rg09_harness as rh
    from pysrc.meta.rg09_boundary_treatment import RG09BoundaryRecoverySpec

    paths = _write_fixture_bundle(tmp_path, config_overrides={})
    config = load_rg09_config(paths["config"])
    assert config.episode_construction == "stable_span"
    frame = pd.read_parquet(paths["fixture"])
    with patch.object(rh, "_derive_episodes", wraps=rh._derive_episodes) as m:
        _materialize_episodes(
            frame,
            config,
            fold_construction=None,
            require_strict_geometry=False,
            boundary_recovery=RG09BoundaryRecoverySpec(mode="baseline"),
        )
        m.assert_called_once()


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_materialize_episodes_transition_calls_derive_transition(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    paths = _write_fixture_bundle(
        tmp_path, config_overrides={"episode_construction": "transition_anchored"}
    )
    config = load_rg09_config(paths["config"])
    frame = pd.read_parquet(paths["fixture"])
    with patch(
        "pysrc.meta.rg09_transition_episodes.derive_transition_anchored_episodes",
        return_value=(pd.DataFrame(), {"COLD_START": 0}),
    ) as m:
        _materialize_episodes(
            frame,
            config,
            fold_construction=None,
            require_strict_geometry=False,
            boundary_recovery=None,
        )
        m.assert_called_once()


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_transition_episode_schema_runs_regime_separability_and_evaluate_fold(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ts0 = pd.Timestamp("2024-01-01T00:00:00+00:00")
    rows: list[dict[str, Any]] = []
    for i in range(8):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=i),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=0.1,
            )
        )
    rows.append(
        _row(
            ts0 + pd.Timedelta(days=8),
            boundary_flag="transition",
            regime_class="bull",
            vol_score_raw=0.2,
        )
    )
    for i in range(8):
        rows.append(
            _row(
                ts0 + pd.Timedelta(days=9 + i),
                boundary_flag="stable",
                regime_class="bear",
                vol_score_raw=0.5,
            )
        )
    rows2: list[dict[str, Any]] = []
    t2 = pd.Timestamp("2024-02-01T00:00:00+00:00")
    for i in range(8):
        rows2.append(
            _row(
                t2 + pd.Timedelta(days=i),
                boundary_flag="stable",
                regime_class="bear",
                vol_score_raw=0.2,
            )
        )
    rows2.append(
        _row(
            t2 + pd.Timedelta(days=8),
            boundary_flag="transition",
            regime_class="bear",
            vol_score_raw=0.2,
        )
    )
    for i in range(8):
        rows2.append(
            _row(
                t2 + pd.Timedelta(days=9 + i),
                boundary_flag="stable",
                regime_class="bull",
                vol_score_raw=0.4,
            )
        )
    frame = pd.DataFrame(rows + rows2)
    config = _cfg(tmp_path, min_support_rows=4, min_query_rows=4)
    eps, _ = derive_transition_anchored_episodes(
        frame, config, fold_construction=None, require_strict_geometry=False
    )
    assert not eps.empty
    _ = _regime_separability_statistic(eps)
    for fid in eps["fold_id"].unique():
        sub = eps.loc[eps["fold_id"] == fid].reset_index(drop=True)
        out = _evaluate_fold(sub, config=config, fixture_sha256="sha256:test", fold_id=int(fid))
        assert "null_families" in out


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_load_rg09_config_rejects_invalid_episode_construction(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    bad = tmp_path / "bad.json"
    _write_json(bad, _pilot_config_payload(episode_construction="invalid_mode"))
    with pytest.raises(ConfigValidationError, match="episode_construction"):
        load_rg09_config(bad)

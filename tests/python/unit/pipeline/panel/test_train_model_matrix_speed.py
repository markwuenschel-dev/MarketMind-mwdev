"""Train-matrix speed helpers: folds and subsampling."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.pipeline.canonical_data import CanonicalPanelSource, assert_canonical_source_unchanged
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.panel.train_model_matrix import (
    _assert_group_index_row_count,
    _enforce_peak_rss_limit,
    _poll_worker_until_exit,
    _validate_quantile_backend_policy,
    build_chronological_date_codes,
    build_walk_forward_boundaries,
    build_walk_forward_folds,
    build_walk_forward_folds_from_codes,
    fold_masks_from_boundaries,
    initialize_train_matrix_scratch_dir,
    normalize_date_labels,
    rank_prediction_frame,
    resolve_panel_memory_mode,
    resolve_schema_target_and_features,
    resolve_train_matrix_scratch_dir,
    resolve_train_row_policy,
    subsample_train_indices,
)


def _scratch_source(tmp_path: Path) -> CanonicalPanelSource:
    return CanonicalPanelSource(
        panel_path=tmp_path / "panel.parquet",
        manifest_path=None,
        manifest={},
        schema={"date": "string", "instrument": "string", "interval": "string"},
        row_count=10,
        expected_row_grain="ticker_date_interval",
        product_identity="test.panel",
        file_size_bytes=None,
        file_mtime_ns=None,
        config_hash=None,
        target_metadata={},
    )


@pytest.mark.determinism("d1")
def test_uncapped_quantile_requires_hist_backend(deterministic_seed: int) -> None:
    _ = deterministic_seed
    config = P2Config(panel_train_max_rows_per_fold=0, panel_quantile_max_train_rows=0)

    with pytest.raises(ValueError, match="Uncapped quantile_regression requires"):
        _validate_quantile_backend_policy(
            [{"family": "quantile_regression", "params": {"quantile": 0.5, "solver": "highs"}}],
            config,
        )

    _validate_quantile_backend_policy(
        [
            {
                "family": "quantile_regression",
                "params": {"backend": "hist_gradient_boosting", "quantile": 0.5},
            }
        ],
        config,
    )


@pytest.mark.determinism("d1")
def test_parent_polling_terminates_worker_on_rss_breach(deterministic_seed: int) -> None:
    _ = deterministic_seed

    class FakeProcess:
        pid = None

        def __init__(self) -> None:
            self.terminated = False
            self.join_calls = 0

        def is_alive(self) -> bool:
            return not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout: float | None = None) -> None:
            del timeout
            self.join_calls += 1

    process = FakeProcess()
    samples = iter([1, 9])

    with pytest.raises(MemoryError, match="worker terminated"):
        _poll_worker_until_exit(
            process,
            limit_bytes=8,
            timeout_s=0,
            context="unit-worker",
            rss_sampler=lambda: next(samples),
            poll_interval_s=0.0,
        )

    assert process.terminated is True
    assert process.join_calls >= 1


@pytest.mark.determinism("d1")
def test_parent_polling_terminates_worker_on_timeout(deterministic_seed: int) -> None:
    _ = deterministic_seed

    class FakeProcess:
        pid = None

        def __init__(self) -> None:
            self.terminated = False
            self.join_calls = 0

        def is_alive(self) -> bool:
            return not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout: float | None = None) -> None:
            del timeout
            self.join_calls += 1

    process = FakeProcess()
    start = time.monotonic()

    with pytest.raises(TimeoutError, match="worker terminated"):
        _poll_worker_until_exit(
            process,
            limit_bytes=0,
            timeout_s=1,
            context="unit-worker",
            rss_sampler=lambda: 1,
            poll_interval_s=0.0,
        )

    assert process.terminated is True
    assert time.monotonic() - start < 5.0


@pytest.mark.determinism("d1")
def test_resolve_train_matrix_scratch_dir_is_run_scoped(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    run_path = tmp_path / "artifacts" / "runs" / "model_matrix_abc123"
    run_path.mkdir(parents=True)

    internal = resolve_train_matrix_scratch_dir(
        config=P2Config(panel_train_scratch_dir=None),
        run_path=run_path,
    )
    assert internal == run_path / "scratch" / "train_matrix"

    external_root = tmp_path / "scratch_root"
    external = resolve_train_matrix_scratch_dir(
        config=P2Config(panel_train_scratch_dir=str(external_root)),
        run_path=run_path,
    )
    assert external == external_root / "model_matrix_abc123" / "train_matrix"


@pytest.mark.determinism("d1")
def test_initialize_scratch_dir_rejects_foreign_owner(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    scratch_dir = tmp_path / "scratch" / "model_matrix_aaa" / "train_matrix"
    scratch_dir.mkdir(parents=True)
    owner_path = scratch_dir / "scratch_owner.json"
    owner_path.write_text(
        json.dumps({"schema_version": "train_matrix_scratch_owner.v1", "run_id": "other_run"}),
        encoding="utf-8",
    )
    source = _scratch_source(tmp_path)

    with pytest.raises(RuntimeError, match="owned by another run"):
        initialize_train_matrix_scratch_dir(
            scratch_dir,
            run_id="model_matrix_bbb",
            source=source,
        )


@pytest.mark.determinism("d1")
def test_initialize_scratch_dir_recreates_same_run_retry(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    scratch_dir = tmp_path / "scratch" / "model_matrix_retry" / "train_matrix"
    scratch_dir.mkdir(parents=True)
    stale = scratch_dir / "packed_keys_000.bin"
    stale.write_bytes(b"stale")
    owner_path = scratch_dir / "scratch_owner.json"
    owner_path.write_text(
        json.dumps(
            {"schema_version": "train_matrix_scratch_owner.v1", "run_id": "model_matrix_retry"}
        ),
        encoding="utf-8",
    )
    source = _scratch_source(tmp_path)

    initialize_train_matrix_scratch_dir(
        scratch_dir,
        run_id="model_matrix_retry",
        source=source,
    )

    assert owner_path.is_file()
    assert not stale.exists()
    payload = json.loads(owner_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "model_matrix_retry"


@pytest.mark.determinism("d1")
def test_group_index_row_count_must_match_panel_rows(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    group_dir = tmp_path / "group_indices"
    group_dir.mkdir(parents=True)
    (group_dir / "group_00000000.int64").write_bytes(np.array([0, 1, 2], dtype=np.int64).tobytes())
    (group_dir / "group_00000001.int64").write_bytes(np.array([3], dtype=np.int64).tobytes())

    assert _assert_group_index_row_count(group_index_dir=group_dir, expected_rows=4) == 4

    with pytest.raises(ValueError, match="Group-index row count mismatch"):
        _assert_group_index_row_count(group_index_dir=group_dir, expected_rows=5)


@pytest.mark.determinism("d1")
def test_assert_canonical_source_unchanged_detects_panel_mutation(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"original")
    stat = panel_path.stat()
    source = CanonicalPanelSource(
        panel_path=panel_path,
        manifest_path=None,
        manifest={},
        schema={"date": "string", "instrument": "string", "interval": "string"},
        row_count=1,
        expected_row_grain="ticker_date_interval",
        product_identity="test.panel",
        file_size_bytes=int(stat.st_size),
        file_mtime_ns=int(stat.st_mtime_ns),
        config_hash=None,
        target_metadata={},
    )

    assert_canonical_source_unchanged(source)

    panel_path.write_bytes(b"mutated-panel")
    with pytest.raises(RuntimeError, match="file size changed"):
        assert_canonical_source_unchanged(source)


@pytest.mark.determinism("d1")
def test_peak_rss_limit_uses_sampled_peak(deterministic_seed: int) -> None:
    _ = deterministic_seed

    with pytest.raises(MemoryError, match="peak process-tree RSS"):
        _enforce_peak_rss_limit(peak_bytes=9, limit_bytes=8, context="unit")

    _enforce_peak_rss_limit(peak_bytes=8, limit_bytes=8, context="unit")
    _enforce_peak_rss_limit(peak_bytes=900, limit_bytes=0, context="unit")


@pytest.mark.determinism("d1")
def test_date_normalization_handles_times_and_formats(deterministic_seed: int) -> None:
    _ = deterministic_seed
    dates = np.asarray(
        [
            "2024-01-03T12:00:00-05:00",
            "2024/01/01 00:00:00Z",
            "2024-1-2",
        ],
        dtype=object,
    )

    normalized = normalize_date_labels(dates)
    assert normalized.tolist() == ["2024-01-03", "2024-01-01", "2024-01-02"]
    codes, unique_dates = build_chronological_date_codes(dates)
    assert unique_dates.tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert codes.tolist() == [2, 0, 1]


@pytest.mark.determinism("d1")
def test_build_walk_forward_folds_matches_vectorized_codes(deterministic_seed: int) -> None:
    _ = deterministic_seed
    dates = ["2020-01-01"] * 3 + ["2020-01-02"] * 2 + ["2020-01-03"] * 4
    legacy = build_walk_forward_folds(dates, n_folds=2)
    codes, _ = __import__("pandas").factorize(__import__("pandas").Series(dates, dtype="string"))
    codes_arr = np.asarray(codes, dtype=np.intp)
    vectorized = build_walk_forward_folds_from_codes(
        codes_arr,
        n_unique_dates=int(codes_arr.max()) + 1,
        n_folds=2,
    )
    assert len(legacy) == len(vectorized)
    for left, right in zip(legacy, vectorized, strict=True):
        assert left.fold_id == right.fold_id
        np.testing.assert_array_equal(left.train_mask, right.train_mask)
        np.testing.assert_array_equal(left.test_mask, right.test_mask)


@pytest.mark.determinism("d1")
def test_subsample_train_indices_is_deterministic_and_capped(deterministic_seed: int) -> None:
    _ = deterministic_seed
    train_indices = np.arange(10_000, dtype=np.intp)
    first = subsample_train_indices(
        train_indices,
        max_rows=500,
        master_seed=42,
        family="random_forest",
        fold_id="fold_0",
    )
    second = subsample_train_indices(
        train_indices,
        max_rows=500,
        master_seed=42,
        family="random_forest",
        fold_id="fold_0",
    )
    assert len(first) == 500
    np.testing.assert_array_equal(first, second)

    unchanged = subsample_train_indices(
        train_indices,
        max_rows=0,
        master_seed=42,
        family="random_forest",
        fold_id="fold_0",
    )
    np.testing.assert_array_equal(unchanged, train_indices)


@pytest.mark.determinism("d1")
def test_resolve_train_row_limit_full_data_except_quantile(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.panel.train_model_matrix import resolve_train_row_limit

    config = P2Config(panel_train_max_rows_per_fold=0, panel_quantile_max_train_rows=0)
    assert resolve_train_row_limit(family="random_forest", config=config) == 0
    assert resolve_train_row_limit(family="ridge", config=config) == 0
    assert resolve_train_row_limit(family="quantile_regression", config=config) == 0

    capped = P2Config(panel_train_max_rows_per_fold=250_000)
    assert resolve_train_row_limit(family="mlp", config=capped) == 250_000


@pytest.mark.determinism("d1")
def test_chronological_folds_purge_label_horizon(deterministic_seed: int) -> None:
    _ = deterministic_seed
    dates = np.asarray(
        [
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
            "2024-01-04",
            "2024-01-05",
        ],
        dtype=object,
    )

    codes, unique_dates = build_chronological_date_codes(dates)
    assert unique_dates.tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert codes.tolist() == [2, 0, 1, 3, 4]

    boundaries = build_walk_forward_boundaries(
        unique_dates,
        n_folds=1,
        target_horizon_days=1,
    )
    train_mask, test_mask = fold_masks_from_boundaries(codes, boundaries[0])
    assert dates[train_mask].tolist() == ["2024-01-01"]
    assert dates[test_mask].tolist() == ["2024-01-03", "2024-01-04"]


@pytest.mark.determinism("d1")
def test_train_row_policy_cli_precedence(deterministic_seed: int) -> None:
    _ = deterministic_seed
    base = P2Config(panel_train_max_rows_per_fold=111, panel_quantile_max_train_rows=222)

    assert resolve_train_row_policy(base).general_max_rows == 111
    assert resolve_train_row_policy(base).quantile_max_rows == 222

    zero = resolve_train_row_policy(base, cli_max_train_rows=0)
    assert zero.general_max_rows is None
    assert zero.quantile_max_rows is None

    general = resolve_train_row_policy(base, cli_max_train_rows=333)
    assert general.general_max_rows == 333
    assert general.quantile_max_rows == 333

    quantile_override = resolve_train_row_policy(
        base,
        cli_max_train_rows=333,
        cli_quantile_max_train_rows=0,
    )
    assert quantile_override.general_max_rows == 333
    assert quantile_override.quantile_max_rows is None


@pytest.mark.determinism("d1")
def test_schema_target_and_features_match_manifest_order(deterministic_seed: int) -> None:
    _ = deterministic_seed
    schema = {
        "date": "string",
        "instrument": "string",
        "interval": "string",
        "z_feature": "float",
        "a_feature": "float",
        "adjusted_return_1d": "double",
        "text_feature": "string",
    }
    manifest = {
        "indicator_columns": ["z_feature", "a_feature", "missing_feature", "text_feature"],
        "target_metadata": {"forward_return": {"column": "adjusted_return_1d", "horizon_days": 1}},
    }
    target, features, report, metadata = resolve_schema_target_and_features(
        P2Config(panel_target="forward_return", panel_target_horizon_days=1),
        schema,
        manifest=manifest,
    )

    assert target == "adjusted_return_1d"
    assert features == ["z_feature", "a_feature"]
    assert report["eligible_feature_count"] == 2
    assert report["used_feature_count"] == 2
    assert metadata["horizon_days"] == 1


@pytest.mark.determinism("d1")
def test_auto_memory_mode_is_deterministic(deterministic_seed: int) -> None:
    _ = deterministic_seed
    config = P2Config(panel_train_memory_mode="auto")

    small = resolve_panel_memory_mode(
        config,
        panel_rows=1_000,
        feature_count=10,
        model_count=2,
        largest_train_rows=700,
        largest_test_rows=300,
        available_memory_bytes=8 * 1024**3,
    )
    assert small.resolved_memory_mode == "in_memory"

    large = resolve_panel_memory_mode(
        config,
        panel_rows=12_603_273,
        feature_count=31,
        model_count=10,
        largest_train_rows=9_000_000,
        largest_test_rows=3_000_000,
        available_memory_bytes=8 * 1024**3,
    )
    assert large.resolved_memory_mode == "low_memory"
    assert large.estimated_in_memory_peak_bytes > 0


@pytest.mark.determinism("d1")
def test_prediction_ranks_are_interval_aware(deterministic_seed: int) -> None:
    _ = deterministic_seed
    frame = pd.DataFrame(
        {
            "model_id": ["ridge", "ridge", "ridge", "ridge"],
            "fold_id": ["fold_0", "fold_0", "fold_0", "fold_0"],
            "date": ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"],
            "interval": ["1d", "1d", "1h", "1h"],
            "prediction": [0.1, 0.2, 0.3, 0.1],
        }
    )
    ranked = rank_prediction_frame(frame)

    daily = ranked.loc[ranked["interval"].eq("1d"), "prediction_rank"].tolist()
    hourly = ranked.loc[ranked["interval"].eq("1h"), "prediction_rank"].tolist()
    assert daily == [2, 1]
    assert hourly == [1, 2]

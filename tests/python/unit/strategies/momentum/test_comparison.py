from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

import pysrc.strategies.momentum.comparison as comparison_module
from pysrc.artifact_registry.run_registry import RunRegistry
from pysrc.backtesting.contracts.types import PitMeta
from pysrc.backtesting.validation.statistical.pbo_bridge import build_pbo_path_pairs
from pysrc.preprocessor.graph.factory import register_builtin_ops
from pysrc.strategies.momentum.artifacts.cpcv_path_scores import (
    build_cpcv_path_score_surface,
    compute_payload_hash,
    normalize_close_prices,
    normalize_weights,
)
from pysrc.strategies.momentum.comparison import (
    _build_shared_cpcv_surface,
    _verify_shared_artifact_hash,
)
from pysrc.strategies.pipeline_strategy import StrategyContext, TradeIntent

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _comparison_corpus(periods: int = 320) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    symbols = ("AAA", "BBB", "CCC")
    start = date(2023, 1, 1)
    for day_idx in range(periods):
        valid_time = start + timedelta(days=day_idx)
        for symbol_idx, symbol in enumerate(symbols):
            oscillation = ((day_idx % 11) - 5) * (symbol_idx + 1) * 0.7
            reversal = (-1.0 if day_idx % 2 == 0 else 1.0) * (symbol_idx + 1) * 0.35
            trend = day_idx * (1.0 + (symbol_idx * 0.05))
            rows.append(
                {
                    "symbol": symbol,
                    "valid_time": valid_time,
                    "knowledge_time": valid_time,
                    "close": 100.0 + (symbol_idx * 8.0) + trend + oscillation + reversal,
                }
            )
    return pd.DataFrame(rows)


def _comparison_ctx(tmp_path) -> StrategyContext:
    return StrategyContext(
        prices=_comparison_corpus(),
        backend="pandas",
        cache_dir=tmp_path / "comparison-root",
        pit_provenance=PitMeta(
            as_of="2023-11-17T00:00:00",
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff="2023-11-17",
        ),
    )


def test_normalize_close_prices_accepts_wide_frame(deterministic_seed) -> None:
    _ = deterministic_seed
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0],
            "B": [99.0, 100.0, 101.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    normalized = normalize_close_prices(prices)
    assert list(normalized.columns) == ["A", "B"]
    assert normalized.iloc[0, 0] == pytest.approx(100.0)


def test_normalize_close_prices_accepts_long_pit_frame(deterministic_seed) -> None:
    _ = deterministic_seed
    prices = pd.DataFrame(
        [
            {
                "symbol": "A",
                "valid_time": "2024-01-01",
                "knowledge_time": "2024-01-01",
                "close": 100.0,
            },
            {
                "symbol": "B",
                "valid_time": "2024-01-01",
                "knowledge_time": "2024-01-01",
                "close": 99.0,
            },
            {
                "symbol": "A",
                "valid_time": "2024-01-02",
                "knowledge_time": "2024-01-02",
                "close": 101.0,
            },
            {
                "symbol": "B",
                "valid_time": "2024-01-02",
                "knowledge_time": "2024-01-02",
                "close": 100.0,
            },
        ]
    )
    normalized = normalize_close_prices(prices)
    assert list(normalized.columns) == ["A", "B"]
    assert normalized.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(101.0)


def test_normalize_close_prices_accepts_single_asset_series(deterministic_seed) -> None:
    _ = deterministic_seed
    prices = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
        name="close",
    )
    normalized = normalize_close_prices(prices)
    assert list(normalized.columns) == ["asset"]
    assert normalized.iloc[-1, 0] == pytest.approx(102.0)


def test_normalize_weights_accepts_multiindex_series(deterministic_seed) -> None:
    _ = deterministic_seed
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2, freq="D"), ["A", "B"]],
        names=["date", "asset"],
    )
    weights = pd.Series([0.5, -0.5, 0.25, -0.25], index=index)
    normalized = normalize_weights(weights)
    assert list(normalized.columns) == ["A", "B"]
    assert normalized.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(0.25)


def test_build_shared_cpcv_surface_uses_production_profile_and_stable_hash(
    deterministic_seed,
) -> None:
    _ = deterministic_seed
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    prices = pd.DataFrame({"asset": [100.0 + idx for idx in range(24)]}, index=index)
    surface = _build_shared_cpcv_surface(prices)
    assert surface["payload"]["split_method"] == "cpcv"
    assert surface["payload"]["purge_window"] == 0
    assert surface["payload"]["embargo_window"] == 0
    assert surface["hash"].startswith("sha256:")
    assert len(surface["payload"]["splits"]) == 15


def test_verify_shared_artifact_hash_rejects_mismatch(deterministic_seed, tmp_path) -> None:
    _ = deterministic_seed
    payload = {
        "schema_version": "1.0.0",
        "split_method": "cpcv",
        "purge_window": 0,
        "embargo_window": 0,
        "splits": [{"path_id": "path-001"}],
    }
    path = tmp_path / "splits_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split surface hash mismatch"):
        _verify_shared_artifact_hash(path, expected_hash="sha256:deadbeef", label="split surface")


def test_compute_payload_hash_is_stable_for_key_order(deterministic_seed) -> None:
    _ = deterministic_seed
    first = compute_payload_hash({"b": 2, "a": 1})
    second = compute_payload_hash({"a": 1, "b": 2})
    assert first == second


def test_build_cpcv_path_score_surface_returns_deterministic_child_artifact(
    deterministic_seed,
) -> None:
    _ = deterministic_seed
    prices = pd.DataFrame(
        [
            {
                "symbol": "A",
                "valid_time": "2024-01-01",
                "knowledge_time": "2024-01-01",
                "close": 100.0,
            },
            {
                "symbol": "B",
                "valid_time": "2024-01-01",
                "knowledge_time": "2024-01-01",
                "close": 98.0,
            },
            {
                "symbol": "A",
                "valid_time": "2024-01-02",
                "knowledge_time": "2024-01-02",
                "close": 101.0,
            },
            {
                "symbol": "B",
                "valid_time": "2024-01-02",
                "knowledge_time": "2024-01-02",
                "close": 99.0,
            },
            {
                "symbol": "A",
                "valid_time": "2024-01-03",
                "knowledge_time": "2024-01-03",
                "close": 102.0,
            },
            {
                "symbol": "B",
                "valid_time": "2024-01-03",
                "knowledge_time": "2024-01-03",
                "close": 100.0,
            },
            {
                "symbol": "A",
                "valid_time": "2024-01-04",
                "knowledge_time": "2024-01-04",
                "close": 103.0,
            },
            {
                "symbol": "B",
                "valid_time": "2024-01-04",
                "knowledge_time": "2024-01-04",
                "close": 101.0,
            },
        ]
    )
    features = pd.DataFrame(
        {
            "valid_time": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "returns": [0.01, -0.01, 0.02, -0.02],
        }
    )
    weights = pd.Series(
        [0.5, -0.5, 0.25, -0.25],
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(features["valid_time"]), features["symbol"]],
            names=["valid_time", "symbol"],
        ),
    )
    trade_intent = TradeIntent(
        weights=weights,
        raw={"features": features},
        diagnostics={},
    )
    splits_manifest = {
        "schema_version": "1.0.0",
        "split_method": "cpcv",
        "purge_window": 0,
        "embargo_window": 0,
        "splits": [
            {
                "path_id": "path-000",
                "split_index": 0,
                "train_indices": [0, 1],
                "test_indices": [2, 3],
                "test_group_ids": [0],
            }
        ],
    }

    payload = build_cpcv_path_score_surface(
        variant="xsec",
        trade_intent=trade_intent,
        prices=prices,
        splits_manifest=splits_manifest,
        commission_bps=5.0,
        slippage_bps=1.0,
        cost_model_id="momentum.phase_i.default",
    )

    assert payload["schema_version"] == "1.0.0"
    assert payload["variant"] == "xsec"
    assert payload["split_surface_hash"] == compute_payload_hash(splits_manifest)
    assert payload["shared_cost_hash"] == compute_payload_hash(
        {
            "commission_bps": 5.0,
            "slippage_bps": 1.0,
            "cost_model_id": "momentum.phase_i.default",
        }
    )
    assert payload["evaluations"][0]["trial_id"] == "xsec"
    assert payload["evaluations"][0]["path_id"] == "path-000"
    assert isinstance(payload["evaluations"][0]["in_sample_net_sharpe"], float)
    assert isinstance(payload["evaluations"][0]["out_of_sample_net_sharpe"], float)
    assert build_pbo_path_pairs(payload["evaluations"])[0]["path_id"] == "path-000"
    assert payload["summary"]["mean_turnover"] >= 0.0


def test_run_variant_comparison_fails_closed_on_cost_identity_mismatch(
    deterministic_seed,
    tmp_path,
    monkeypatch,
) -> None:
    _ = deterministic_seed
    register_builtin_ops()
    original_run = comparison_module.momentum_entry.run

    def _tampered_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if kwargs.get("variant") == "dual":
            payload = json.loads(
                (result.bundle_dir / "cpcv_path_scores.json").read_text(encoding="utf-8")
            )
            payload["shared_cost_hash"] = "sha256:tampered"
            (result.bundle_dir / "cpcv_path_scores.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(comparison_module.momentum_entry, "run", _tampered_run)

    with pytest.raises(ValueError, match="shared cost model"):
        comparison_module.run_variant_comparison(
            _comparison_ctx(tmp_path),
            bundle_dir=tmp_path / "comparison-bundle",
            run_registry=RunRegistry(tmp_path / "registry"),
        )


def test_run_variant_comparison_fails_closed_on_split_identity_mismatch(
    deterministic_seed,
    tmp_path,
    monkeypatch,
) -> None:
    _ = deterministic_seed
    register_builtin_ops()
    original_run = comparison_module.momentum_entry.run

    def _tampered_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if kwargs.get("variant") == "dual":
            payload = json.loads(
                (result.bundle_dir / "cpcv_path_scores.json").read_text(encoding="utf-8")
            )
            payload["split_surface_hash"] = "sha256:tampered"
            (result.bundle_dir / "cpcv_path_scores.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(comparison_module.momentum_entry, "run", _tampered_run)

    with pytest.raises(ValueError, match="split surface hash mismatch"):
        comparison_module.run_variant_comparison(
            _comparison_ctx(tmp_path),
            bundle_dir=tmp_path / "comparison-bundle",
            run_registry=RunRegistry(tmp_path / "registry"),
        )


def test_run_variant_comparison_fails_closed_on_missing_child_cpcv_path_scores(
    deterministic_seed,
    tmp_path,
    monkeypatch,
) -> None:
    _ = deterministic_seed
    register_builtin_ops()
    original_run = comparison_module.momentum_entry.run

    def _tampered_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if kwargs.get("variant") == "dual":
            (result.bundle_dir / "cpcv_path_scores.json").unlink()
        return result

    monkeypatch.setattr(comparison_module.momentum_entry, "run", _tampered_run)

    with pytest.raises(FileNotFoundError, match="cpcv_path_scores.json"):
        comparison_module.run_variant_comparison(
            _comparison_ctx(tmp_path),
            bundle_dir=tmp_path / "comparison-bundle",
            run_registry=RunRegistry(tmp_path / "registry"),
        )


def test_run_variant_comparison_fails_closed_on_malformed_child_cpcv_path_scores(
    deterministic_seed,
    tmp_path,
    monkeypatch,
) -> None:
    _ = deterministic_seed
    register_builtin_ops()
    original_run = comparison_module.momentum_entry.run

    def _tampered_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if kwargs.get("variant") == "dual":
            (result.bundle_dir / "cpcv_path_scores.json").write_text(
                json.dumps({"schema_version": "1.0.0", "variant": "dual", "evaluations": []}),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(comparison_module.momentum_entry, "run", _tampered_run)

    with pytest.raises(
        (KeyError, ValueError), match="split_surface_hash|shared_cost_hash|non-empty"
    ):
        comparison_module.run_variant_comparison(
            _comparison_ctx(tmp_path),
            bundle_dir=tmp_path / "comparison-bundle",
            run_registry=RunRegistry(tmp_path / "registry"),
        )

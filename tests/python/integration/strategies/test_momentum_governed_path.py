from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import pytest

from pysrc.backtesting.contracts.types import PitMeta
from pysrc.backtesting.data.pit import PITSafeDataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter
from pysrc.preprocessor.graph.factory import register_builtin_ops
from pysrc.strategies.momentum.entry import run
from pysrc.strategies.pipeline_strategy import MaterializationError, StrategyContext

pytestmark = [pytest.mark.integration, pytest.mark.determinism("d1")]


def _momentum_corpus() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = [
        100.0,
        101.5,
        103.0,
        102.0,
        104.5,
        106.0,
        107.0,
        108.5,
        110.0,
        111.5,
        113.0,
        114.0,
    ]
    for idx, close in enumerate(closes, start=1):
        valid_time = date(2024, 1, idx)
        knowledge_time = date(2024, 1, idx if idx < 6 else idx + 1)
        rows.append(
            {
                "symbol": "SPY",
                "valid_time": valid_time,
                "knowledge_time": knowledge_time,
                "close": close,
            }
        )
    return pd.DataFrame(rows)


def _ctx(tmp_path, prices: pd.DataFrame) -> StrategyContext:
    return StrategyContext(
        prices=prices,
        backend="pandas",
        cache_dir=tmp_path / "bundle",
        pit_provenance=PitMeta(
            as_of="2024-01-13T00:00:00",
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff="2024-01-13",
        ),
    )


def test_momentum_governed_path_proves_canonical_wiring(tmp_path, monkeypatch) -> None:
    register_builtin_ops()

    executed_graph_ops: list[str] = []
    artifact_roles: list[str] = []
    orchestrator_call: dict[str, Any] = {"called": False, "pit_input": None}

    import pysrc.pipeline.orchestrator as orchestrator
    import pysrc.strategies.pipeline_strategy as pipeline_strategy
    from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore

    original_execute_graph_step = pipeline_strategy._execute_graph_step
    original_put_json = BundleBacktestArtifactStore.put_json
    original_orchestrator_run = orchestrator.run

    def record_execute_graph_step(step, resolved_key, feats):
        executed_graph_ops.append(resolved_key)
        return original_execute_graph_step(step, resolved_key, feats)

    def record_put_json(self, role: str, payload: dict[str, Any]):
        artifact_roles.append(role)
        return original_put_json(self, role, payload)

    def record_orchestrator_run(*args, **kwargs):
        orchestrator_call["called"] = True
        orchestrator_call["pit_input"] = kwargs.get("pit_input")
        return original_orchestrator_run(*args, **kwargs)

    monkeypatch.setattr(pipeline_strategy, "_execute_graph_step", record_execute_graph_step)
    monkeypatch.setattr(BundleBacktestArtifactStore, "put_json", record_put_json)
    monkeypatch.setattr(orchestrator, "run", record_orchestrator_run)

    result = run(_ctx(tmp_path, _momentum_corpus()), orchestrator_hooks=None)

    signal_card = json.loads((result.bundle_dir / "signal_card.json").read_text(encoding="utf-8"))
    stat_validity = json.loads(
        (result.bundle_dir / "stat_validity_report.json").read_text(encoding="utf-8")
    )

    assert orchestrator_call["called"] is True
    assert isinstance(orchestrator_call["pit_input"], (PITSafeDataView, DataViewAsOfAdapter))
    assert "feature.returns" in executed_graph_ops
    assert {
        "execution_assumptions.json",
        "signal_card.json",
        "stat_validity_report.json",
    }.issubset(set(artifact_roles))
    assert signal_card["schema_version"] == "v1"
    assert stat_validity["schema_version"] == "v1"
    assert (result.bundle_dir / "plan.json").exists()
    assert (result.bundle_dir / "env_fingerprint.json").exists()
    assert (result.bundle_dir / "dataset_manifest.json").exists()
    assert (result.bundle_dir / "preprocessing_report.json").exists()
    assert (result.bundle_dir / "splits_manifest.json").exists()


def test_momentum_governed_path_rejects_non_pit_corpus_input(tmp_path) -> None:
    register_builtin_ops()

    bad_prices = _momentum_corpus().drop(columns=["knowledge_time"])

    with pytest.raises(
        (MaterializationError, ValueError),
        match="knowledge_time|PIT|bitemporal",
    ):
        run(_ctx(tmp_path, bad_prices), orchestrator_hooks=None)

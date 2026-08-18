from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from marketmind_gate.gates.phase2_ml_evidence_shell import (
    Phase2MLEvidenceShellStatus,
    evaluate_phase2_ml_evidence_shell,
)
from pysrc.pipeline.orchestrator import OrchestratorConfig, run, run_orchestration
from pysrc.strategies.pipeline_strategy import StrategyContext, StrategyRegistry, TradeIntent

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


class _Phase2OrchestrationStubStrategy:
    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def generate_trade_intent(self, ctx: StrategyContext) -> TradeIntent:
        assert isinstance(ctx.prices, pd.DataFrame)
        signal = pd.Series([0.0, 1.0, 0.0, -1.0], index=ctx.prices.index, name="stub_signal")
        features = pd.DataFrame(
            {
                "returns": [0.0, 0.01, -0.005, 0.02],
                "feature_stub": [1.0, 1.5, 1.25, 1.75],
            },
            index=ctx.prices.index,
        )
        return TradeIntent(weights=signal, raw={"signal": signal, "features": features})


def _ctx(tmp_path: Path) -> StrategyContext:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    return StrategyContext(
        prices=pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=idx),
        backend="pandas",
        cache_dir=tmp_path / "cache",
    )


def _source_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * 4,
            "valid_time": idx,
            "knowledge_time": idx,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )


def _register_stub(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setitem(StrategyRegistry._REGISTRY, name, _Phase2OrchestrationStubStrategy)


def _write_stale_root_triple(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "task_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "rg09.task_manifest.scaffold.v1",
                "non_promotable": True,
                "tasks": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "meta_validity_report.json").write_text(
        json.dumps(
            {
                "schema_version": "meta_validity_report.scaffold.v1",
                "non_promotable": True,
                "overall_result": "SCAFFOLD_INCOMPLETE",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "execution_assumptions.json").write_text(
        json.dumps(
            {
                "schema_version": "execution_assumptions.scaffold.v1",
                "non_promotable": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


@pytest.mark.determinism("d1")
def test_canonical_orchestration_run_emits_governed_phase2_evidence_subdir(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.phase2_governed import (
        PHASE2_GOVERNED_EVIDENCE_SUBDIR,
        PHASE2_GOVERNED_SUMMARY_FILENAME,
    )

    _register_stub(monkeypatch, "phase2_orchestration_stub")

    bundle_dir = tmp_path / "bundle"
    run(
        strategy_id="phase2_orchestration_stub",
        ctx=_ctx(tmp_path),
        strategy_kwargs={},
        bundle_dir=bundle_dir,
        source_prices=_source_prices(),
        run_metadata={
            "content_hash": "sha256:" + ("a" * 64),
            "content_hash_expected": "sha256:" + ("a" * 64),
            "download_timestamp": "2026-04-17T16:22:00Z",
        },
        run_id="canonical-run-stable",
    )

    governed_dir = bundle_dir / PHASE2_GOVERNED_EVIDENCE_SUBDIR
    for name in ("task_manifest.json", "meta_validity_report.json", "execution_assumptions.json"):
        assert (governed_dir / name).is_file()
    assert not (bundle_dir / "task_manifest.json").exists()

    shell = evaluate_phase2_ml_evidence_shell(governed_dir)
    assert shell.status == Phase2MLEvidenceShellStatus.EVIDENCE_STRUCTURALLY_USABLE

    summary = json.loads(
        (bundle_dir / PHASE2_GOVERNED_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert summary["phase"] == "II-0B"
    assert summary["non_promotable"] is True
    assert summary["current_governed_evidence"] is True
    assert summary["evidence_subdir"] == PHASE2_GOVERNED_EVIDENCE_SUBDIR
    assert summary["root_phase2_triple_review"]["status"] == "absent"
    refs = summary["phase2_ml_evidence_shell"]["evidence"]["threshold_governance"]["references"]
    assert any(ref["state"] == "PROVISIONAL" for ref in refs)


@pytest.mark.determinism("d1")
def test_canonical_orchestration_excludes_stale_root_triple_from_current_governed_evidence(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.phase2_governed import PHASE2_GOVERNED_SUMMARY_FILENAME

    _register_stub(monkeypatch, "phase2_orchestration_stub_stale")

    bundle_dir = tmp_path / "bundle"
    _write_stale_root_triple(bundle_dir)
    run(
        strategy_id="phase2_orchestration_stub_stale",
        ctx=_ctx(tmp_path),
        strategy_kwargs={},
        bundle_dir=bundle_dir,
        source_prices=_source_prices(),
        run_metadata={
            "content_hash": "sha256:" + ("b" * 64),
            "content_hash_expected": "sha256:" + ("b" * 64),
            "download_timestamp": "2026-04-17T16:22:00Z",
        },
        run_id="canonical-run-stale-root",
    )

    summary = json.loads(
        (bundle_dir / PHASE2_GOVERNED_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert summary["current_governed_evidence"] is True
    assert summary["root_phase2_triple_review"]["status"] == "excluded_stale_pre_content_hash"
    assert summary["root_phase2_triple_review"]["excluded_from_current_governed_evidence"] is True


@pytest.mark.determinism("d1")
def test_canonical_run_fails_closed_when_governed_phase2_summary_is_not_current(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed

    _register_stub(monkeypatch, "phase2_orchestration_stub_incomplete")
    monkeypatch.setattr(
        "pysrc.pipeline.orchestrator.emit_governed_phase2_orchestration_evidence",
        lambda **_: {
            "current_governed_evidence": False,
            "phase2_ml_evidence_shell": {
                "status": Phase2MLEvidenceShellStatus.EVIDENCE_INCOMPLETE.value,
                "reason_codes": ["MISSING_ARTIFACTS"],
            },
        },
    )

    with pytest.raises(RuntimeError, match="governed II-0B evidence"):
        run(
            strategy_id="phase2_orchestration_stub_incomplete",
            ctx=_ctx(tmp_path),
            strategy_kwargs={},
            bundle_dir=tmp_path / "bundle",
            source_prices=_source_prices(),
            run_id="canonical-run-incomplete",
        )


@pytest.mark.determinism("d1")
def test_run_orchestration_fails_closed_when_governed_phase2_emission_raises(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed

    class _CapturingEngine:
        def run(self, plan: object, pit_view: object, store: object) -> object:
            return SimpleNamespace(metrics={"sharpe": 0.0})

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10, 10, 10, 10],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "pysrc.pipeline.orchestrator.resolve_engine", lambda _engine_id: _CapturingEngine()
    )
    monkeypatch.setattr(
        "pysrc.pipeline.orchestrator.emit_governed_phase2_orchestration_evidence",
        lambda **_: (_ for _ in ()).throw(RuntimeError("synthetic governed emission failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic governed emission failure"):
        run_orchestration(OrchestratorConfig(input_path=csv_path, bundle_dir=tmp_path / "bundle"))

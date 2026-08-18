from __future__ import annotations

import inspect
import json
from dataclasses import dataclass

import pandas as pd
import pytest

from pysrc.artifact_registry.run_registry import RunRegistry, RunStatus
from pysrc.backtesting.contracts.types import PitMeta
from pysrc.preprocessor.graph.factory import register_builtin_ops
from pysrc.strategies.momentum import entry
from pysrc.strategies.momentum.entry import (
    _coerce_returns,
    _finalize_managed_run,
    _trial_counter_family,
    run,
)
from pysrc.strategies.momentum.exceptions import (
    ConvergenceError,
    CostGateRejection,
    MissingExecutionAssumptionsError,
)
from pysrc.strategies.pipeline_strategy import MaterializationError, StrategyContext, TradeIntent

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _ctx(tmp_path) -> StrategyContext:
    return StrategyContext(
        prices=pd.DataFrame({"close": [100.0 + idx for idx in range(30)]}),
        backend="pandas",
        cache_dir=tmp_path / "bundle",
        pit_provenance=PitMeta(
            as_of="2024-01-04T00:00:00",
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff="2024-01-04",
        ),
    )


class _NoopHooks:
    def apply_crash_override(self, **kwargs):
        return None

    def apply_cost_gate(self, **kwargs):
        return None


@dataclass
class _StubStrategy:
    variant: str = "xsec"
    trade_intent: TradeIntent | None = None
    to_raise: Exception | None = None

    def __post_init__(self) -> None:
        self.params = {"variant": self.variant}

    def generate_trade_intent(self, ctx: StrategyContext) -> TradeIntent:
        if self.to_raise is not None:
            raise self.to_raise
        if self.trade_intent is None:
            raise AssertionError("trade_intent must be provided for stub strategy")
        return self.trade_intent


def _governed_alpha_ir(tmp_path) -> entry.AlphaIR:
    ctx = _ctx(tmp_path)
    strategy = entry.MomentumStrategy()
    strategy._active_ctx = ctx
    try:
        return strategy.generate_signal(
            pd.DataFrame(
                {
                    "returns": [0.01 * ((idx % 5) - 2) for idx in range(20)],
                    "realized_vol_60": [0.2 + (idx * 0.001) for idx in range(20)],
                    "mom_scaled": [0.05 * ((idx % 7) - 3) for idx in range(20)],
                }
            )
        )
    finally:
        strategy._active_ctx = None


def test_run_without_orchestrator_hooks_writes_bundle(tmp_path) -> None:
    register_builtin_ops()
    result = run(_ctx(tmp_path), orchestrator_hooks=None)
    assert result.bundle_dir.exists()
    assert (result.bundle_dir / "plan.json").exists()
    assert (result.bundle_dir / "env_fingerprint.json").exists()
    assert (result.bundle_dir / "dataset_manifest.json").exists()
    assert (result.bundle_dir / "preprocessing_report.json").exists()
    assert (result.bundle_dir / "splits_manifest.json").exists()
    assert (result.bundle_dir / "execution_assumptions.json").exists()
    assert (result.bundle_dir / "signal_card.json").exists()
    assert (result.bundle_dir / "stat_validity_report.json").exists()


def test_run_without_split_override_preserves_none_splits_manifest(tmp_path) -> None:
    register_builtin_ops()
    result = run(_ctx(tmp_path), orchestrator_hooks=None)
    splits_manifest = json.loads(
        (result.bundle_dir / "splits_manifest.json").read_text(encoding="utf-8")
    )
    assert splits_manifest["split_method"] == "none"
    assert splits_manifest["splits"] == []
    assert "cpcv_path_scores.json" not in result.artifacts
    assert not (result.bundle_dir / "cpcv_path_scores.json").exists()


def test_run_with_split_override_writes_provided_splits_manifest(tmp_path) -> None:
    register_builtin_ops()
    splits_override = {
        "split_method": "cpcv",
        "purge_window": 2,
        "embargo_window": 1,
        "splits": [
            {
                "path_id": "path-000",
                "split_index": 0,
                "train_indices": [0, 1, 2],
                "test_indices": [3, 4],
                "test_group_ids": [1, 2],
            }
        ],
    }
    result = run(
        _ctx(tmp_path),
        orchestrator_hooks=None,
        splits_manifest_override=splits_override,
    )
    splits_manifest = json.loads(
        (result.bundle_dir / "splits_manifest.json").read_text(encoding="utf-8")
    )
    assert splits_manifest["split_method"] == "cpcv"
    assert splits_manifest["purge_window"] == 2
    assert splits_manifest["embargo_window"] == 1
    assert splits_manifest["splits"][0]["path_id"] == "path-000"
    assert result.artifacts["cpcv_path_scores.json"] == result.bundle_dir / "cpcv_path_scores.json"
    cpcv_path_scores = json.loads(
        (result.bundle_dir / "cpcv_path_scores.json").read_text(encoding="utf-8")
    )
    assert cpcv_path_scores["schema_version"] == "1.0.0"
    assert cpcv_path_scores["variant"] == "xsec"
    assert cpcv_path_scores["split_surface_hash"].startswith("sha256:")
    assert cpcv_path_scores["shared_cost_hash"].startswith("sha256:")
    assert cpcv_path_scores["evaluations"][0]["trial_id"] == "xsec"
    assert cpcv_path_scores["evaluations"][0]["path_id"] == "path-000"


def test_missing_execution_assumptions_raises_before_cost_gate(tmp_path, monkeypatch) -> None:
    register_builtin_ops()

    def _skip_write(*args, **kwargs):
        return tmp_path / "bundle" / "execution_assumptions.json"

    monkeypatch.setattr(entry, "_write_execution_assumptions", _skip_write)
    with pytest.raises(MissingExecutionAssumptionsError):
        run(_ctx(tmp_path), orchestrator_hooks=_NoopHooks())


def test_cost_gate_rejection_propagates(tmp_path) -> None:
    register_builtin_ops()

    class RejectHooks(_NoopHooks):
        def apply_cost_gate(self, **kwargs):
            raise CostGateRejection(
                "turnover rejected",
                variant="xsec",
                run_id="run-1",
                reason_code="TURNOVER_LIMIT",
            )

    with pytest.raises(CostGateRejection):
        run(_ctx(tmp_path), orchestrator_hooks=RejectHooks())


def test_cost_gate_rejection_writes_canonical_gate_result(tmp_path) -> None:
    register_builtin_ops()
    bundle_dir = tmp_path / "bundle-cost-gate"

    class RejectHooks(_NoopHooks):
        def apply_cost_gate(self, **kwargs):
            raise CostGateRejection(
                "turnover rejected",
                variant="xsec",
                run_id="run-1",
                reason_code="TURNOVER_LIMIT",
            )

    with pytest.raises(CostGateRejection):
        run(_ctx(tmp_path), orchestrator_hooks=RejectHooks(), bundle_dir=bundle_dir)

    gate_result = json.loads((bundle_dir / "gate_result.json").read_text(encoding="utf-8"))
    assert gate_result["overall_result"] == "FAIL"
    assert gate_result["gates"][0]["reason_code"] == "COST_GATE_REJECTED"
    assert gate_result["gates"][0]["evidence"]["upstream_reason_code"] == "TURNOVER_LIMIT"


def test_explicit_crash_override_request_fails_closed(tmp_path) -> None:
    register_builtin_ops()

    with pytest.raises(
        NotImplementedError,
        match="MOM-007: crash trigger awaits dedicated governed source adapter — see OI-34",
    ):
        run(
            _ctx(tmp_path),
            orchestrator_hooks=_NoopHooks(),
            enable_crash_override=True,
        )


def test_convergence_error_clause_precedes_materialization_error_clause() -> None:
    source = inspect.getsource(entry.run)
    assert source.index("except ConvergenceError") < source.index("except MaterializationError")


def test_run_writes_signal_card_and_v1_stat_validity(tmp_path) -> None:
    register_builtin_ops()
    result = run(_ctx(tmp_path), orchestrator_hooks=None)
    signal_card = json.loads((result.bundle_dir / "signal_card.json").read_text(encoding="utf-8"))
    stat_validity = json.loads(
        (result.bundle_dir / "stat_validity_report.json").read_text(encoding="utf-8")
    )
    assert signal_card["schema_version"] == "v1"
    assert stat_validity["schema_version"] == "v1"


def test_run_registry_trial_counter_accumulates_across_instances(tmp_path) -> None:
    register_builtin_ops()
    registry_root = tmp_path / "registry"
    first_registry = RunRegistry(registry_root)
    second_registry = RunRegistry(registry_root)

    run(
        _ctx(tmp_path),
        run_registry=first_registry,
        bundle_dir=tmp_path / "bundle-one",
    )
    run(
        _ctx(tmp_path),
        variant="tsmom",
        run_registry=second_registry,
        bundle_dir=tmp_path / "bundle-two",
    )

    counters = json.loads((registry_root / "trial_counters.json").read_text(encoding="utf-8"))
    assert counters["momentum.phase_i.production_v1"] == 2
    assert second_registry.get_trial_count("momentum.phase_i.production_v1") == 2
    assert len(list(second_registry.iter_runs(status_filter={RunStatus.COMPLETE}))) == 2


def test_coerce_returns_falls_back_when_returns_column_missing() -> None:
    assert _coerce_returns(pd.DataFrame({"signal": [1.0, 2.0]})) == [0.0] * 12


def test_trial_counter_family_uses_variant_specific_family_outside_phase_i_three() -> None:
    assert _trial_counter_family("industry") == "momentum.industry"


def test_finalize_managed_run_noops_for_non_registering_run(tmp_path) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = registry.begin_run()
    registry.finalize_run(run_id, RunStatus.COMPLETE)
    _finalize_managed_run(registry, run_id, status=RunStatus.FAILED)
    record = registry.get_run(run_id, include_failed=True)
    assert record is not None
    assert record.status is RunStatus.COMPLETE


def test_run_requires_alpha_ir_in_trade_intent_raw(tmp_path) -> None:
    register_builtin_ops()
    strategy = _StubStrategy(
        trade_intent=TradeIntent(
            weights=pd.Series([1.0, 0.0]),
            raw={"features": pd.DataFrame({"returns": [0.1, -0.1]})},
        )
    )
    with pytest.raises(TypeError, match="TradeIntent.raw\\['alpha_ir'\\]"):
        run(_ctx(tmp_path), strategy=strategy, orchestrator_hooks=None)


def test_crash_override_hook_can_replace_trade_intent(tmp_path) -> None:
    register_builtin_ops()
    original_intent = TradeIntent(
        weights=pd.Series([1.0, 0.0]),
        raw={},
    )
    replacement_intent = TradeIntent(
        weights=pd.Series([0.0, 1.0]),
        raw={},
    )
    alpha_ir = _governed_alpha_ir(tmp_path)
    original_intent.raw["alpha_ir"] = alpha_ir
    original_intent.raw["features"] = pd.DataFrame({"returns": [0.1, -0.1]})

    class ReplaceCrashHooks(_NoopHooks):
        def apply_crash_override(self, **kwargs):
            return replacement_intent

    result = run(
        _ctx(tmp_path),
        strategy=_StubStrategy(trade_intent=original_intent),
        orchestrator_hooks=ReplaceCrashHooks(),
    )
    assert result.trade_intent is replacement_intent


def test_cost_gate_hook_can_replace_trade_intent(tmp_path) -> None:
    register_builtin_ops()
    original_intent = TradeIntent(
        weights=pd.Series([1.0, 0.0]),
        raw={},
    )
    replacement_intent = TradeIntent(
        weights=pd.Series([0.0, 1.0]),
        raw={},
    )
    alpha_ir = _governed_alpha_ir(tmp_path)
    original_intent.raw["alpha_ir"] = alpha_ir
    original_intent.raw["features"] = pd.DataFrame({"returns": [0.1, -0.1]})

    class ReplaceCostHooks(_NoopHooks):
        def apply_cost_gate(self, **kwargs):
            return replacement_intent

    result = run(
        _ctx(tmp_path),
        strategy=_StubStrategy(trade_intent=original_intent),
        orchestrator_hooks=ReplaceCostHooks(),
    )
    assert result.trade_intent is replacement_intent


def test_convergence_error_from_cost_gate_finalizes_managed_run_failed(tmp_path) -> None:
    register_builtin_ops()
    registry = RunRegistry(tmp_path / "registry")

    class ConvergeHooks(_NoopHooks):
        def apply_cost_gate(self, **kwargs):
            raise ConvergenceError("did not converge")

    with pytest.raises(ConvergenceError):
        run(_ctx(tmp_path), run_registry=registry, orchestrator_hooks=ConvergeHooks())

    records = list(registry.iter_runs(status_filter={RunStatus.FAILED}))
    assert len(records) == 1
    assert records[0].status is RunStatus.FAILED


def test_materialization_error_with_convergence_text_is_normalized(tmp_path) -> None:
    register_builtin_ops()
    registry = RunRegistry(tmp_path / "registry")
    strategy = _StubStrategy(to_raise=MaterializationError("convergence failed in residual step"))

    with pytest.raises(ConvergenceError, match="convergence failed in residual step"):
        run(_ctx(tmp_path), strategy=strategy, run_registry=registry)

    records = list(registry.iter_runs(status_filter={RunStatus.FAILED}))
    assert len(records) == 1


def test_materialization_error_without_convergence_text_is_reraised(tmp_path) -> None:
    register_builtin_ops()
    registry = RunRegistry(tmp_path / "registry")
    strategy = _StubStrategy(to_raise=MaterializationError("other materialization failure"))

    with pytest.raises(MaterializationError, match="other materialization failure"):
        run(_ctx(tmp_path), strategy=strategy, run_registry=registry)

    records = list(registry.iter_runs(status_filter={RunStatus.FAILED}))
    assert len(records) == 1

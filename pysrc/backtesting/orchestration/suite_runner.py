from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pysrc.backtesting.contracts.registry import resolve_engine, resolve_validator
from pysrc.backtesting.contracts.types import RunBundleRef
from pysrc.backtesting.data.pit import PitUnsafeFrame
from pysrc.backtesting.orchestration.plan import BacktestSuitePlan


@dataclass
class BacktestSuiteRunner:
    config: dict[str, Any] = field(default_factory=dict)

    def execute(self, suite_plan: BacktestSuitePlan) -> RunBundleRef:
        store = suite_plan.store
        last_run_id: str | None = None
        for plan in suite_plan.plans:
            engine = resolve_engine(plan.engine_id)
            data = suite_plan.context.get("data")
            if data is None:
                data = PitUnsafeFrame(payload_ref="synthetic", metadata={})
            result = engine.run(plan, data, store)
            for validator_id in plan.validator_ids:
                validator = resolve_validator(validator_id)
                validator.validate(result, dict(suite_plan.context), store)
            last_run_id = plan.run_id
        return RunBundleRef(bundle_path=suite_plan.bundle_path, run_id=last_run_id)

    def run(self, *args: Any, **kwargs: Any) -> RunBundleRef:
        if args and isinstance(args[0], BacktestSuitePlan):
            return self.execute(args[0])
        raise TypeError("BacktestSuiteRunner.run() expects a BacktestSuitePlan instance.")

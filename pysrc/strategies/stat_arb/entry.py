from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import pandas as pd

from pysrc.backtesting.contracts.bundle import RunBundle
from pysrc.backtesting.contracts.types import PitMeta
from pysrc.data.dataview import DataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter
from pysrc.strategies.pipeline_strategy import StrategyContext

from .config import PAIRS_DEFAULT, PairsConfig


@dataclass(frozen=True)
class StatArbRunConfig:
    leg_a: str
    leg_b: str
    start: str
    end: str
    config: PairsConfig = PAIRS_DEFAULT
    bundle_dir: Path | None = None
    prices: pd.DataFrame | None = None


def run_stat_arb_pairs(
    leg_a: str,
    leg_b: str,
    *,
    run_cfg: StatArbRunConfig,
) -> RunBundle:
    """
    Assemble PIT-safe context and delegate to the canonical orchestrator.

    This function must not:
    - resolve or run engines directly
    - write manifest files directly
    - resolve or call validators directly
    - build RunBundle directly

    All of the above are owned by the canonical orchestrator.
    """
    config = run_cfg.config or PAIRS_DEFAULT
    if run_cfg.prices is None or run_cfg.prices.empty:
        raise ValueError("run_stat_arb_pairs requires run_cfg.prices (non-empty DataFrame)")
    prices = run_cfg.prices
    if "symbol" not in prices.columns:
        raise ValueError("run_cfg.prices must have a 'symbol' column")
    if "valid_time" not in prices.columns or "knowledge_time" not in prices.columns:
        raise ValueError("run_cfg.prices must have 'valid_time' and 'knowledge_time' columns")

    dataview = DataView(pit_required=True)
    dataview.register_source(
        prices,
        valid_time_col="valid_time",
        knowledge_time_col="knowledge_time",
    )

    adapter = DataViewAsOfAdapter(
        dataview=dataview,
        symbols=[leg_a, leg_b],
        fields=[f"{leg_a}.close", f"{leg_b}.close"],
    )

    from datetime import datetime

    knowledge_dates = sorted(
        {
            datetime.combine(pd.Timestamp(d).date(), datetime.min.time()).replace(tzinfo=UTC)
            if hasattr(d, "date")
            else datetime.fromisoformat(str(d)).replace(tzinfo=UTC)
            for d in prices["knowledge_time"]
        }
    )
    if not knowledge_dates:
        raise ValueError("run_stat_arb_pairs requires at least one knowledge_time value")

    wide_frame = adapter.as_wide_frame(knowledge_dates)
    if wide_frame.empty:
        raise ValueError("run_stat_arb_pairs could not build wide frame (no PIT snapshots)")

    pit_meta = PitMeta(
        as_of=run_cfg.end,
        source="pysrc.data.dataview.DataView",
        knowledge_cutoff=run_cfg.end,
    )
    ctx = StrategyContext(
        prices=wide_frame,
        backend="pandas",
        pit_provenance=pit_meta,
    )

    from pysrc.pipeline.orchestrator import run as orchestrator_run

    bundle_dir = run_cfg.bundle_dir or Path("bundles") / "stat_arb_pairs"
    return orchestrator_run(
        strategy_id="stat_arb_pairs",
        ctx=ctx,
        strategy_kwargs={"leg_a": leg_a, "leg_b": leg_b, "config": config},
        bundle_dir=bundle_dir,
        source_prices=prices,
        run_metadata={"evaluation_window": f"{run_cfg.start}:{run_cfg.end}"},
    )

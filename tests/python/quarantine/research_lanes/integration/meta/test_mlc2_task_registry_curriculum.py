from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pysrc.meta.bocpd_service import RegimeLabelRecord
from pysrc.meta.curriculum import (
    CurriculumSampler,
    CurriculumSamplerConfig,
    HoldoutExclusionSurface,
)
from pysrc.meta.task import MAX_SIGNALS
from pysrc.meta.task_generator import TaskGeneratorConfig, build_meta_task
from pysrc.meta.task_registry import TaskRegistry
from pysrc.meta_learning.contracts.encoder_contracts import (
    EncoderInputContract,
    EncoderOutputContract,
)


class _Encoder:
    def encode(self, input: EncoderInputContract) -> EncoderOutputContract:
        _ = input
        return EncoderOutputContract(
            regime_embedding=np.ones(64, dtype=np.float32), schema_version="v1"
        )

    def is_frozen(self) -> bool:
        return True


def _episode(offset: int) -> tuple[datetime, ...]:
    start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=offset)
    return tuple(start + timedelta(days=i) for i in range(50))


class _DataView:
    def as_of(
        self, symbols: Sequence[str], fields: Sequence[str], knowledge_date: date
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = [
            {"symbol": symbol, "valid_time": knowledge_date, "knowledge_time": knowledge_date}
            for symbol in symbols
        ]
        for row in rows:
            for field in fields:
                row[field] = 1.0
        return pd.DataFrame(rows)


def _dv(episodes: list[tuple[datetime, ...]]) -> _DataView:
    _ = episodes
    return _DataView()


def _label(ts: datetime, regime_class: str, regime_id: str) -> RegimeLabelRecord:
    return RegimeLabelRecord(
        entity_id="SPY",
        decision_ts=ts,
        regime_id=regime_id,
        regime_label=regime_id,
        effective_at=ts,
        state_snapshot_id="sha256:state",
        input_snapshot_id="sha256:input",
        config_version="test",
        change_probability=0.01,
        boundary_flag="stable",
        regime_class=regime_class,  # type: ignore[arg-type]
        diag_regime_class_bocpd_gated=regime_class,  # type: ignore[arg-type]
        diag_regime_class_extended=regime_class,  # type: ignore[arg-type]
        run_length_mode=10,
        run_length_expectation=10.0,
        transition_probability=0.01,
        posterior_entropy=0.2,
        trend_score_raw=0.01,
        vol_score_raw=0.1,
        cold_start=False,
    )


def _signals() -> tuple[list[str], np.ndarray[Any, np.dtype[np.bool_]]]:
    ids = [""] * MAX_SIGNALS
    ids[0] = "sig_a"
    mask = np.zeros(MAX_SIGNALS, dtype=np.bool_)
    mask[0] = True
    return ids, mask


@pytest.mark.integration
@pytest.mark.determinism("d0")
def test_build_meta_task_registry_roundtrip_and_holdout_exclusion(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episodes = [_episode(i * 60) for i in range(6)]
    dv = _dv(episodes)
    ids, mask = _signals()
    classes = ("bull", "bear", "sideways", "high_vol", "crisis", "crisis")
    regime_ids = (
        "trend_hi__vol_med__bocpd_stable",
        "trend_lo__vol_med__bocpd_stable",
        "trend_flat__vol_lo__bocpd_stable",
        "trend_hi__vol_hi__bocpd_transition",
        "trend_lo__vol_hi__bocpd_cp",
        "trend_hi__vol_hi__bocpd_cp",
    )
    registry = TaskRegistry()
    built = []
    for episode, regime_class, regime_id in zip(episodes, classes, regime_ids, strict=True):
        task = build_meta_task(
            data_view=dv,
            regime_label=_label(episode[19], regime_class, regime_id),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=TaskGeneratorConfig(
                episode_timestamps=episode,
                symbols=("SPY",),
                fields=("close",),
                n_support=20,
                n_query=10,
                purge_window=5,
                embargo_window=2,
            ),
        )
        registry.append(task)
        built.append(task)

    first = built[0]
    assert registry.get_by_stable(regime_id=first.regime_id, t0=first.t0) is first

    holdout = HoldoutExclusionSurface(task_ids=frozenset({built[-1].task_id}))
    sampler = CurriculumSampler(
        list(registry),
        config=CurriculumSamplerConfig(
            batch_size=5,
            bucket_minimums={"bull": 1, "bear": 1, "sideways": 1, "high_vol": 1, "crisis": 0},
            seed=5,
        ),
        holdouts=holdout,
    )
    batch = sampler.sample_bootstrap()
    assert built[-1].task_id not in {task.task_id for task in batch.tasks}

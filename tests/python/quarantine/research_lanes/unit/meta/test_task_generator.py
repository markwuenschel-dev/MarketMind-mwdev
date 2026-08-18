from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pysrc.meta.bocpd_service import RegimeLabelRecord
from pysrc.meta.task import MAX_SIGNALS
from pysrc.meta.task_generator import (
    EpisodeConstructionError,
    TaskGeneratorConfig,
    _validate_geometry,
    build_meta_task,
)
from pysrc.meta_learning.contracts.encoder_contracts import (
    TASK_EPISODE_ENCODER_FEATURE_NAMES,
    EncoderInputContract,
    EncoderOutputContract,
    build_task_episode_encoder_input,
)


class _Encoder:
    def encode(self, input: EncoderInputContract) -> EncoderOutputContract:
        _ = input
        return EncoderOutputContract(
            regime_embedding=np.arange(64, dtype=np.float32),
            schema_version="v1",
        )

    def is_frozen(self) -> bool:
        return True


class _RecordingEncoder:
    def __init__(self) -> None:
        self.inputs: list[EncoderInputContract] = []

    def encode(self, input: EncoderInputContract) -> EncoderOutputContract:
        self.inputs.append(input)
        return EncoderOutputContract(
            regime_embedding=np.arange(64, dtype=np.float32),
            schema_version="v1",
        )

    def is_frozen(self) -> bool:
        return True


def _label(*, ts: datetime) -> RegimeLabelRecord:
    return RegimeLabelRecord(
        entity_id="SPY",
        decision_ts=ts,
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_label="trend_hi__vol_med__bocpd_stable",
        effective_at=ts,
        state_snapshot_id="sha256:state",
        input_snapshot_id="sha256:input",
        config_version="test",
        change_probability=0.01,
        boundary_flag="stable",
        regime_class="bull",
        diag_regime_class_bocpd_gated="bull",
        diag_regime_class_extended="bull",
        run_length_mode=12,
        run_length_expectation=12.0,
        transition_probability=0.02,
        posterior_entropy=0.5,
        trend_score_raw=0.01,
        vol_score_raw=0.2,
        cold_start=False,
    )


def _signals() -> tuple[list[str], np.ndarray[Any, np.dtype[np.bool_]]]:
    ids = [""] * MAX_SIGNALS
    ids[0] = "mom_12_1"
    ids[1] = "quality"
    ids[2] = "low_vol"
    mask = np.zeros(MAX_SIGNALS, dtype=np.bool_)
    mask[:3] = True
    return ids, mask


def _episode(n: int = 50) -> tuple[datetime, ...]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return tuple(start + timedelta(days=i) for i in range(n))


class _DataView:
    def as_of(
        self, symbols: Sequence[str], fields: Sequence[str], knowledge_date: date
    ) -> pd.DataFrame:
        rows = [
            {"symbol": symbol, "valid_time": knowledge_date, "knowledge_time": knowledge_date}
            for symbol in symbols
        ]
        for row in rows:
            for field in fields:
                row[field] = 1.0
        return pd.DataFrame(rows)


class _FutureDataView:
    def as_of(
        self, symbols: Sequence[str], fields: Sequence[str], knowledge_date: date
    ) -> pd.DataFrame:
        _ = symbols, fields
        future = knowledge_date + timedelta(days=1)
        return pd.DataFrame([{"valid_time": future, "knowledge_time": knowledge_date}])


class _NonFrameDataView:
    def as_of(self, symbols: Sequence[str], fields: Sequence[str], knowledge_date: date) -> str:
        _ = symbols, fields, knowledge_date
        return "not-a-frame"


class _BadShapeEncoder:
    def encode(self, input: EncoderInputContract) -> EncoderOutputContract:
        _ = input
        return EncoderOutputContract(
            regime_embedding=np.ones((1, 64), dtype=np.float32),
            schema_version="v1",
        )

    def is_frozen(self) -> bool:
        return True


def _dataview(timestamps: tuple[datetime, ...]) -> _DataView:
    _ = timestamps
    return _DataView()


def _config(
    *,
    episode_timestamps: tuple[datetime, ...] | None = None,
    encoder_input: EncoderInputContract | None = None,
) -> TaskGeneratorConfig:
    return TaskGeneratorConfig(
        episode_timestamps=episode_timestamps or _episode(),
        symbols=("SPY",),
        fields=("close",),
        n_support=20,
        n_query=10,
        purge_window=5,
        embargo_window=2,
        encoder_input=encoder_input,
    )


@pytest.mark.determinism("d0")
def test_valid_synthetic_episode_produces_meta_task(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    task = build_meta_task(
        data_view=_dataview(episode),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=_Encoder(),
        horizon=5,
        config=_config(episode_timestamps=episode),
    )

    assert task.regime_id == "trend_hi__vol_med__bocpd_stable"
    assert task.regime_class == "bull"
    assert task.support_set[-1] == task.pit_boundary
    assert len(task.support_set) == 20
    assert len(task.query_set) == 10
    assert task.regime_embedding is not None
    assert task.regime_embedding.shape == (64,)


@pytest.mark.determinism("d0")
def test_support_and_query_are_strictly_disjoint(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    task = build_meta_task(
        data_view=_dataview(episode),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=_Encoder(),
        horizon=5,
        config=_config(episode_timestamps=episode),
    )
    assert set(task.support_set).isdisjoint(set(task.query_set))


@pytest.mark.determinism("d0")
def test_purge_embargo_inequality_is_enforced(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    cfg = _config(episode_timestamps=episode)
    cfg = replace(cfg, purge_window=5, embargo_window=3, n_support=20, n_query=10)
    bad_geometry = replace(cfg, bar_interval=pd.Timedelta(days=2))

    with pytest.raises(EpisodeConstructionError, match="purge_window"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=bad_geometry,
        )


@pytest.mark.determinism("d0")
def test_short_episode_raises_typed_exception(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode(25)
    ids, mask = _signals()
    with pytest.raises(EpisodeConstructionError, match="minimum feasible"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )


@pytest.mark.determinism("d0")
def test_signal_identity_and_mask_are_preserved(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    task = build_meta_task(
        data_view=_dataview(episode),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=_Encoder(),
        horizon=5,
        config=_config(episode_timestamps=episode),
    )
    assert task.signal_ids == tuple(ids)
    assert task.signal_mask == tuple(bool(x) for x in mask)
    assert len(task.signal_mask) == 64
    assert task.active_k == 3


@pytest.mark.determinism("d0")
def test_deterministic_task_id_for_identical_inputs(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    dv = _dataview(episode)
    label = _label(ts=episode[19])
    config = _config(episode_timestamps=episode)
    a = build_meta_task(
        data_view=dv,
        regime_label=label,
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=_Encoder(),
        horizon=5,
        config=config,
    )
    b = build_meta_task(
        data_view=dv,
        regime_label=label,
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=_Encoder(),
        horizon=5,
        config=config,
    )
    assert a.task_id == b.task_id
    assert a.signal_ids_hash == b.signal_ids_hash


@pytest.mark.determinism("d0")
def test_task_episode_encoder_lowering_is_deterministic(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    label = _label(ts=episode[19])
    a = build_task_episode_encoder_input(
        regime_label=label,
        pit_boundary=episode[19],
        signal_set_version=1,
    )
    b = build_task_episode_encoder_input(
        regime_label=label,
        pit_boundary=episode[19],
        signal_set_version=1,
    )
    expected = np.asarray(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, 12.0, 12.0, 0.02, 0.5, 0.01, 0.2],
        dtype=np.float32,
    )
    assert TASK_EPISODE_ENCODER_FEATURE_NAMES[0] == "regime_class_bull"
    assert len(TASK_EPISODE_ENCODER_FEATURE_NAMES) == 16
    assert a.pit_boundary == episode[19]
    assert a.signal_set_version == 1
    assert a.regime_features.dtype == np.float32
    assert np.array_equal(a.regime_features, b.regime_features)
    assert np.array_equal(a.regime_features, expected)


@pytest.mark.determinism("d0")
def test_regime_embedding_populated_through_governed_lowering(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    encoder = _RecordingEncoder()
    task = build_meta_task(
        data_view=_dataview(episode),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=encoder,
        horizon=5,
        config=_config(episode_timestamps=episode),
    )
    assert task.regime_embedding is not None
    assert task.regime_embedding.dtype == np.float32
    assert len(encoder.inputs) == 1
    assert encoder.inputs[0].pit_boundary == episode[19]
    assert encoder.inputs[0].signal_set_version == 1


@pytest.mark.determinism("d0")
def test_task_episode_encoder_lowering_is_query_blind(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    shifted_query_tail = tuple(
        list(episode[:27])
        + [episode[26] + timedelta(days=100 + i) for i in range(len(episode) - 27)]
    )
    ids, mask = _signals()
    encoder_a = _RecordingEncoder()
    encoder_b = _RecordingEncoder()
    task_a = build_meta_task(
        data_view=_dataview(episode),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=encoder_a,
        horizon=5,
        config=_config(episode_timestamps=episode),
    )
    task_b = build_meta_task(
        data_view=_dataview(shifted_query_tail),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=encoder_b,
        horizon=5,
        config=_config(episode_timestamps=shifted_query_tail),
    )
    assert task_a.query_set != task_b.query_set
    assert np.array_equal(encoder_a.inputs[0].regime_features, encoder_b.inputs[0].regime_features)


@pytest.mark.determinism("d0")
def test_rejected_episode_raises_typed_exception_not_none(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    mask = np.zeros(MAX_SIGNALS, dtype=np.bool_)
    with pytest.raises(EpisodeConstructionError):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )


@pytest.mark.determinism("d0")
def test_episode_timestamps_must_be_non_empty_and_sorted(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ids, mask = _signals()
    with pytest.raises(EpisodeConstructionError, match="non-empty"):
        build_meta_task(
            data_view=_dataview(()),
            regime_label=_label(ts=datetime(2024, 1, 1, tzinfo=UTC)),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=replace(_config(), episode_timestamps=()),
        )

    episode = tuple(reversed(_episode()))
    with pytest.raises(EpisodeConstructionError, match="strictly increasing"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[-1]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )


@pytest.mark.determinism("d0")
def test_sizing_config_rejections_are_typed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()

    long_episode = _episode(60)
    cases = (
        (episode, 0, _config(episode_timestamps=episode), "horizon"),
        (episode, 5, replace(_config(episode_timestamps=episode), n_support=0), "positive"),
        (episode, 5, replace(_config(episode_timestamps=episode), purge_window=-1), "non-negative"),
        (
            long_episode,
            5,
            replace(_config(episode_timestamps=long_episode), purge_window=30, embargo_window=15),
            "configured purge/embargo",
        ),
    )
    for case_episode, horizon, config, message in cases:
        with pytest.raises(EpisodeConstructionError, match=message):
            build_meta_task(
                data_view=_dataview(case_episode),
                regime_label=_label(ts=case_episode[19]),
                signal_ids=ids,
                signal_mask=mask,
                signal_set_version=1,
                encoder=_Encoder(),
                horizon=horizon,
                config=config,
            )


@pytest.mark.determinism("d0")
def test_intraday_defaults_are_config_driven(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(80))
    ids, mask = _signals()
    task = build_meta_task(
        data_view=_dataview(episode),
        regime_label=_label(ts=episode[19]),
        signal_ids=ids,
        signal_mask=mask,
        signal_set_version=1,
        encoder=_Encoder(),
        horizon=5,
        config=replace(
            _config(episode_timestamps=episode),
            purge_window=None,
            embargo_window=None,
            frequency="intraday",
            bar_interval=pd.Timedelta(minutes=1),
        ),
    )
    assert task.query_set[0] == pd.Timestamp(episode[29]).isoformat()


@pytest.mark.determinism("d0")
def test_signal_surface_rejects_invalid_width_dtype_and_slot_contract(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()

    with pytest.raises(EpisodeConstructionError, match="fixed 64-slot"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids[:-1],
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )

    with pytest.raises(EpisodeConstructionError, match="boolean"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=cast(Any, np.ones(MAX_SIGNALS, dtype=np.int64)),
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )

    bad_ids = list(ids)
    bad_ids[3] = "inactive_signal"
    with pytest.raises(EpisodeConstructionError):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=bad_ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )


@pytest.mark.determinism("d0")
def test_pit_front_door_rejections_are_typed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()

    cases = (
        (cast(Any, object()), _config(episode_timestamps=episode), "as_of"),
        (
            _dataview(episode),
            replace(_config(episode_timestamps=episode), symbols=()),
            "PIT access",
        ),
        (cast(Any, _NonFrameDataView()), _config(episode_timestamps=episode), "pandas DataFrame"),
        (_FutureDataView(), _config(episode_timestamps=episode), "beyond pit_boundary"),
    )
    for data_view, config, message in cases:
        with pytest.raises(EpisodeConstructionError, match=message):
            build_meta_task(
                data_view=data_view,
                regime_label=_label(ts=episode[19]),
                signal_ids=ids,
                signal_mask=mask,
                signal_set_version=1,
                encoder=_Encoder(),
                horizon=5,
                config=config,
            )


@pytest.mark.determinism("d0")
def test_regime_label_rejections_are_typed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()

    with pytest.raises(EpisodeConstructionError, match="RegimeLabelRecord"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=cast(Any, object()),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )

    with pytest.raises(EpisodeConstructionError, match="pit_boundary"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[21]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )

    with pytest.raises(EpisodeConstructionError):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=replace(_label(ts=episode[19]), regime_class=cast(Any, "not_a_bucket")),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )


@pytest.mark.determinism("d0")
def test_geometry_rejects_overlapping_support_and_query(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ts = pd.Timestamp(datetime(2024, 1, 1, tzinfo=UTC))
    with pytest.raises(EpisodeConstructionError, match="disjoint"):
        _validate_geometry(
            support=(ts,),
            query=(ts,),
            purge_window=0,
            embargo_window=0,
            bar_interval=pd.Timedelta(days=1),
        )


@pytest.mark.determinism("d0")
def test_encoder_contract_rejections_are_typed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episode = _episode()
    ids, mask = _signals()
    base_input = EncoderInputContract(
        regime_features=np.ones(16, dtype=np.float32),
        pit_boundary=episode[19],
        signal_set_version=1,
        schema_version="v1",
    )

    with pytest.raises(EpisodeConstructionError, match="pit_boundary"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(
                episode_timestamps=episode,
                encoder_input=replace(base_input, pit_boundary=episode[18]),
            ),
        )

    with pytest.raises(EpisodeConstructionError, match="signal_set_version"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(
                episode_timestamps=episode, encoder_input=replace(base_input, signal_set_version=2)
            ),
        )

    with pytest.raises(EpisodeConstructionError, match="canonical task episode"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_Encoder(),
            horizon=5,
            config=_config(episode_timestamps=episode, encoder_input=base_input),
        )

    with pytest.raises(EpisodeConstructionError, match="1-D"):
        build_meta_task(
            data_view=_dataview(episode),
            regime_label=_label(ts=episode[19]),
            signal_ids=ids,
            signal_mask=mask,
            signal_set_version=1,
            encoder=_BadShapeEncoder(),
            horizon=5,
            config=_config(episode_timestamps=episode),
        )

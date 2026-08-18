"""MLC-2 governed MetaTask builder and episode construction helpers.

Canonical path resolution
-------------------------

The governing documents name ``py/meta/task_generator.py``.  This repo's
Python root is ``pysrc/``; therefore the governed in-repo path is
``pysrc/meta/task_generator.py``.

This module owns construction-time admissibility only.  It slices candidate
episodes, enforces PIT and leakage geometry, computes derived identity fields,
and calls :class:`pysrc.meta.task.MetaTask` exactly once for valid episodes.
Curriculum, replay priority, and holdout sampling policy live in
``pysrc/meta/curriculum.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import ceil
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.bocpd_service import RegimeLabelRecord
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta_learning.contracts.encoder_contracts import (
    ContextEncoderProtocol,
    EncoderInputContract,
    build_task_episode_encoder_input,
)
from pysrc.meta_learning.contracts.encoder_contracts import (
    RegimeLabelRecord as ContractRegimeLabelRecord,
)
from pysrc.meta_learning.dynamic_k_contract import (
    validate_fixed_slot_task_surface,
    validate_signal_set_version,
)
from pysrc.meta_learning.regime_vocabulary import (
    validate_meta_task_regime_id,
    validate_regime_class,
)
from pysrc.meta_learning.task_generator import compute_task_id, derive_signal_ids_hash
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

Frequency = Literal["daily", "intraday"]


class DataViewLike(Protocol):
    def as_of(
        self,
        symbols: Sequence[str],
        fields: Sequence[str],
        knowledge_date: date,
    ) -> pd.DataFrame: ...


class EpisodeConstructionError(DataPreconditionError):
    """Raised when a candidate episode cannot become a governed MetaTask."""


@dataclass(frozen=True, slots=True)
class TaskGeneratorConfig:
    """Typed construction config for governed task generation.

    The sizing defaults remain validation-gated operating defaults, not frozen
    empirical policy.  Callers must pass the candidate episode timestamps; this
    module does not discover raw data outside the PIT ``DataView.as_of`` front
    door.
    """

    episode_timestamps: Sequence[datetime | date | str | pd.Timestamp]
    symbols: tuple[str, ...]
    fields: tuple[str, ...]
    n_support: int = 20
    n_query: int = 10
    purge_window: int | None = None
    embargo_window: int | None = None
    frequency: Frequency = "daily"
    bar_interval: pd.Timedelta = pd.Timedelta(days=1)
    # Compatibility verifier only: supplied c_t must equal canonical OI-60 lowering.
    encoder_input: EncoderInputContract | None = None


def _raise(message: str, details: dict[str, Any] | None = None) -> None:
    raise EpisodeConstructionError(message, details=details)


def _normalize_timestamp(value: datetime | date | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts


def _iso_utc(ts: pd.Timestamp) -> str:
    return ts.isoformat()


def _normalize_episode(
    timestamps: Sequence[datetime | date | str | pd.Timestamp],
) -> tuple[pd.Timestamp, ...]:
    if not timestamps:
        _raise("episode_timestamps must be non-empty", {"n": 0})
    out = tuple(_normalize_timestamp(x) for x in timestamps)
    if any(a >= b for a, b in zip(out, out[1:], strict=False)):
        _raise("episode_timestamps must be strictly increasing", {"n": len(out)})
    return out


def _purge_window(config: TaskGeneratorConfig, horizon: int) -> int:
    if config.purge_window is not None:
        return int(config.purge_window)
    return max(int(horizon), 5)


def _embargo_window(config: TaskGeneratorConfig, episode_len: int) -> int:
    if config.embargo_window is not None:
        return int(config.embargo_window)
    if config.frequency == "daily":
        return 2
    return int(ceil(0.05 * float(episode_len)))


def _validate_sizing(
    config: TaskGeneratorConfig, horizon: int, episode_len: int
) -> tuple[int, int]:
    if horizon < 1:
        _raise("horizon must be >= 1", {"horizon": horizon})
    if config.n_support < 1 or config.n_query < 1:
        _raise(
            "n_support and n_query must be positive",
            {"n_support": config.n_support, "n_query": config.n_query},
        )
    purge = _purge_window(config, horizon)
    embargo = _embargo_window(config, episode_len)
    if purge < 0 or embargo < 0:
        _raise(
            "purge_window and embargo_window must be non-negative",
            {"purge": purge, "embargo": embargo},
        )
    minimum = int(config.n_support + config.n_query + 2 * horizon + embargo)
    if episode_len < minimum:
        _raise(
            "episode length violates minimum feasible length",
            {"episode_len": episode_len, "minimum feasible": minimum},
        )
    needed_for_slice = int(config.n_support + purge + embargo + config.n_query)
    if episode_len < needed_for_slice:
        _raise(
            "episode length cannot satisfy configured purge/embargo slice",
            {"episode_len": episode_len, "needed": needed_for_slice},
        )
    return purge, embargo


def _slice_episode(
    episode: tuple[pd.Timestamp, ...],
    *,
    n_support: int,
    n_query: int,
    purge_window: int,
    embargo_window: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    support = episode[:n_support]
    q_start = n_support + purge_window + embargo_window
    query = episode[q_start : q_start + n_query]
    if len(support) != n_support or len(query) != n_query:
        _raise(
            "episode slicing did not produce configured support/query sizes",
            {"n_support": len(support), "n_query": len(query)},
        )
    return support, query


def _validate_signal_surface(
    signal_ids: Sequence[str],
    signal_mask: NDArray[np.bool_] | Sequence[bool],
) -> tuple[tuple[str, ...], tuple[bool, ...], int, str]:
    if len(signal_ids) != MAX_SIGNALS or len(signal_mask) != MAX_SIGNALS:
        _raise(
            "signal_ids and signal_mask must be fixed 64-slot surfaces",
            {"n_ids": len(signal_ids), "n_mask": len(signal_mask), "MAX_SIGNALS": MAX_SIGNALS},
        )
    ids_t = tuple(str(x) for x in signal_ids)
    mask_arr = np.asarray(signal_mask)
    if mask_arr.shape != (MAX_SIGNALS,):
        _raise(
            "signal_mask must be one-dimensional with width 64", {"shape": tuple(mask_arr.shape)}
        )
    if mask_arr.dtype != np.bool_:
        _raise("signal_mask must be boolean", {"dtype": str(mask_arr.dtype)})
    mask_t = tuple(bool(x) for x in mask_arr.tolist())
    active_k = int(sum(1 for x in mask_t if x))
    if active_k < 1:
        _raise("active_k requires at least one True entry in signal_mask", {"active_k": active_k})
    try:
        validate_fixed_slot_task_surface(signal_ids=ids_t, signal_mask=mask_t, active_k=active_k)
    except DataPreconditionError as exc:
        raise EpisodeConstructionError(exc.msg, details=exc.details) from exc
    sig_hash = derive_signal_ids_hash(signal_ids=ids_t, signal_mask=mask_t)
    return ids_t, mask_t, active_k, sig_hash


def _validate_pit_snapshot(snapshot: pd.DataFrame, pit_boundary: pd.Timestamp) -> None:
    if not isinstance(snapshot, pd.DataFrame):
        _raise("DataView.as_of must return a pandas DataFrame", {"type": type(snapshot).__name__})
    pit_date = pit_boundary.date()
    for col in ("valid_time", "knowledge_time"):
        if col not in snapshot.columns:
            continue
        values = pd.to_datetime(snapshot[col]).dt.date
        if bool((values > pit_date).any()):
            _raise(
                "DataView.as_of returned data beyond pit_boundary",
                {"column": col, "pit_boundary": pit_boundary.isoformat()},
            )


def _touch_pit_front_door(
    data_view: DataViewLike,
    *,
    symbols: tuple[str, ...],
    fields: tuple[str, ...],
    pit_boundary: pd.Timestamp,
) -> None:
    if not symbols or not fields:
        _raise("TaskGeneratorConfig.symbols and fields are required for PIT access", {})
    if not hasattr(data_view, "as_of"):
        _raise("data_view must expose DataView.as_of", {"type": type(data_view).__name__})
    snapshot = data_view.as_of(symbols, fields, pit_boundary.date())
    _validate_pit_snapshot(snapshot, pit_boundary)


def _validate_regime_label(
    regime_label: RegimeLabelRecord, pit_boundary: pd.Timestamp
) -> tuple[str, str]:
    if not isinstance(regime_label, RegimeLabelRecord):
        _raise("regime_label must be RegimeLabelRecord", {"type": type(regime_label).__name__})
    effective = _normalize_timestamp(regime_label.effective_at)
    decision = _normalize_timestamp(regime_label.decision_ts)
    if effective > pit_boundary or decision > pit_boundary:
        _raise(
            "regime label must be available no later than pit_boundary",
            {
                "effective_at": effective.isoformat(),
                "decision_ts": decision.isoformat(),
                "pit_boundary": pit_boundary.isoformat(),
            },
        )
    try:
        rid = validate_meta_task_regime_id(regime_label.regime_id)
        rcls = validate_regime_class(regime_label.regime_class)
    except DataPreconditionError as exc:
        raise EpisodeConstructionError(exc.msg, details=exc.details) from exc
    return rid, rcls


def _validate_geometry(
    *,
    support: tuple[pd.Timestamp, ...],
    query: tuple[pd.Timestamp, ...],
    purge_window: int,
    embargo_window: int,
    bar_interval: pd.Timedelta,
) -> None:
    support_max = support[-1]
    query_min = query[0]
    if set(support) & set(query):
        _raise("support and query timestamps must be disjoint", {})
    required_gap = bar_interval * (purge_window + embargo_window)
    if not (support_max + required_gap < query_min):
        _raise(
            "support_set.index.max + purge_window + embargo_window must be < query_set.index.min",
            {
                "support_max": support_max.isoformat(),
                "query_min": query_min.isoformat(),
                "purge_window": purge_window,
                "embargo_window": embargo_window,
            },
        )


def _regime_embedding(
    *,
    encoder: ContextEncoderProtocol,
    regime_label: RegimeLabelRecord,
    supplied_encoder_input: EncoderInputContract | None,
    pit_boundary: pd.Timestamp,
    signal_set_version: int,
) -> NDArray[np.floating[Any]]:
    try:
        canonical_input = build_task_episode_encoder_input(
            regime_label=cast(ContractRegimeLabelRecord, regime_label),
            pit_boundary=pit_boundary.to_pydatetime(),
            signal_set_version=signal_set_version,
        )
    except DataPreconditionError as exc:
        raise EpisodeConstructionError(exc.msg, details=exc.details) from exc

    if supplied_encoder_input is not None:
        _validate_supplied_encoder_input_matches_canonical(
            supplied_encoder_input=supplied_encoder_input,
            canonical_input=canonical_input,
            pit_boundary=pit_boundary,
            signal_set_version=signal_set_version,
        )
    out = encoder.encode(canonical_input)
    emb = np.asarray(out.regime_embedding, dtype=np.float32)
    if emb.ndim != 1:
        _raise("ContextEncoder.encode must return a 1-D regime_embedding", {"ndim": int(emb.ndim)})
    return np.ascontiguousarray(emb, dtype=np.float32)


def _validate_supplied_encoder_input_matches_canonical(
    *,
    supplied_encoder_input: EncoderInputContract,
    canonical_input: EncoderInputContract,
    pit_boundary: pd.Timestamp,
    signal_set_version: int,
) -> None:
    if _normalize_timestamp(supplied_encoder_input.pit_boundary) != pit_boundary:
        _raise(
            "EncoderInputContract.pit_boundary must equal task pit_boundary",
            {
                "encoder_pit_boundary": _normalize_timestamp(
                    supplied_encoder_input.pit_boundary
                ).isoformat(),
                "task_pit_boundary": pit_boundary.isoformat(),
            },
        )
    if int(supplied_encoder_input.signal_set_version) != int(signal_set_version):
        _raise(
            "EncoderInputContract.signal_set_version must match task signal_set_version",
            {
                "encoder_signal_set_version": supplied_encoder_input.signal_set_version,
                "task_signal_set_version": signal_set_version,
            },
        )
    if supplied_encoder_input.schema_version != canonical_input.schema_version:
        _raise(
            "EncoderInputContract.schema_version must match canonical task episode c_t lowering",
            {"schema_version": supplied_encoder_input.schema_version},
        )
    supplied_features = np.asarray(supplied_encoder_input.regime_features, dtype=np.float32)
    canonical_features = np.asarray(canonical_input.regime_features, dtype=np.float32)
    if supplied_features.shape != canonical_features.shape or not bool(
        np.array_equal(supplied_features, canonical_features)
    ):
        _raise(
            "TaskGeneratorConfig.encoder_input must match canonical task episode c_t lowering",
            {
                "expected_shape": tuple(canonical_features.shape),
                "actual_shape": tuple(supplied_features.shape),
            },
        )


def build_meta_task(
    data_view: DataViewLike,
    regime_label: RegimeLabelRecord,
    signal_ids: list[str],
    signal_mask: NDArray[np.bool_],
    signal_set_version: int,
    encoder: ContextEncoderProtocol,
    horizon: int,
    config: TaskGeneratorConfig,
) -> MetaTask:
    """Build one governed :class:`MetaTask` or raise :class:`EpisodeConstructionError`.

    Holdout exclusion is intentionally absent here.  The authoritative training
    exclusion surface lives in ``pysrc.meta.curriculum`` so the generator does
    not duplicate Anti-Goodhart policy.
    """
    episode = _normalize_episode(config.episode_timestamps)
    purge, embargo = _validate_sizing(config, horizon, len(episode))
    support, query = _slice_episode(
        episode,
        n_support=config.n_support,
        n_query=config.n_query,
        purge_window=purge,
        embargo_window=embargo,
    )
    pit_boundary = support[-1]
    _validate_geometry(
        support=support,
        query=query,
        purge_window=purge,
        embargo_window=embargo,
        bar_interval=config.bar_interval,
    )
    _touch_pit_front_door(
        data_view,
        symbols=config.symbols,
        fields=config.fields,
        pit_boundary=pit_boundary,
    )
    rid, rcls = _validate_regime_label(regime_label, pit_boundary)
    ids_t, mask_t, active_k, sig_hash = _validate_signal_surface(signal_ids, signal_mask)
    signal_version = validate_signal_set_version(str(signal_set_version))
    emb = _regime_embedding(
        encoder=encoder,
        regime_label=regime_label,
        supplied_encoder_input=config.encoder_input,
        pit_boundary=pit_boundary,
        signal_set_version=signal_set_version,
    )

    t0 = episode[0]
    t1 = episode[-1]
    task_id = compute_task_id(
        regime_id=rid,
        t0=_iso_utc(t0),
        t1=_iso_utc(t1),
        signal_ids_hash=sig_hash,
    )

    try:
        task = MetaTask(
            task_id=task_id,
            regime_id=rid,
            regime_class=rcls,
            t0=_iso_utc(t0),
            t1=_iso_utc(t1),
            pit_boundary=_iso_utc(pit_boundary),
            support_set=tuple(_iso_utc(x) for x in support),
            query_set=tuple(_iso_utc(x) for x in query),
            signal_ids=ids_t,
            signal_mask=mask_t,
            signal_set_version=signal_version,
            signal_ids_hash=sig_hash,
            horizon=int(horizon),
            active_k=active_k,
            regime_embedding=emb,
        )
    except DataPreconditionError as exc:
        raise EpisodeConstructionError(exc.msg, details=exc.details) from exc

    LOG.info(
        "meta_task_built",
        task_id=task.task_id,
        regime_id=task.regime_id,
        regime_class=task.regime_class,
        t0=task.t0,
        t1=task.t1,
        pit_boundary=task.pit_boundary,
        active_k=task.active_k,
    )
    return task


__all__ = [
    "DataViewLike",
    "EpisodeConstructionError",
    "TaskGeneratorConfig",
    "build_meta_task",
]

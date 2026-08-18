"""MLN-01 canonical MetaTask construction — single permitted constructor for governed tasks.

A **task** is a regime episode: support/query temporal geometry, signal identity, and PIT boundary
are validated here. Do not instantiate :class:`MetaTask` except via :func:`build_meta_task`.

MLN-02: ``regime_id`` is primary identity (token validation); ``regime_class`` is validated against
the five canonical Level-2 labels via :mod:`pysrc.meta_learning.regime_vocabulary`.

HMAC key contract (frozen for MLN-01 closure)
--------------------------------------------
``task_id = HMAC-SHA256(key, regime_id || t0 || t1 || signal_ids_hash)`` with **SHA-256**.

- **Key version:** :data:`TASK_ID_HMAC_KEY_VERSION`
- **Key material:** :data:`TASK_ID_HMAC_KEY_MATERIAL` is the empty byte string ``b""`` (RG09-V13).

Any change to key material or message format **invalidates all historical** ``task_id`` values
unless a governed migration policy exists. This is intentional: identity is content-addressed,
not secret-dependent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import pandas as pd

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.dynamic_k_contract import (
    CONTRACT_VERSION as DYNAMIC_K_CONTRACT_VERSION,
)
from pysrc.meta_learning.dynamic_k_contract import (
    MAX_SIGNALS,
    validate_fixed_slot_task_surface,
    validate_signal_set_version,
)
from pysrc.meta_learning.regime_vocabulary import (
    validate_meta_task_regime_id,
    validate_regime_class,
)
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

TASK_ID_HMAC_KEY_VERSION: Final[str] = "mln01.v1.empty_key"
TASK_ID_HMAC_KEY_MATERIAL: Final[bytes] = b""


def compute_task_id(
    *,
    regime_id: str,
    t0: str,
    t1: str,
    signal_ids_hash: str,
) -> str:
    """Deterministic task identity; ``signal_ids_hash`` is part of the HMAC message (required)."""
    if not str(signal_ids_hash).strip():
        raise DataPreconditionError(
            "signal_ids_hash is required inside task_id identity (MLN-01)",
            details={"regime_id": regime_id},
        )
    msg = f"{regime_id}{t0}{t1}{signal_ids_hash}".encode()
    return hmac.new(TASK_ID_HMAC_KEY_MATERIAL, msg, hashlib.sha256).hexdigest()


def derive_signal_ids_hash(*, signal_ids: tuple[str, ...], signal_mask: tuple[bool, ...]) -> str:
    """Canonical binding of ordered signal ids + mask (MLN-04: positional slot order, length 64)."""
    payload = json.dumps(
        {"ids": list(signal_ids), "mask": list(signal_mask)},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_ts(value: pd.Timestamp | str) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    return pd.Timestamp(value, tz="UTC")


def _iso_utc(ts: pd.Timestamp) -> str:
    return ts.isoformat()


@dataclass(frozen=True, slots=True)
class MetaTask:
    """Canonical regime-episode MetaTask. Construct only through :func:`build_meta_task`."""

    task_id: str
    regime_id: str
    regime_class: str
    regime_embedding: tuple[float, ...] | None
    support_set: tuple[str, ...]
    query_set: tuple[str, ...]
    horizon: int
    signal_ids: tuple[str, ...]
    signal_mask: tuple[bool, ...]
    signal_set_version: str
    signal_ids_hash: str
    pit_boundary: str
    t0: str
    t1: str
    active_k: int


def build_meta_task(
    *,
    regime_id: str,
    regime_class: str,
    regime_embedding: Sequence[float] | None,
    support_set: Sequence[pd.Timestamp | str],
    query_set: Sequence[pd.Timestamp | str],
    horizon: int,
    signal_ids: Sequence[str],
    signal_mask: Sequence[bool],
    signal_set_version: str,
    t0: str,
    t1: str,
    purge_window: pd.Timedelta,
    embargo_window: pd.Timedelta,
    pit_boundary: str | None = None,
) -> MetaTask:
    """
    Canonical builder: enforces disjoint support/query, purge/embargo sizing, and PIT boundary.

    Architecture Vision (support sizing): support_max + purge_window + embargo_window < query_min.
    """
    if horizon < 1:
        raise DataPreconditionError("horizon must be >= 1", details={"horizon": horizon})
    if len(signal_ids) != len(signal_mask):
        raise DataPreconditionError(
            "signal_mask length must match signal_ids",
            details={"n_ids": len(signal_ids), "n_mask": len(signal_mask)},
        )
    if len(signal_ids) != MAX_SIGNALS or len(signal_mask) != MAX_SIGNALS:
        raise DataPreconditionError(
            "MetaTask requires MLN-04 Dynamic-K fixed-slot surface: "
            f"len(signal_ids)==len(signal_mask)=={MAX_SIGNALS}",
            details={"n_ids": len(signal_ids), "n_mask": len(signal_mask)},
        )

    rid = validate_meta_task_regime_id(regime_id)
    rcls = validate_regime_class(regime_class)

    vss = validate_signal_set_version(signal_set_version)
    sig_ids_t = tuple(str(signal_ids[i]) for i in range(MAX_SIGNALS))
    mask_t = tuple(bool(signal_mask[i]) for i in range(MAX_SIGNALS))
    active_k = int(sum(1 for m in mask_t if m))
    if active_k < 1:
        raise DataPreconditionError(
            "active_k requires at least one True entry in signal_mask",
            details={"signal_mask": list(mask_t)},
        )
    validate_fixed_slot_task_surface(signal_ids=sig_ids_t, signal_mask=mask_t, active_k=active_k)

    sig_hash = derive_signal_ids_hash(signal_ids=sig_ids_t, signal_mask=mask_t)

    sup_norm = sorted({_normalize_ts(x) for x in support_set})
    qry_norm = sorted({_normalize_ts(x) for x in query_set})
    if not sup_norm or not qry_norm:
        raise DataPreconditionError(
            "support_set and query_set must each contain at least one timestamp",
            details={"n_support": len(sup_norm), "n_query": len(qry_norm)},
        )

    sup_set_iso = tuple(_iso_utc(x) for x in sup_norm)
    qry_set_iso = tuple(_iso_utc(x) for x in qry_norm)

    sup_pd = set(sup_norm)
    qry_pd = set(qry_norm)
    overlap = sup_pd & qry_pd
    if overlap:
        LOG.error("meta_task_support_query_overlap", overlap_count=len(overlap))
        raise DataPreconditionError(
            "support_set and query_set must be temporally disjoint",
            details={"overlap_sample": _iso_utc(min(overlap))},
        )

    max_sup = max(sup_pd)
    min_q = min(qry_pd)
    boundary = max_sup if pit_boundary is None else _normalize_ts(pit_boundary)
    if boundary != max_sup:
        LOG.error(
            "meta_task_pit_boundary_mismatch",
            pit_boundary=str(pit_boundary),
            support_max=_iso_utc(max_sup),
        )
        raise DataPreconditionError(
            "pit_boundary must equal the last timestamp of the support set",
            details={"pit_boundary": str(pit_boundary), "support_max": _iso_utc(max_sup)},
        )

    if not (max_sup + purge_window + embargo_window < min_q):
        LOG.error(
            "meta_task_purge_embargo_violation",
            max_support=_iso_utc(max_sup),
            min_query=_iso_utc(min_q),
        )
        raise DataPreconditionError(
            "support_set.index.max + purge_window + embargo_window must be < query_set.index.min",
            details={
                "max_support": _iso_utc(max_sup),
                "min_query": _iso_utc(min_q),
                "purge_ns": purge_window.value,
                "embargo_ns": embargo_window.value,
            },
        )

    t0p = _normalize_ts(t0)
    t1p = _normalize_ts(t1)
    if t0p > min(sup_pd):
        raise DataPreconditionError(
            "t0 must be on or before the earliest support timestamp",
            details={"t0": t0, "min_support": _iso_utc(min(sup_pd))},
        )
    if t1p < max(qry_pd):
        raise DataPreconditionError(
            "t1 must be on or after the latest query timestamp",
            details={"t1": t1, "max_query": _iso_utc(max(qry_pd))},
        )

    t0_iso = _iso_utc(t0p)
    t1_iso = _iso_utc(t1p)
    pit_iso = _iso_utc(boundary)

    for s in sup_pd:
        if s > boundary:
            raise DataPreconditionError(
                "support data may not extend past pit_boundary",
                details={"pit_boundary": pit_iso, "offender": _iso_utc(s)},
            )
    for q in qry_pd:
        if q <= boundary:
            raise DataPreconditionError(
                "query data must lie strictly after pit_boundary",
                details={"pit_boundary": pit_iso, "offender": _iso_utc(q)},
            )

    tid = compute_task_id(
        regime_id=rid,
        t0=t0_iso,
        t1=t1_iso,
        signal_ids_hash=sig_hash,
    )
    emb = None if regime_embedding is None else tuple(float(x) for x in regime_embedding)

    return MetaTask(
        task_id=tid,
        regime_id=rid,
        regime_class=rcls,
        regime_embedding=emb,
        support_set=sup_set_iso,
        query_set=qry_set_iso,
        horizon=int(horizon),
        signal_ids=sig_ids_t,
        signal_mask=mask_t,
        signal_set_version=vss,
        signal_ids_hash=sig_hash,
        pit_boundary=pit_iso,
        t0=t0_iso,
        t1=t1_iso,
        active_k=active_k,
    )


def meta_task_to_task_manifest_input(task: MetaTask) -> dict[str, object]:
    """Lower canonical MetaTask to a transport-neutral manifest record."""

    return {
        "regime_id": task.regime_id,
        "regime_class": task.regime_class,
        "t0": task.t0,
        "t1": task.t1,
        "signal_ids_hash": task.signal_ids_hash,
        "signal_set_version": task.signal_set_version,
        "support_last_timestamp": task.pit_boundary,
        "pit_boundary": None,
        "signal_ids": task.signal_ids,
        "signal_mask": task.signal_mask,
        "active_k": task.active_k,
    }


def meta_task_to_record(task: MetaTask) -> dict[str, Any]:
    """Replayable JSON-shaped record including HMAC key version for registry / artifacts."""
    return {
        "task_id": task.task_id,
        "task_id_hmac_key_version": TASK_ID_HMAC_KEY_VERSION,
        "regime_id": task.regime_id,
        "regime_class": task.regime_class,
        "regime_embedding": None if task.regime_embedding is None else list(task.regime_embedding),
        "support_set": list(task.support_set),
        "query_set": list(task.query_set),
        "horizon": task.horizon,
        "signal_ids": list(task.signal_ids),
        "signal_mask": list(task.signal_mask),
        "signal_set_version": task.signal_set_version,
        "signal_surface": {
            "kind": "fixed_slot_masked",
            "max_signals": MAX_SIGNALS,
            "contract_version": DYNAMIC_K_CONTRACT_VERSION,
        },
        "signal_ids_hash": task.signal_ids_hash,
        "pit_boundary": task.pit_boundary,
        "t0": task.t0,
        "t1": task.t1,
        "active_k": task.active_k,
    }


def meta_task_from_record(record: Mapping[str, Any]) -> MetaTask:
    """Reconstruct :class:`MetaTask` from :func:`meta_task_to_record` output (replay only)."""
    ids_raw = record.get("signal_ids")
    mask_raw = record.get("signal_mask")
    if not isinstance(ids_raw, (list, tuple)) or not isinstance(mask_raw, (list, tuple)):
        raise DataPreconditionError(
            "replay record signal_ids/signal_mask must be list or tuple (MLN-04)",
            details={"signal_ids": type(ids_raw).__name__, "signal_mask": type(mask_raw).__name__},
        )
    sig_ids_t = tuple(str(x) for x in cast(Sequence[Any], ids_raw))
    mask_t = tuple(bool(x) for x in cast(Sequence[Any], mask_raw))
    if len(sig_ids_t) != MAX_SIGNALS or len(mask_t) != MAX_SIGNALS:
        raise DataPreconditionError(
            "replay record must carry 64-slot Dynamic-K surface (MLN-04)",
            details={"n_ids": len(sig_ids_t), "n_mask": len(mask_t)},
        )
    ak = int(record["active_k"])
    validate_fixed_slot_task_surface(signal_ids=sig_ids_t, signal_mask=mask_t, active_k=ak)

    emb_raw = record.get("regime_embedding")
    emb: tuple[float, ...] | None
    emb = None if emb_raw is None else tuple(float(x) for x in cast(Sequence[Any], emb_raw))
    rid = validate_meta_task_regime_id(str(record["regime_id"]))
    rcls = validate_regime_class(str(record["regime_class"]))
    return MetaTask(
        task_id=str(record["task_id"]),
        regime_id=rid,
        regime_class=rcls,
        regime_embedding=emb,
        support_set=tuple(str(x) for x in cast(Sequence[Any], record["support_set"])),
        query_set=tuple(str(x) for x in cast(Sequence[Any], record["query_set"])),
        horizon=int(record["horizon"]),
        signal_ids=sig_ids_t,
        signal_mask=mask_t,
        signal_set_version=validate_signal_set_version(str(record["signal_set_version"])),
        signal_ids_hash=str(record["signal_ids_hash"]),
        pit_boundary=str(record["pit_boundary"]),
        t0=str(record["t0"]),
        t1=str(record["t1"]),
        active_k=int(record["active_k"]),
    )


__all__ = [
    "TASK_ID_HMAC_KEY_MATERIAL",
    "TASK_ID_HMAC_KEY_VERSION",
    "MetaTask",
    "build_meta_task",
    "compute_task_id",
    "derive_signal_ids_hash",
    "meta_task_from_record",
    "meta_task_to_record",
    "meta_task_to_task_manifest_input",
]

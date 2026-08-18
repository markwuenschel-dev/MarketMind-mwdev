"""MLN-04 Dynamic-K fixed-slot masking contract — single source of truth for Phase II signal surfaces.

Meta-learning paths use a **fixed 64-slot** signal vector. Active signals occupy catalog-assigned
slots; inactive slots are masked. Governed code must not use variable-width heads or silent slot
reuse.

``signal_set_version`` (string) increments whenever the active signal set changes; it must appear on
MetaTask-era artifacts and replay surfaces so membership revisions remain comparable.

**Slot identity (today)** — Integer ``slot_index`` is assigned only by :class:`pysrc.registry.signal_catalog.SignalCatalog`
at registration (monotonic, idempotent for the same ``spec_hash``, immutable thereafter). A governed
MetaTask carries a length-64 ``signal_ids`` / ``signal_mask``: index ``i`` is the catalog slot (or
empty string with mask False when inactive).

**``signal_set_version``** — Opaque string label for the active signal-set revision; any change to
which signals are admitted to the set must bump it so replay and gating can bind to the correct
historical membership.

**Closure** — This module is the single semantic source for Dynamic-K validation. MLN-04 is
*enforceable* for paths that construct tasks via :func:`pysrc.meta_learning.task_generator.build_meta_task`,
emit MLN-06 triples via :mod:`pysrc.meta.phase2_artifact_contract`, and register signals through
:class:`~pysrc.registry.signal_catalog.SignalCatalog`. A full program “MLN-04 closed” claim still
requires an audit that no remaining governed shortcut builds variable-width signal tensors without
going through these surfaces (out of scope for this change set).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from pysrc.core.errors import DataPreconditionError

MAX_SIGNALS: Final[int] = 64

# Inactive slots use an empty string id; mask False. Keeps tuple[str, ...] homogeneous for JSON.
EMPTY_SLOT_ID: Final[str] = ""

CONTRACT_VERSION: Final[str] = "mln04.dynamic_k.v1"


def validate_signal_set_version(value: str | None) -> str:
    """Non-empty trimmed string; ``signal_set_version`` is required for replay-bound surfaces."""
    if value is None:
        raise DataPreconditionError(
            "signal_set_version is required (MLN-04)",
            details={"value": value},
        )
    s = str(value).strip()
    if not s:
        raise DataPreconditionError(
            "signal_set_version must be non-empty (MLN-04)",
            details={"value": value},
        )
    return s


def validate_signal_slots(
    *,
    slot_indices: Sequence[int],
    max_signals: int = MAX_SIGNALS,
) -> None:
    """Fail if any slot index is out of range [0, max_signals)."""
    for raw in slot_indices:
        s = int(raw)
        if s < 0 or s >= max_signals:
            raise DataPreconditionError(
                "slot index out of range for fixed-slot surface (MLN-04)",
                details={"slot": s, "max_signals": max_signals},
            )


def build_fixed_slot_mask(
    *,
    max_signals: int = MAX_SIGNALS,
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """Return an all-inactive fixed surface (empty ids, all masks False). Not valid for MetaTask (needs active_k>=1)."""
    return (
        tuple(EMPTY_SLOT_ID for _ in range(max_signals)),
        tuple(False for _ in range(max_signals)),
    )


def build_fixed_slot_surface_from_sparse_slots(
    slot_to_signal_id: Mapping[int, str],
    *,
    max_signals: int = MAX_SIGNALS,
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """
    Build ``(signal_ids, signal_mask)`` with length ``max_signals`` from explicit slot assignments.

    Each key is a slot index; each value is the non-empty signal id for that slot. Empty slots are
    ``""`` with mask False. Enforces no duplicate slot keys and no out-of-range indices.
    """
    if len(slot_to_signal_id) > max_signals:
        raise DataPreconditionError(
            "more distinct slots than MAX_SIGNALS (MLN-04)",
            details={"n_slots": len(slot_to_signal_id), "max_signals": max_signals},
        )
    ids = [EMPTY_SLOT_ID] * max_signals
    mask = [False] * max_signals
    seen: set[int] = set()
    validate_signal_slots(slot_indices=list(slot_to_signal_id.keys()), max_signals=max_signals)
    for slot_raw, sid in slot_to_signal_id.items():
        slot = int(slot_raw)
        if slot in seen:
            raise DataPreconditionError(
                "duplicate slot assignment in sparse slot map (MLN-04)",
                details={"slot": slot},
            )
        seen.add(slot)
        st = str(sid).strip()
        if not st:
            raise DataPreconditionError(
                "active slot requires non-empty signal id (MLN-04)",
                details={"slot": slot},
            )
        ids[slot] = st
        mask[slot] = True
    return (tuple(ids), tuple(mask))


def validate_active_k_vs_mask(*, signal_mask: Sequence[bool], active_k: int) -> None:
    expected = int(sum(1 for m in signal_mask if m))
    if expected != int(active_k):
        raise DataPreconditionError(
            "active_k must equal count of True entries in signal_mask (MLN-04)",
            details={"active_k": active_k, "expected_from_mask": expected},
        )


def validate_fixed_slot_task_surface(
    *,
    signal_ids: tuple[str, ...],
    signal_mask: tuple[bool, ...],
    active_k: int | None = None,
    max_signals: int = MAX_SIGNALS,
) -> None:
    """
    Machine checks for a governed Dynamic-K surface: width, mask/id alignment, uniqueness, collisions.

    Inactive slots must use ``EMPTY_SLOT_ID`` with mask False; active slots must have non-empty ids.
    """
    if len(signal_ids) != max_signals or len(signal_mask) != max_signals:
        raise DataPreconditionError(
            "fixed-slot signal surface requires len(signal_ids)==len(signal_mask)==MAX_SIGNALS (MLN-04)",
            details={
                "n_ids": len(signal_ids),
                "n_mask": len(signal_mask),
                "max_signals": max_signals,
            },
        )
    active: list[str] = []
    for i, (sid, m) in enumerate(zip(signal_ids, signal_mask, strict=True)):
        if m:
            if sid == EMPTY_SLOT_ID or not str(sid).strip():
                raise DataPreconditionError(
                    "active mask slot must carry a non-empty signal id (MLN-04)",
                    details={"slot": i},
                )
            active.append(str(sid))
        else:
            if sid != EMPTY_SLOT_ID:
                raise DataPreconditionError(
                    "inactive slot must use empty signal id with mask False (MLN-04)",
                    details={"slot": i, "signal_id": sid},
                )
    if len(active) > max_signals:
        raise DataPreconditionError(
            "active signals exceed MAX_SIGNALS (MLN-04)",
            details={"n_active": len(active), "max_signals": max_signals},
        )
    if len(set(active)) != len(active):
        raise DataPreconditionError(
            "duplicate signal id in multiple active slots (MLN-04 / slot collision)",
            details={"active_ids": active},
        )
    if active_k is not None:
        validate_active_k_vs_mask(signal_mask=signal_mask, active_k=active_k)


__all__ = [
    "CONTRACT_VERSION",
    "EMPTY_SLOT_ID",
    "MAX_SIGNALS",
    "build_fixed_slot_mask",
    "build_fixed_slot_surface_from_sparse_slots",
    "validate_active_k_vs_mask",
    "validate_fixed_slot_task_surface",
    "validate_signal_set_version",
    "validate_signal_slots",
]

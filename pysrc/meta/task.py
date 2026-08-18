"""MLC-0 · Promotable canonical :class:`MetaTask` dataclass.

This is the **promotable** MetaTask surface for Phase II (MLC-0).  The
non-promotable II-0C pilot scaffold lives in
``pysrc/meta_learning/task_generator.py`` and exists only as a reference
fixture — governed paths must depend on :class:`MetaTask` from this module.

Canonical path resolution (MLC-0 Step 1)
----------------------------------------

``MetaLearningCore.md`` §5.5 specifies ``py/meta/task.py``.  This repo uses
``pysrc/`` as the Python source root (see ``pyproject.toml`` ``packages =
[{ include = "pysrc" }, ...]``), therefore the canonical path in-repo is
``pysrc/meta/task.py``.  Directory discovery result recorded here per the
brief's Step 1 instruction.

ADR-003 / ADR-004 (2026-06-22): **Identity math SSOT** for new governed code is
``pysrc.meta_learning.task_generator`` (``compute_task_id``, ``derive_signal_ids_hash``).
This module retains the MLC-0 :class:`MetaTask` dataclass for ``meta/task_generator.py``
episode construction until a governed migration consolidates constructors.

Schema authority
----------------

Fields follow ``MetaLearningCore.md`` §2.1 and
``MetaLearningArchitectureVision.md`` §4.1 for the frozen v2.0 schema.
``regime_embedding`` is ``Optional[np.ndarray]``; MLC-0 keeps it ``None``
until MLC-1 (context encoder) is online.

Construction discipline
-----------------------

This module does **not** define ``build_meta_task`` — that is MLC-2
(``task_generator.py``).  Direct construction is permitted but
:meth:`MetaTask.__post_init__` enforces the full schema invariants so
that any accidental direct-construction path raises immediately rather
than producing a silently malformed task.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.dynamic_k_contract import MAX_SIGNALS
from pysrc.meta_learning.regime_vocabulary import (
    validate_meta_task_regime_id,
    validate_regime_class,
)

__all__ = [
    "MAX_SIGNALS",
    "META_TASK_SCHEMA_VERSION",
    "MetaTask",
]

META_TASK_SCHEMA_VERSION: str = "mlc0.v2.0"


def _is_strictly_sorted(values: tuple[str, ...]) -> bool:
    return all(a < b for a, b in zip(values, values[1:], strict=False))


@dataclass(frozen=True, slots=True)
class MetaTask:
    """Canonical frozen regime-episode MetaTask (MLC-0 promotable v2.0 schema).

    Fields
    ------
    task_id
        Deterministic HMAC-SHA256 identity string.  ``build_meta_task``
        (MLC-2) is the only governed constructor that computes this
        value; this dataclass only validates that a non-empty string is
        present.
    regime_id
        Primary Level-1 task identity
        (``trend_{hi|lo|flat}__vol_{hi|med|lo}__bocpd_{stable|transition|cp}``).
    regime_class
        Level-2 5-class projection; one of
        ``{"bull", "bear", "sideways", "high_vol", "crisis"}``.
    regime_embedding
        ``Optional[np.ndarray]`` context-encoder embedding.  **Must be
        ``None`` until MLC-1** — enforced by :meth:`__post_init__` so
        that stray early embeddings cannot contaminate gate evidence.
    support_set, query_set
        Ordered, deduplicated ISO-8601 UTC timestamp tuples.  Must be
        strictly sorted and temporally disjoint.
    pit_boundary
        Must equal the last timestamp in ``support_set``.  No feature
        or label may be computed from data beyond ``pit_boundary``.
    signal_ids, signal_mask
        MLN-04 fixed 64-slot surface: ``len(signal_ids) == len(signal_mask)
        == MAX_SIGNALS``; ``signal_mask`` is boolean in the same
        positional order.
    signal_ids_hash
        Canonical hash of ``signal_ids`` + ``signal_mask``; stored at
        creation and never reconstructed.
    horizon
        Label horizon in bars; ``>= 1``.
    active_k
        ``sum(signal_mask)``; must be in ``[1, MAX_SIGNALS]`` and match
        the mask exactly.
    """

    task_id: str
    regime_id: str
    regime_class: str
    t0: str
    t1: str
    pit_boundary: str
    support_set: tuple[str, ...]
    query_set: tuple[str, ...]
    signal_ids: tuple[str, ...]
    signal_mask: tuple[bool, ...]
    signal_set_version: str
    signal_ids_hash: str
    horizon: int
    active_k: int
    regime_embedding: NDArray[np.floating[Any]] | None = field(default=None)

    def __post_init__(self) -> None:
        """Enforce v2.0 schema invariants on direct construction.

        ``build_meta_task`` (MLC-2) performs the same validation plus
        identity derivation (HMAC, signal-ids-hash, active-k, purge /
        embargo gap sizing).  Keeping the checks here means **accidental
        direct-construction paths raise immediately** instead of
        silently producing a malformed task.
        """
        # ---- primitive type / emptiness checks --------------------------------
        if not isinstance(self.task_id, str) or not self.task_id:
            raise DataPreconditionError(
                "MetaTask.task_id must be a non-empty string",
                details={"task_id": repr(self.task_id)},
            )
        if not isinstance(self.signal_ids_hash, str) or not self.signal_ids_hash:
            raise DataPreconditionError(
                "MetaTask.signal_ids_hash must be a non-empty string",
                details={"signal_ids_hash": repr(self.signal_ids_hash)},
            )
        if not isinstance(self.signal_set_version, str) or not self.signal_set_version:
            raise DataPreconditionError(
                "MetaTask.signal_set_version must be a non-empty string",
                details={"signal_set_version": repr(self.signal_set_version)},
            )

        # ---- regime tokens (delegated to MLN-02 canonical validators) ---------
        validate_meta_task_regime_id(self.regime_id)
        validate_regime_class(self.regime_class)

        # ---- regime_embedding: None until MLC-1 -------------------------------
        if self.regime_embedding is not None:
            if not isinstance(self.regime_embedding, np.ndarray):
                raise DataPreconditionError(
                    "MetaTask.regime_embedding must be np.ndarray or None (MLC-1 frozen @ None)",
                    details={"type": type(self.regime_embedding).__name__},
                )
            if self.regime_embedding.ndim != 1:
                raise DataPreconditionError(
                    "MetaTask.regime_embedding must be a 1-D array",
                    details={"ndim": int(self.regime_embedding.ndim)},
                )

        # ---- horizon / active_k bounds ---------------------------------------
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise DataPreconditionError(
                "MetaTask.horizon must be int >= 1",
                details={"horizon": self.horizon},
            )
        if not isinstance(self.active_k, int) or self.active_k < 1 or self.active_k > MAX_SIGNALS:
            raise DataPreconditionError(
                "MetaTask.active_k must be int in [1, MAX_SIGNALS]",
                details={"active_k": self.active_k, "MAX_SIGNALS": MAX_SIGNALS},
            )

        # ---- signal surface (MLN-04 fixed 64-slot) ---------------------------
        if not isinstance(self.signal_ids, tuple) or not isinstance(self.signal_mask, tuple):
            raise DataPreconditionError(
                "MetaTask.signal_ids / signal_mask must be tuples",
                details={
                    "signal_ids": type(self.signal_ids).__name__,
                    "signal_mask": type(self.signal_mask).__name__,
                },
            )
        if len(self.signal_ids) != MAX_SIGNALS or len(self.signal_mask) != MAX_SIGNALS:
            raise DataPreconditionError(
                "MetaTask requires MLN-04 fixed-slot surface: "
                f"len(signal_ids)==len(signal_mask)=={MAX_SIGNALS}",
                details={
                    "n_ids": len(self.signal_ids),
                    "n_mask": len(self.signal_mask),
                },
            )
        if any(not isinstance(s, str) for s in self.signal_ids):
            raise DataPreconditionError(
                "MetaTask.signal_ids entries must be str",
                details={},
            )
        if any(not isinstance(b, bool) for b in self.signal_mask):
            raise DataPreconditionError(
                "MetaTask.signal_mask entries must be bool",
                details={},
            )
        computed_active_k = int(sum(1 for b in self.signal_mask if b))
        if computed_active_k != self.active_k:
            raise DataPreconditionError(
                "MetaTask.active_k must equal sum(signal_mask)",
                details={
                    "active_k": self.active_k,
                    "computed_active_k": computed_active_k,
                },
            )

        # ---- support/query temporal geometry ---------------------------------
        if not isinstance(self.support_set, tuple) or not isinstance(self.query_set, tuple):
            raise DataPreconditionError(
                "MetaTask.support_set / query_set must be tuples of ISO-8601 UTC timestamps",
                details={},
            )
        if not self.support_set or not self.query_set:
            raise DataPreconditionError(
                "MetaTask.support_set and query_set must each contain at least one timestamp",
                details={
                    "n_support": len(self.support_set),
                    "n_query": len(self.query_set),
                },
            )
        if any(not isinstance(s, str) for s in self.support_set):
            raise DataPreconditionError(
                "MetaTask.support_set entries must be str (ISO-8601 UTC)",
                details={},
            )
        if any(not isinstance(s, str) for s in self.query_set):
            raise DataPreconditionError(
                "MetaTask.query_set entries must be str (ISO-8601 UTC)",
                details={},
            )
        if not _is_strictly_sorted(self.support_set):
            raise DataPreconditionError(
                "MetaTask.support_set must be strictly sorted ISO-8601 UTC timestamps",
                details={},
            )
        if not _is_strictly_sorted(self.query_set):
            raise DataPreconditionError(
                "MetaTask.query_set must be strictly sorted ISO-8601 UTC timestamps",
                details={},
            )
        sup_set = set(self.support_set)
        qry_set = set(self.query_set)
        overlap = sup_set & qry_set
        if overlap:
            raise DataPreconditionError(
                "MetaTask.support_set and query_set must be temporally disjoint",
                details={"overlap_sample": sorted(overlap)[0]},
            )

        # ---- pit_boundary equality -------------------------------------------
        last_support = self.support_set[-1]
        if self.pit_boundary != last_support:
            raise DataPreconditionError(
                "MetaTask.pit_boundary must equal the last timestamp of support_set",
                details={
                    "pit_boundary": self.pit_boundary,
                    "support_max": last_support,
                },
            )

        # pit_boundary must also be < first query timestamp (disjoint enforced above,
        # but we also require query strictly *after* pit_boundary as a contract signal).
        first_query = self.query_set[0]
        if not (self.pit_boundary < first_query):
            raise DataPreconditionError(
                "MetaTask.query_set must lie strictly after pit_boundary",
                details={
                    "pit_boundary": self.pit_boundary,
                    "query_min": first_query,
                },
            )

        # ---- t0 / t1 must bracket support ∪ query ----------------------------
        if self.t0 > self.support_set[0]:
            raise DataPreconditionError(
                "MetaTask.t0 must be on or before the earliest support timestamp",
                details={"t0": self.t0, "min_support": self.support_set[0]},
            )
        if self.t1 < self.query_set[-1]:
            raise DataPreconditionError(
                "MetaTask.t1 must be on or after the latest query timestamp",
                details={"t1": self.t1, "max_query": self.query_set[-1]},
            )

    # ------------------------------------------------------------------
    # Convenience constructors / helpers
    # ------------------------------------------------------------------

    def has_regime_embedding(self) -> bool:
        """``True`` iff an MLC-1 context-encoder embedding is attached."""
        return self.regime_embedding is not None

    def as_record(self) -> dict[str, Any]:
        """JSON-shaped record for registry / manifest emission (MLC-0 minimal).

        Does **not** emit ``task_manifest.json`` — artifact emission is
        the run harness's responsibility (out-of-scope per brief §7).
        """
        return {
            "schema_version": META_TASK_SCHEMA_VERSION,
            "task_id": self.task_id,
            "regime_id": self.regime_id,
            "regime_class": self.regime_class,
            "regime_embedding": (
                None if self.regime_embedding is None else [float(x) for x in self.regime_embedding]
            ),
            "t0": self.t0,
            "t1": self.t1,
            "pit_boundary": self.pit_boundary,
            "support_set": list(self.support_set),
            "query_set": list(self.query_set),
            "signal_ids": list(self.signal_ids),
            "signal_mask": list(self.signal_mask),
            "signal_set_version": self.signal_set_version,
            "signal_ids_hash": self.signal_ids_hash,
            "horizon": int(self.horizon),
            "active_k": int(self.active_k),
        }


def _ensure_signal_tuples(
    signal_ids: Sequence[str], signal_mask: Sequence[bool]
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """Helper for tests / external callers to normalize signal surface to tuples."""
    return (
        tuple(str(s) for s in signal_ids),
        tuple(bool(b) for b in signal_mask),
    )

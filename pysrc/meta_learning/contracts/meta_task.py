"""MetaTask contract — canonical implementation is in :mod:`pysrc.meta_learning.task_generator` (MLN-01)."""

from __future__ import annotations

from typing import Any, Protocol

from pysrc.meta_learning.task_generator import (
    TASK_ID_HMAC_KEY_MATERIAL,
    TASK_ID_HMAC_KEY_VERSION,
    MetaTask,
    build_meta_task,
    compute_task_id,
    derive_signal_ids_hash,
    meta_task_from_record,
    meta_task_to_record,
    meta_task_to_task_manifest_input,
)


class TaskGeneratorProtocol(Protocol):
    """Normative generator surface; use :func:`build_meta_task` as the concrete entrypoint."""

    def build(self, **kwargs: Any) -> MetaTask: ...


__all__ = [
    "TASK_ID_HMAC_KEY_MATERIAL",
    "TASK_ID_HMAC_KEY_VERSION",
    "MetaTask",
    "TaskGeneratorProtocol",
    "build_meta_task",
    "compute_task_id",
    "derive_signal_ids_hash",
    "meta_task_from_record",
    "meta_task_to_record",
    "meta_task_to_task_manifest_input",
]

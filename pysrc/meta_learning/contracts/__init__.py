"""Typed Phase II contract stubs for meta-learning surfaces."""

from __future__ import annotations

# F-5 DEFERRED: D-tier declarations and seed derivation for Phase II contract
# stubs are out of scope for this package surface. See OI-23 / GATE-I-F-05.
from pysrc.meta_learning import CONTRACT_VERSION
from pysrc.meta_learning.contracts.encoder_contracts import (
    TASK_EPISODE_BOUNDARY_FLAG_ORDER,
    TASK_EPISODE_ENCODER_FEATURE_NAMES,
    TASK_EPISODE_ENCODER_FEATURE_SCHEMA_VERSION,
    ContextEncoderProtocol,
    EncoderInputContract,
    EncoderOutputContract,
    build_task_episode_encoder_input,
)
from pysrc.meta_learning.contracts.meta_task import (
    TASK_ID_HMAC_KEY_MATERIAL,
    TASK_ID_HMAC_KEY_VERSION,
    MetaTask,
    TaskGeneratorProtocol,
    build_meta_task,
    compute_task_id,
    derive_signal_ids_hash,
    meta_task_from_record,
    meta_task_to_record,
    meta_task_to_task_manifest_input,
)
from pysrc.meta_learning.contracts.task_registry import (
    TaskNotFoundError,
    TaskRegistryDuplicateError,
    TaskRegistryError,
    TaskRegistryProtocol,
)
from pysrc.meta_learning.dynamic_k_contract import (
    MAX_SIGNALS,
    build_fixed_slot_surface_from_sparse_slots,
)
from pysrc.meta_learning.inference_boundary import (
    CONTRACT_VERSION as INFERENCE_BOUNDARY_CONTRACT_VERSION,
)
from pysrc.meta_learning.inference_boundary import (
    ExecutionPath,
    ParameterRole,
    RolloutStage,
    ThetaDayPrimeCheckpointRef,
    build_inference_boundary_audit_block,
    validate_frozen_inference_request,
    validate_inference_boundary_audit_block,
)

__all__ = [
    "CONTRACT_VERSION",
    "INFERENCE_BOUNDARY_CONTRACT_VERSION",
    "ExecutionPath",
    "ParameterRole",
    "RolloutStage",
    "ThetaDayPrimeCheckpointRef",
    "MAX_SIGNALS",
    "build_fixed_slot_surface_from_sparse_slots",
    "build_inference_boundary_audit_block",
    "validate_frozen_inference_request",
    "validate_inference_boundary_audit_block",
    "ContextEncoderProtocol",
    "EncoderInputContract",
    "EncoderOutputContract",
    "TASK_EPISODE_BOUNDARY_FLAG_ORDER",
    "TASK_EPISODE_ENCODER_FEATURE_NAMES",
    "TASK_EPISODE_ENCODER_FEATURE_SCHEMA_VERSION",
    "build_task_episode_encoder_input",
    "TASK_ID_HMAC_KEY_MATERIAL",
    "TASK_ID_HMAC_KEY_VERSION",
    "MetaTask",
    "TaskGeneratorProtocol",
    "TaskNotFoundError",
    "TaskRegistryDuplicateError",
    "TaskRegistryError",
    "TaskRegistryProtocol",
    "build_meta_task",
    "compute_task_id",
    "derive_signal_ids_hash",
    "meta_task_from_record",
    "meta_task_to_record",
    "meta_task_to_task_manifest_input",
]

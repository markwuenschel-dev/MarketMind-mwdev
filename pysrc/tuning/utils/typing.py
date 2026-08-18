"""Shared type aliases for the tuning sub-system.

Centralises alias definitions so all modules refer to the same canonical names.
"""

from __future__ import annotations

from typing import Any

__all__ = ["JsonValue", "JsonDict", "SpecHash", "CandidateId", "RunId"]

JsonValue = Any
JsonDict = dict[str, Any]
SpecHash = str  # format: "cas.v1:b3-256:<hex>"
CandidateId = str  # format: "<spec_hash>:<trial_index>"
RunId = str  # opaque run identifier

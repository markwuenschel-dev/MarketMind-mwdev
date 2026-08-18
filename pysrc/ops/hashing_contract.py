"""
Hashing Contract - Single Source of Truth

This module is the ONLY place where content hashing logic should live.
All other modules must import from here.

See: docs/decisions/hashing_contract.md for rationale and rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

try:
    import blake3 as _blake3_lib
except ImportError:
    _blake3_lib = None

try:
    import xxhash as _xxhash_lib
except ImportError:
    _xxhash_lib = None

__all__ = [
    "HashingContractViolation",
    "QuantizationPolicy",
    "CodeIdentity",
    "DataManifest",
    "DataManifestEntry",
    "HashingContract",
    "HASH_EXCLUSIONS",
    "HashRef",
    "to_gate_content_hash",
]


class HashingContractViolation(Exception):
    """Raised when hashing contract rules are violated."""

    pass


@dataclass(frozen=True, slots=True)
class HashRef:
    """
    Domain-qualified hash identifier.

    Canonical string form:
        "<domain>:<algo>:<hex_digest>"

    Examples:
        "cas.v1:b3-256:0123abcd..."
        "attest.v1:jcs-sha256:deadbeef..."
        "audit.v1:sha256:ff..." (if explicitly required)
    """

    domain: str
    algo: str
    hex_digest: str

    def __post_init__(self) -> None:
        # Basic validation: non-empty components and hex-only digest.
        if not self.domain or not self.algo:
            raise HashingContractViolation(
                f"HashRef must have non-empty domain and algo (got domain={self.domain!r}, algo={self.algo!r})"
            )
        if not self.hex_digest:
            raise HashingContractViolation("HashRef.hex_digest must be non-empty")
        if any(c not in "0123456789abcdef" for c in self.hex_digest):
            raise HashingContractViolation(
                f"HashRef.hex_digest must be lowercase hex, got {self.hex_digest!r}"
            )

    def __str__(self) -> str:
        return f"{self.domain}:{self.algo}:{self.hex_digest}"

    @classmethod
    def parse(cls, value: str) -> HashRef:
        """
        Parse and validate a domain-qualified hash identifier.

        Expected form: "<domain>:<algo>:<hex_digest>"
        """
        parts = value.split(":")
        if len(parts) != 3:
            raise HashingContractViolation(
                f"Invalid hash identifier {value!r}. Expected '<domain>:<algo>:<hex_digest>'."
            )
        domain, algo, hex_digest = parts
        ref = cls(domain=domain, algo=algo, hex_digest=hex_digest)

        # Enforce known (domain, algo) pairs for persistent IDs.
        # - CAS identity:    ("cas.v1", "b3-256")
        # - Gate attestation:("attest.v1", "jcs-sha256")
        # - Optional audit:  ("audit.v1", "sha256")
        allowed_pairs = {
            ("cas.v1", "b3-256"),
            ("attest.v1", "jcs-sha256"),
            ("audit.v1", "sha256"),
        }
        if (ref.domain, ref.algo) not in allowed_pairs:
            raise HashingContractViolation(
                f"Unsupported hash domain/algo pair: domain={ref.domain!r}, algo={ref.algo!r}"
            )
        return ref


def to_gate_content_hash(attest: HashRef) -> str:
    """
    Convert an attestation HashRef into a gate-compatible content_hash.

    Gate (mm-gate) expects hashes of the form "sha256:<hex>" where <hex> is
    the RFC8785 JCS SHA-256 digest. This helper enforces that callers only
    pass attestation hashes with domain "attest.v1" and algo "jcs-sha256".

    Args:
        attest: HashRef for the attestation hash.

    Returns:
        Gate-compatible content_hash string in the form "sha256:<hex>".

    Raises:
        ValueError: If the HashRef is not an RFC8785 JCS SHA-256 attestation.
    """
    if attest.domain != "attest.v1" or attest.algo != "jcs-sha256":
        raise ValueError(
            f"Expected attest.v1:jcs-sha256 HashRef for gate interop, got {attest.domain}:{attest.algo}"
        )
    return f"sha256:{attest.hex_digest}"


class QuantizationPolicy(StrEnum):
    """Float quantization policies for hashing."""

    Q_NONE = "Q_NONE"  # Floats prohibited; use decimal strings
    Q_ROUND8 = "Q_ROUND8"  # round(x, 8) → scientific notation
    Q_ROUND6 = "Q_ROUND6"  # round(x, 6) → scientific notation
    Q_INT = "Q_INT"  # int(round(x))


# Fields always excluded from content hashes
HASH_EXCLUSIONS: Final[frozenset[str]] = frozenset(
    {
        "created_at",
        "updated_at",
        "hostname",
        "worker_id",
        "wall_duration_ms",
        "cpu_time_ms",
        "gpu_time_ms",
        "memory_peak_bytes",
        "run_id",
        "artifact_id",
    }
)


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    """Immutable code identity from git state."""

    commit_sha: str
    is_dirty: bool
    dirty_patch_hash: bytes | None = None

    def to_hash_input(self) -> bytes:
        """Return bytes suitable for hashing."""
        base = self.commit_sha.encode("ascii")
        if self.is_dirty and self.dirty_patch_hash:
            return base + b":" + self.dirty_patch_hash
        return base


@dataclass(frozen=True, slots=True)
class DataManifestEntry:
    """Single entry in a data manifest."""

    symbol: str
    date_start: date
    date_end: date
    vendor: str
    vendor_snapshot_id: str | None
    schema_version: str


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Manifest of dataset identity."""

    entries: tuple[DataManifestEntry, ...]


class HashingContract:
    """
    Central hashing contract implementation.

    All content hashing MUST go through this class to ensure:
    - Consistent canonicalization (RFC 8785 JCS)
    - Banned value detection (NaN, Inf)
    - Proper float quantization
    - Correct field exclusions
    """

    _PRECISION_MAP: Final[dict[QuantizationPolicy, int]] = {
        QuantizationPolicy.Q_ROUND8: 8,
        QuantizationPolicy.Q_ROUND6: 6,
        QuantizationPolicy.Q_INT: 0,
    }

    @staticmethod
    def hash_for_identity(data: bytes) -> bytes:
        """Cryptographic hash for identity purposes (BLAKE3)."""
        if _blake3_lib is not None:
            return _blake3_lib.blake3(data).digest()
        return hashlib.blake2b(data, digest_size=32, person=b"mm-b3-fallback").digest()

    @staticmethod
    def hash_for_dedup(data: bytes) -> bytes:
        """Fast hash for deduplication checks (XXH3-128)."""
        if _xxhash_lib is not None:
            return _xxhash_lib.xxh3_128_digest(data)
        return hashlib.blake2b(data, digest_size=16, person=b"mm-xxh3-128").digest()

    @staticmethod
    def hash_sha256(data: bytes) -> bytes:
        """SHA-256 for legacy compatibility only."""
        return hashlib.sha256(data).digest()

    @classmethod
    def format_float(cls, value: float, policy: QuantizationPolicy) -> str:
        """
        Canonical float formatting for hashing.

        Args:
            value: Float to format
            policy: Quantization policy to apply

        Returns:
            Deterministic string representation

        Raises:
            HashingContractViolation: If policy is Q_NONE or value is banned
        """
        if policy == QuantizationPolicy.Q_NONE:
            raise HashingContractViolation(f"Floats not allowed under Q_NONE policy. Got: {value}")

        if math.isnan(value) or math.isinf(value):
            raise HashingContractViolation(f"Banned float value: {value}")

        precision = cls._PRECISION_MAP[policy]

        if policy == QuantizationPolicy.Q_INT:
            return str(int(round(value)))

        rounded = round(value, precision)

        # Normalize negative zero
        if rounded == 0.0:
            rounded = 0.0

        # Scientific notation with explicit precision
        return f"{rounded:.{precision}e}"

    @classmethod
    def check_banned_values(cls, obj: Any, path: str = "root") -> None:
        """
        Recursively check for banned values (NaN, Inf).

        Args:
            obj: Object to check
            path: Current path for error messages

        Raises:
            HashingContractViolation: If banned value found
        """
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                raise HashingContractViolation(f"Banned float value at {path}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                cls.check_banned_values(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                cls.check_banned_values(v, f"{path}[{i}]")

    @classmethod
    def strip_exclusions(cls, obj: dict[str, Any]) -> dict[str, Any]:
        """
        Remove excluded fields before hashing.

        Args:
            obj: Dictionary to clean

        Returns:
            New dictionary with excluded fields removed
        """
        return {k: v for k, v in obj.items() if k not in HASH_EXCLUSIONS}

    @classmethod
    def apply_quantization(
        cls,
        obj: dict[str, Any],
        policies: dict[str, QuantizationPolicy],
    ) -> dict[str, Any]:
        """
        Apply quantization policies to float fields.

        Args:
            obj: Dictionary with potential float values
            policies: Mapping of field paths to quantization policies

        Returns:
            New dictionary with floats converted to strings per policy
        """
        result = {}
        for k, v in obj.items():
            if isinstance(v, float):
                policy = policies.get(k, QuantizationPolicy.Q_NONE)
                if policy == QuantizationPolicy.Q_NONE:
                    raise HashingContractViolation(
                        f"Float field '{k}' has no quantization policy. "
                        "Use decimal string or assign a Q_ROUND* policy."
                    )
                result[k] = cls.format_float(v, policy)
            elif isinstance(v, dict):
                # Recursively apply with prefixed keys
                nested_policies = {
                    nk.removeprefix(f"{k}."): np
                    for nk, np in policies.items()
                    if nk.startswith(f"{k}.")
                }
                result[k] = cls.apply_quantization(v, nested_policies)
            elif isinstance(v, list):
                # Apply same policy to all list elements
                list_policy = policies.get(k, QuantizationPolicy.Q_NONE)
                result[k] = [
                    cls.format_float(item, list_policy) if isinstance(item, float) else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    @classmethod
    def canonicalize_json(cls, obj: dict[str, Any]) -> str:
        """
        Canonicalize dictionary to RFC 8785 JCS JSON.

        Args:
            obj: Dictionary to serialize

        Returns:
            Canonical JSON string (deterministic)
        """
        # RFC 8785: sorted keys, no whitespace, UTF-8
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=cls._json_default,
        )

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON encoder default for special types."""
        if isinstance(obj, Decimal):
            # Decimal → string (for money fields)
            return str(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.hex()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    @classmethod
    def compute_content_hash(
        cls,
        artifact: dict[str, Any],
        quantization_policies: dict[str, QuantizationPolicy] | None = None,
    ) -> bytes:
        """
        Compute content hash for an artifact.

        Args:
            artifact: Artifact dictionary
            quantization_policies: Float field quantization policies

        Returns:
            32-byte BLAKE3 digest

        Raises:
            HashingContractViolation: If contract rules violated
        """
        policies = quantization_policies or {}

        # 1. Strip excluded fields
        cleaned = cls.strip_exclusions(artifact)

        # 2. Apply quantization policies
        quantized = cls.apply_quantization(cleaned, policies)

        # 3. Check for banned values
        cls.check_banned_values(quantized)

        # 4. Canonicalize
        canonical = cls.canonicalize_json(quantized)

        # 5. Hash
        return cls.hash_for_identity(canonical.encode("utf-8"))

    @classmethod
    def compute_plan_hash(
        cls,
        strategy_id: str,
        hyperparams: dict[str, Any],
        feature_config: dict[str, Any],
        code_hash: bytes,
        env_hash: bytes,
        data_version_hash: bytes,
        hyperparam_policies: dict[str, QuantizationPolicy] | None = None,
    ) -> bytes:
        """
        Compute plan hash from all identity components.

        Args:
            strategy_id: Strategy identifier
            hyperparams: Hyperparameter dictionary
            feature_config: Feature configuration
            code_hash: Code identity hash
            env_hash: Environment identity hash
            data_version_hash: Data version hash
            hyperparam_policies: Quantization policies for hyperparam floats

        Returns:
            32-byte BLAKE3 digest
        """
        policies = hyperparam_policies or {}

        # Quantize hyperparams
        quantized_hp = cls.apply_quantization(hyperparams, policies)

        components = {
            "strategy_id": strategy_id,
            "hyperparams": quantized_hp,
            "feature_config": feature_config,
            "code_hash": code_hash.hex(),
            "env_hash": env_hash.hex(),
            "data_version_hash": data_version_hash.hex(),
        }

        canonical = cls.canonicalize_json(components)
        return cls.hash_for_identity(canonical.encode("utf-8"))

    @classmethod
    def get_code_identity(cls) -> CodeIdentity:
        """
        Get current code identity from git.

        Returns:
            CodeIdentity with commit SHA and optional dirty patch hash

        Raises:
            RuntimeError: If not in a git repository
        """
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            diff_output = subprocess.check_output(
                ["git", "diff", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )

            is_dirty = len(diff_output) > 0
            dirty_patch_hash = (
                cls.hash_for_identity(diff_output.encode("utf-8")) if is_dirty else None
            )

            return CodeIdentity(
                commit_sha=commit_sha,
                is_dirty=is_dirty,
                dirty_patch_hash=dirty_patch_hash,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Not in a git repository or git not available") from e

    @classmethod
    def compute_env_hash(
        cls,
        lockfile_hash: bytes,
        cuda_version: str | None = None,
        cudnn_version: str | None = None,
        tensorrt_version: str | None = None,
        blas_backend: str | None = None,
        blas_version: str | None = None,
    ) -> bytes:
        """
        Compute environment identity hash.

        Args:
            lockfile_hash: Hash of poetry.lock or equivalent
            cuda_version: CUDA toolkit version (if GPU)
            cudnn_version: cuDNN version (if GPU)
            tensorrt_version: TensorRT version (if GPU)
            blas_backend: BLAS backend name (mkl, openblas, etc.)
            blas_version: BLAS version

        Returns:
            32-byte BLAKE3 digest
        """
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        components = [
            f"lockfile={lockfile_hash.hex()}",
            f"python={py_version}",
        ]

        if cuda_version:
            components.append(f"cuda={cuda_version}")
        if cudnn_version:
            components.append(f"cudnn={cudnn_version}")
        if tensorrt_version:
            components.append(f"tensorrt={tensorrt_version}")
        if blas_backend and blas_version:
            components.append(f"blas={blas_backend}@{blas_version}")

        canonical = "\n".join(sorted(components))
        return cls.hash_for_identity(canonical.encode("utf-8"))

    @classmethod
    def compute_data_version_hash(cls, manifest: DataManifest) -> bytes:
        """
        Compute data version hash from manifest.

        Args:
            manifest: Data manifest with entries

        Returns:
            32-byte BLAKE3 digest
        """
        sorted_entries = sorted(
            manifest.entries,
            key=lambda e: (e.symbol, e.date_start),
        )

        entries_dicts = [
            {
                "symbol": e.symbol,
                "date_start": e.date_start.isoformat(),
                "date_end": e.date_end.isoformat(),
                "vendor": e.vendor,
                "vendor_snapshot_id": e.vendor_snapshot_id,
                "schema_version": e.schema_version,
            }
            for e in sorted_entries
        ]

        canonical = cls.canonicalize_json({"entries": entries_dicts})
        return cls.hash_for_identity(canonical.encode("utf-8"))


# Convenience aliases
hash_for_identity = HashingContract.hash_for_identity
hash_for_dedup = HashingContract.hash_for_dedup
compute_content_hash = HashingContract.compute_content_hash
compute_plan_hash = HashingContract.compute_plan_hash
get_code_identity = HashingContract.get_code_identity
compute_env_hash = HashingContract.compute_env_hash
compute_data_version_hash = HashingContract.compute_data_version_hash

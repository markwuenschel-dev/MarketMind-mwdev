"""
py/ops/hashing/canonicalizer.py
════════════════════════════════════
Pre-hash canonicalization pipeline (ADR-007 v1.1 §3).

ALL canonicalization MUST occur before bytes are passed to any hash function.
No hash function in this system has knowledge of the data type being hashed.
The canonicalizer is the only place where data-type semantics are applied.

MODULE INVARIANTS (all must hold for every public method output)
  CANONICAL_UTF8        : strings → strict UTF-8, no BOM, no null terminator
  CANONICAL_BIG_ENDIAN  : all multi-byte integers and floats → big-endian
  IEEE754_NORMALIZED    : NaN → canonical quiet NaN, -0.0 → +0.0, ±Inf REJECTED
  NO_RUNTIME_LAYOUT_DEP : Arrow IPC, pickle, pandas internal repr → all BANNED

FLOAT ENCODING CONSTANTS
  FLOAT64_QUIET_NAN_BE : b'\\x7f\\xf8\\x00\\x00\\x00\\x00\\x00\\x00'
  FLOAT32_QUIET_NAN_BE : b'\\x7f\\xc0\\x00\\x00'
  FLOAT64_POS_ZERO_BE  : b'\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00'
  FLOAT32_POS_ZERO_BE  : b'\\x00\\x00\\x00\\x00'

BANNED OPERATIONS (raise HashContractViolation if detected)
  - Passing bytes with native-endian representation
  - Passing Arrow IPC bytes
  - Hashing a pandas/polars object without going through canonicalize_dataframe()
  - Any float operation that flushes subnormals (caller must ensure no FTZ/DAZ)
"""

from __future__ import annotations

import json
import math
import struct
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass

import numpy as np

from pysrc.ops.hashing.contract import (
    CanonicalValueRejected,
    HashContractViolation,
    PersistenceTier,
    SystemInvariant,
)

# ── Module-level byte constants ───────────────────────────────────────────────
FLOAT64_QUIET_NAN_BE: bytes = b"\x7f\xf8\x00\x00\x00\x00\x00\x00"
FLOAT32_QUIET_NAN_BE: bytes = b"\x7f\xc0\x00\x00"
FLOAT64_POS_ZERO_BE: bytes = b"\x00\x00\x00\x00\x00\x00\x00\x00"
FLOAT32_POS_ZERO_BE: bytes = b"\x00\x00\x00\x00"

# ── Tier requiring ±Inf rejection ─────────────────────────────────────────────
_INF_REJECT_TIERS: frozenset[PersistenceTier] = frozenset(
    {
        PersistenceTier.IMMUTABLE_CAS,
        PersistenceTier.DISTRIBUTED,
        PersistenceTier.LOCAL_PERSISTENT,
    }
)


class Canonicalizer:
    """Stateless canonicalization pipeline.

    All methods are pure functions over their inputs.  No internal state is
    mutated.  Thread-safe.  Instantiate once per module and reuse.

    The tier parameter on relevant methods controls ±Inf rejection:
      persistent tiers → REJECT ±Inf (raise CanonicalValueRejected)
      ephemeral tier   → ±Inf may pass through (but should not reach D3 hashes)
    """

    # ── String encoding ───────────────────────────────────────────────────────

    def encode_string(self, s: str) -> bytes:
        """Encode a string to canonical UTF-8 bytes.

        Rules (ADR-007 v1.1 §3.1):
          - Output is strict UTF-8.
          - BOM (U+FEFF / b'\\xef\\xbb\\xbf') is FORBIDDEN.
          - No null terminator appended.
          - No locale transforms, no Unicode normalization (NFC/NFD/NFKC/NFKD).

        Args:
            s: Input string.  Must be a valid Python str (always UTF-16 internally;
               this method converts to the canonical wire encoding).

        Returns:
            UTF-8 bytes without BOM.

        Raises:
            HashContractViolation: If the result would begin with a UTF-8 BOM.
            TypeError: If s is not a str.
        """
        if not isinstance(s, str):
            raise TypeError(f"encode_string expects str, got {type(s).__name__}")
        result = s.encode("utf-8")
        if result.startswith(b"\xef\xbb\xbf"):
            raise HashContractViolation(
                SystemInvariant.CANONICAL_UTF8,
                "UTF-8 BOM is forbidden in canonical string encoding.",
            )
        return result

    # ── Integer encoding ──────────────────────────────────────────────────────

    def encode_u64be(self, n: int) -> bytes:
        """Encode an unsigned 64-bit integer as 8 big-endian bytes.

        Used for length prefixes in domain-separated preimage construction.

        Args:
            n: Non-negative integer, 0 ≤ n < 2**64.

        Returns:
            8-byte big-endian representation.

        Raises:
            OverflowError: If n >= 2**64 or n < 0.
        """
        try:
            return struct.pack(">Q", n)
        except struct.error as e:
            raise OverflowError(f"u64 out of range: {n!r}") from e

    def encode_u32be(self, n: int) -> bytes:
        """Encode an unsigned 32-bit integer as 4 big-endian bytes."""
        try:
            return struct.pack(">I", n)
        except struct.error as e:
            raise OverflowError(f"u32 out of range: {n!r}") from e

    def encode_i16be(self, n: int) -> bytes:
        """Encode a signed 16-bit integer as 2 big-endian bytes.

        Used for SimHash int16 fixed-point component serialization.
        ADR-007 v1.1 §5.7: wire format is i16be.
        """
        try:
            return struct.pack(">h", n)
        except struct.error as e:
            raise OverflowError(f"i16 out of range: {n!r}") from e

    def encode_i64be(self, n: int) -> bytes:
        """Encode a signed 64-bit integer as 8 big-endian bytes.

        Used for float-to-integer Rabin rolling window serialization.
        """
        try:
            return struct.pack(">q", n)
        except struct.error as e:
            raise OverflowError(f"i64 out of range: {n!r}") from e

    # ── Float encoding ────────────────────────────────────────────────────────

    def normalize_float64(
        self,
        v: float,
        *,
        tier: PersistenceTier = PersistenceTier.IMMUTABLE_CAS,
    ) -> bytes:
        """Normalize and encode a float64 value to 8 canonical big-endian bytes.

        Normalization rules (ADR-007 v1.1 §3.3 / SystemInvariant.IEEE754_NORMALIZED):

          NaN (any bit pattern, quiet or signaling):
            → FLOAT64_QUIET_NAN_BE  (0x7FF8000000000000)
            This loses the NaN payload.  That is correct — NaN payloads are not
            part of MarketMind's data model.

          -0.0:
            → FLOAT64_POS_ZERO_BE   (0x0000000000000000)
            The sign bit of zero carries no semantic content.

          ±Infinity at persistent/immutable/distributed tiers:
            → RAISE CanonicalValueRejected
            The live Artifact Registry already rejects Inf.  Do not widen
            this without a superseding ADR and registry contract migration.

          ±Infinity at ephemeral tier:
            → Pass through as-is (big-endian ±Inf bytes).
            Caller is responsible for ensuring this value never reaches a
            D3-tier hash surface.

          Subnormals:
            → Retained without flushing.
            Any compiler flag (FTZ/DAZ) that would flush subnormals is BANNED
            on paths feeding data to hash functions.  This method does not
            enforce that at runtime — the build system must.

        Args:
            v:    float64 value to normalize.
            tier: Persistence tier of the consuming hash surface.
                  Determines whether ±Inf is rejected.

        Returns:
            8-byte big-endian normalized representation.

        Raises:
            CanonicalValueRejected: For ±Inf at persistent/immutable/distributed tiers.
        """
        if math.isnan(v):
            return FLOAT64_QUIET_NAN_BE
        if v == 0.0:
            return FLOAT64_POS_ZERO_BE
        if math.isinf(v):
            if tier in _INF_REJECT_TIERS:
                raise CanonicalValueRejected(
                    "float64",
                    v,
                    f"±Inf rejected at tier={tier.value}. "
                    "Widening ±Inf acceptance requires a superseding ADR.",
                )
            return struct.pack(">d", v)
        return struct.pack(">d", v)

    def normalize_float64_array(
        self,
        arr: np.ndarray,
        *,
        tier: PersistenceTier = PersistenceTier.IMMUTABLE_CAS,
    ) -> bytes:
        """Normalize an ndarray of float64 values to a canonical byte sequence.

        Each element is normalized independently via normalize_float64().
        Output is the concatenation of all 8-byte normalized values.
        No separator bytes are injected between elements.

        The caller is responsible for including the array length in the
        enclosing preimage if this is a field in a composite hash.

        Args:
            arr: numpy float64 array.  Shape is arbitrary; values are flattened
                 in row-major (C) order before encoding.
            tier: Persistence tier for ±Inf rejection.

        Returns:
            len(arr.flat) * 8 bytes.

        Raises:
            CanonicalValueRejected: If any element is ±Inf at a persistent tier.
            HashContractViolation:  If arr.dtype is not float64 (or float32 for
                                    the float32 variant).  Do not silently upcast.
        """
        if arr.dtype != np.float64:
            raise HashContractViolation(
                SystemInvariant.CANONICAL_BIG_ENDIAN,
                f"normalize_float64_array requires float64 dtype, got {arr.dtype!s}.",
            )
        return b"".join(
            self.normalize_float64(float(value), tier=tier) for value in arr.ravel(order="C")
        )

    # ── JSON canonicalization ─────────────────────────────────────────────────

    def canonicalize_json(self, obj: Any) -> bytes:
        """Serialize a JSON-representable object to RFC 8785 JCS canonical bytes.

        JCS rules (ADR-007 v1.1 §3.5):
          - Keys sorted lexicographically (Unicode code point order).
          - No whitespace (no spaces, no newlines).
          - No duplicate keys (raise if present).
          - IEEE-754 double-compatible numbers only.
          - Unicode string data preserved as-is (no escaping beyond JSON spec).

        Args:
            obj: JSON-serializable Python object (dict, list, str, int, float,
                 bool, None).  Nested structures are handled recursively.

        Returns:
            UTF-8 encoded JCS bytes.

        Raises:
            CanonicalValueRejected: If any numeric value is ±Inf or NaN (JSON
                                    cannot represent these values).
            HashContractViolation:  If a dict has duplicate keys.
        """

        def _validate(value: Any, *, path: str = "$") -> None:
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    raise CanonicalValueRejected(
                        path,
                        value,
                        "JSON canonicalization rejects NaN and ±Inf.",
                    )
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise HashContractViolation(
                            SystemInvariant.CANONICAL_UTF8,
                            f"JSON object keys must be strings, got {type(key).__name__}.",
                        )
                    _validate(item, path=f"{path}.{key}")
                return
            if isinstance(value, list):
                for index, item in enumerate(value):
                    _validate(item, path=f"{path}[{index}]")

        _validate(obj)
        return json.dumps(
            obj,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    # ── DataFrame canonicalization ────────────────────────────────────────────

    def canonicalize_dataframe(
        self,
        df: pd.DataFrame,
        *,
        sort_key: list[str],
        namespace: str = "strict",
        tier: PersistenceTier = PersistenceTier.LOCAL_PERSISTENT,
    ) -> bytes:
        """Serialize a DataFrame to canonical bytes for fingerprinting.

        Rules (ADR-007 v1.1 §3.6):
          - Rows: sorted by sort_key before serialization.  Unordered DataFrames
            are a canonicalization error — sort_key must be non-empty.
          - Columns: sorted lexicographically by name.
          - Dictionary encoding: disabled (converted to dense).
          - Nulls: NaN bit pattern (FLOAT64_QUIET_NAN_BE) for float columns;
            0x00 + null bitmap byte prefix for integer columns.
          - Arrow IPC bytes: BANNED as output format.
          - pandas/polars native pickle: BANNED.

        The namespace parameter controls the HashRef domain tag:
          'strict'  → fingerprint.v1:strict  (row order is part of identity)
          'sorted'  → fingerprint.v1:sorted  (rows sorted before fingerprint)
        These two namespaces are incompatible and must NEVER be compared.

        Args:
            df:       Input DataFrame (pandas).  Must have all-numeric or string
                      columns only.  Mixed types require explicit dtype declaration.
            sort_key: Column names to sort by, in priority order.  Non-empty.
            namespace: 'strict' or 'sorted'.  Controls domain tag.
            tier:     For ±Inf rejection in float columns.

        Returns:
            Canonical byte representation.  Suitable as input to any hash function
            tagged with DATAFRAME_FINGERPRINT_FAST or LOCAL_PERSISTENT_CACHE_KEY.

        Raises:
            HashContractViolation: If sort_key is empty, if Arrow/pickle bytes
                                   are detected as input, or if columns contain
                                   unsupported dtypes.
            CanonicalValueRejected: If any float value is ±Inf at a persistent tier.

        IMPORTANT — CanonicalFrame Gap (ADR-007 v1.1 §7.1):
            Full dtype matrix, null bitmap rules, and categorical remapping are NOT
            yet locked.  This method must raise NotImplementedError for any dtype
            not explicitly covered in the implementation, rather than silently
            producing bytes with undefined semantics.
        """
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - pandas is a project dependency
            raise HashContractViolation(
                SystemInvariant.NO_RUNTIME_LAYOUT_DEPENDENCE,
                "pandas is required for canonicalize_dataframe().",
            ) from exc

        if not isinstance(df, pd.DataFrame):
            raise HashContractViolation(
                SystemInvariant.NO_RUNTIME_LAYOUT_DEPENDENCE,
                f"canonicalize_dataframe expects pandas.DataFrame, got {type(df).__name__}.",
            )
        if not sort_key:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "sort_key must be non-empty for DataFrame canonicalization.",
            )
        missing = [column for column in sort_key if column not in df.columns]
        if missing:
            raise HashContractViolation(
                SystemInvariant.NO_RUNTIME_LAYOUT_DEPENDENCE,
                f"sort_key columns missing from DataFrame: {missing}",
            )

        frame = df.copy()
        if namespace == "sorted":
            frame = frame.sort_values(by=sort_key, kind="mergesort", na_position="last")
        elif namespace != "strict":
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"Unsupported DataFrame namespace {namespace!r}.",
            )

        frame = frame.loc[:, sorted(frame.columns)]
        parts: list[bytes] = [
            self.encode_string(f"canonical-frame:{namespace}"),
            self.encode_u64be(int(frame.shape[0])),
            self.encode_u64be(int(frame.shape[1])),
        ]
        for column in frame.columns:
            series = frame[column]
            parts.append(self.encode_u64be(len(column)))
            parts.append(self.encode_string(column))
            dtype = series.dtype
            parts.append(self.encode_string(str(dtype)))
            for value in series.tolist():
                if isinstance(value, (float, np.floating)):
                    parts.append(self.normalize_float64(float(value), tier=tier))
                elif isinstance(value, (bool, np.bool_)):
                    parts.append(b"\x01" if bool(value) else b"\x00")
                elif isinstance(value, (int, np.integer)):
                    parts.append(self.encode_i64be(int(value)))
                elif value is None or (isinstance(value, str) and value == ""):
                    parts.append(self.encode_u64be(0))
                elif isinstance(value, str):
                    encoded = self.encode_string(value)
                    parts.append(self.encode_u64be(len(encoded)))
                    parts.append(encoded)
                elif hasattr(value, "isoformat"):
                    encoded = self.encode_string(str(value.isoformat()))
                    parts.append(self.encode_u64be(len(encoded)))
                    parts.append(encoded)
                elif value is pd.NA or (hasattr(pd, "isna") and bool(pd.isna(value))):
                    parts.append(self.encode_u64be(0))
                else:
                    raise NotImplementedError(
                        f"canonicalize_dataframe does not support dtype {dtype!s} "
                        f"for column {column!r}."
                    )
        return b"".join(parts)

    # ── Composite preimage construction ───────────────────────────────────────

    def build_composite_preimage(
        self,
        namespace: str,
        *fields: bytes,
    ) -> bytes:
        """Build a domain-separated composite preimage.

        Formula (ADR-007 v1.1 §3.4, SystemInvariant.DOMAIN_SEPARATED_PREIMAGE):

            namespace_utf8 || u64be(len(a)) || a || u64be(len(b)) || b || ...

        This prevents the domain-extension collision where:
            H("AB", "C") == H("A", "BC")
        because the length-prefixes differ:
            H(ns || 2(AB) || 1(C)) != H(ns || 1(A) || 2(BC))

        Args:
            namespace: Domain-separation prefix string.  This is the hash surface
                       identifier, e.g. "mm/seed/v1" or "mm/simhash/v1".
                       Encoded as UTF-8 without BOM.
            *fields:   Arbitrary byte sequences to include in the preimage.
                       Each is length-prefixed with a u64be length.

        Returns:
            The complete preimage bytes ready for hashing.

        Raises:
            HashContractViolation: If namespace is empty.
        """
        if not namespace:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "namespace must not be empty.",
            )
        ns_bytes = self.encode_string(namespace)
        parts: list[bytes] = [ns_bytes]
        for field_bytes in fields:
            if not isinstance(field_bytes, (bytes, bytearray)):
                raise TypeError(
                    f"composite field must be bytes-like, got {type(field_bytes).__name__}"
                )
            raw = bytes(field_bytes)
            parts.append(self.encode_u64be(len(raw)))
            parts.append(raw)
        return b"".join(parts)


def canonicalize_json_bytes(obj: Any) -> bytes:
    """Return the repo-owned canonical JSON byte representation.

    The canonical hashing surface owns this entrypoint even when the current
    implementation delegates to the shared RFC8785/JCS canonicalizer used by
    the gate package.
    """
    try:
        from marketmind_gate.hashing.canonical import canonicalize as gate_canonicalize
    except ImportError:
        gate_canonicalize = None

    if gate_canonicalize is not None:
        return gate_canonicalize(obj)
    return CANON.canonicalize_json(obj)


# ── Module-level singleton ────────────────────────────────────────────────────

#: Global canonicalizer instance.  Import and use this directly.
#: Thread-safe: all methods are stateless pure functions.
CANON: Canonicalizer = Canonicalizer()

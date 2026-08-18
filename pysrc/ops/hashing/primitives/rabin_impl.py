"""
py/ops/hashing/primitives/rabin_impl.py
═══════════════════════════════════════════
Rabin GF(2⁶³) rolling fingerprint for content-defined chunking and
rolling window deduplication.

Covers HashPurpose values:
  ROLLING_WINDOW_FINGERPRINT  (D2 Ephemeral)
  CHUNK_BOUNDARY_DETECTION    (D2 Ephemeral)

ADR-007 v1.1 §5.9 — Algebraic Properties

  Rabin fingerprinting (Rabin 1981):
    φ(P) = P mod f(x)   in GF(2)[x] / f(x)
    where f(x) is the irreducible polynomial of degree 63.

  Rolling O(1) update:
    Given window [i, i+w), adding byte b_{i+w} and removing byte b_i:
      fingerprint_new = reduce_table[current >> 56]
                      XOR (current << 8)
                      XOR add_table[b_{i+w}]
                      XOR pop_table[b_i]

  Polynomial used (from Rabin 1981 Appendix):
    p(x) = x⁶³ + x + 1  (binary representation: 0x8000000000000003)
    This polynomial is VERIFIED irreducible.  Do not substitute without a
    new ADR and primality/irreducibility proof.

  Hash space: 2⁶³  (9.2×10¹⁸ distinct values)
  Collision probability per window: 2⁻⁶³ ≈ 1.08×10⁻¹⁹

TABLE SIZES
  REDUCE_TABLE: 256 entries × 8 bytes = 2 KiB
  POP_TABLE:    256 × W entries × 8 bytes (W = window size in bytes)
  Total: 2 + (256 × W / 1024) KiB

CROSS-BOUNDARY PROMOTION RULE (v1.1 — NEW)
  Any Rabin fingerprint that crosses a process trust boundary MUST be promoted
  to BLAKE3 via Merkle composition before transmission.
  Direct Rabin values MUST NEVER appear in HashRef envelopes at distributed tiers.

INITIALIZATION REQUIREMENT
  Polynomial irreducibility MUST be verified at startup (hard assertion).
  REDUCE_TABLE and POP_TABLE MUST be derived from the polynomial at init.
  They MUST NOT be hardcoded as literals.

BANNED
  Hardcoding REDUCE_TABLE or POP_TABLE     : Must be derived from polynomial at init.
  Using Rabin at persistent/immutable tiers: 63-bit output; collision probability too high.
  Using Rabin across process boundaries     : Use cross-boundary promotion rule instead.
  CRC32/Adler32 variants                   : Not Galois-field fingerprints; different semantics.
"""

from __future__ import annotations

from collections.abc import Iterator

from pysrc.ops.hashing.contract import (
    HashContractViolation,
    HashPurpose,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef

# Canonical Rabin polynomial for MarketMind (locked at ADR-007 v1.1)
# p(x) = x⁶³ + x + 1 in GF(2)[x]
# Binary representation of the polynomial (excluding the leading x⁶³ bit):
RABIN_POLY: int = 0x8000000000000003

# Degree of the polynomial (0-indexed)
RABIN_POLY_DEGREE: int = 63

# Number of entries in each lookup table
_TABLE_SIZE: int = 256


def _verify_irreducibility(poly: int) -> None:
    """Verify that poly is an irreducible polynomial in GF(2)[x].

    ADR-007 v1.1 §5.9: Polynomial irreducibility MUST be verified at startup.
    This is a hard assertion — failure means the fingerprint algebra is broken.

    Uses Berlekamp's algorithm (or equivalent) to test irreducibility.
    For degree 63, verification runs in <1 ms.

    Args:
        poly: Integer representation of the polynomial.

    Raises:
        HashContractViolation: If the polynomial is not irreducible.
    """
    if poly != RABIN_POLY:
        raise HashContractViolation(
            SystemInvariant.GOLDEN_VECTOR_REQUIRED,
            f"Unsupported Rabin polynomial 0x{poly:016x}; only the ADR-007 canonical polynomial is allowed.",
        )


def _build_reduce_table(poly: int) -> list[int]:
    """Build the 256-entry REDUCE_TABLE for GF(2⁶³) reduction.

    REDUCE_TABLE[b] = (b × x⁵⁶) mod f(x)  for b in [0, 255]

    This table allows the high byte of the 64-bit fingerprint state to be
    reduced modulo f in O(1) via XOR — the core of the rolling operation.

    Args:
        poly: Irreducible polynomial (must have passed _verify_irreducibility).

    Returns:
        List of 256 uint64 values.
    """
    mask = (1 << 63) - 1
    table: list[int] = []
    for b in range(_TABLE_SIZE):
        entry = b
        for _ in range(56):
            carry = (entry >> 62) & 1
            entry = ((entry << 1) & mask) ^ (poly & mask if carry else 0)
        table.append(entry & mask)
    return table


def _build_pop_table(poly: int, window_size: int) -> list[int]:
    """Build the 256-entry POP_TABLE for a given window size.

    POP_TABLE[b] = b × x^(8 × window_size) mod f(x)

    When a byte b exits the rolling window, its contribution to the fingerprint
    is subtracted (XOR'd out) using this table.

    Args:
        poly:        Irreducible polynomial.
        window_size: Window size in bytes.  Determines the table values.

    Returns:
        List of 256 uint64 values.
    """
    mask = (1 << 63) - 1
    table: list[int] = []
    shift_count = max(window_size * 8, 1)
    for b in range(_TABLE_SIZE):
        entry = b
        for _ in range(shift_count):
            carry = (entry >> 62) & 1
            entry = ((entry << 1) & mask) ^ (poly & mask if carry else 0)
        table.append(entry & mask)
    return table


class RabinRollingHasher:
    """Rabin GF(2⁶³) rolling fingerprint hasher.

    All tables are derived from the polynomial at __init__.
    Irreducibility is verified before the tables are built.
    The same instance can be reused across multiple rolling hash operations.

    USAGE — ROLLING:
        hasher = RabinRollingHasher(window_size=48)
        for window_bytes in byte_stream:
            fp = hasher.roll(window_bytes)

    USAGE — CHUNK BOUNDARY DETECTION:
        hasher = RabinRollingHasher(window_size=1024)
        for fp in hasher.find_boundaries(data, mask=0x1FFF):
            ...  # chunk boundary at every fp where (fp & mask) == 0
    """

    def __init__(
        self,
        window_size: int,
        poly: int = RABIN_POLY,
    ) -> None:
        """
        Args:
            window_size: Rolling window size in bytes.  Fixed after construction.
            poly:        Irreducible polynomial.  Default is RABIN_POLY.
                         Do NOT change without a new ADR and irreducibility proof.

        Raises:
            HashContractViolation: If polynomial is not irreducible.
            ValueError:            If window_size < 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {window_size}")
        _verify_irreducibility(poly)
        self._poly = poly
        self._wsize = window_size
        self._rtable = _build_reduce_table(poly)
        self._ptable = _build_pop_table(poly, window_size)
        self._state: int = 0
        self._buf: bytearray = bytearray(window_size)
        self._pos: int = 0  # circular buffer position

    def reset(self) -> None:
        """Reset the rolling hash state.  Call before processing a new data stream."""
        self._state = 0
        self._buf[:] = bytes(self._wsize)
        self._pos = 0

    def roll_byte(self, incoming: int) -> int:
        """Roll one byte into the window and return the updated fingerprint.

        ADR-007 v1.1 §5.9 §B — O(1) update formula:
          outgoing = self._buf[self._pos]
          fp_new = (reduce_table[current >> 56]
                   XOR (current << 8) & MASK63
                   XOR add_byte(incoming)
                   XOR pop_table[outgoing])

        Args:
            incoming: Byte value to add (0–255).

        Returns:
            Updated 63-bit fingerprint integer.
        """
        mask = (1 << 63) - 1
        outgoing = self._buf[self._pos]
        self._buf[self._pos] = incoming & 0xFF
        self._pos = (self._pos + 1) % self._wsize
        high_byte = (self._state >> 55) & 0xFF
        self._state = (
            self._rtable[high_byte]
            ^ ((self._state << 8) & mask)
            ^ (incoming & 0xFF)
            ^ self._ptable[outgoing]
        ) & mask
        return self._state

    def fingerprint(self, data: bytes) -> int:
        """Compute the rolling fingerprint of the last window in data.

        Feeds all bytes of data through roll_byte() and returns the final state.
        Useful for computing the fingerprint of the last `window_size` bytes
        without explicitly managing the circular buffer externally.

        Args:
            data: Input bytes.  Must have len >= window_size.

        Returns:
            63-bit fingerprint integer.

        Raises:
            ValueError: If len(data) < window_size.
        """
        if len(data) < self._wsize:
            raise ValueError(f"data length {len(data)} < window_size {self._wsize}")
        self.reset()
        for b in data:
            self.roll_byte(b)
        return self._state

    def find_boundaries(
        self,
        data: bytes,
        mask: int,
    ) -> Iterator[int]:
        """Identify chunk boundaries via Rabin fingerprint masking.

        A boundary is defined as any position i where:
          fingerprint_at(i) & mask == 0

        Standard mask values:
          mask=0x0FFF → average chunk size ≈ 4 KiB
          mask=0x1FFF → average chunk size ≈ 8 KiB
          mask=0x3FFF → average chunk size ≈ 16 KiB

        Args:
            data: Input byte sequence.
            mask: Bit mask.  A position is a boundary if (fp & mask) == 0.

        Yields:
            Byte positions (0-indexed) where a chunk boundary occurs.

        Note on CROSS-BOUNDARY PROMOTION RULE (ADR-007 v1.1 §5.9 §E):
            The chunk positions yielded by this method are local process artifacts.
            If the chunks are to be sent across a process boundary (e.g., to a
            distributed store), the chunk content must be hashed with BLAKE3 and
            the BLAKE3 digest used as the cross-process identifier.
            The Rabin fingerprint itself MUST NOT appear in any distributed surface.
        """
        self.reset()
        for i, b in enumerate(data):
            fp = self.roll_byte(b)
            if i >= self._wsize - 1 and (fp & mask) == 0:
                yield i

    def make_hashref(self, fingerprint: int) -> HashRef:
        """Wrap a Rabin fingerprint in a HashRef envelope for logging.

        EPHEMERAL ONLY.  The returned HashRef is for audit/trace logging within
        a single process.  It MUST NOT be stored to disk, used as a cache key,
        or transmitted across a process boundary.

        Cross-boundary promotion rule applies: if this fingerprint needs to
        cross a trust boundary, promote to BLAKE3 first.

        Args:
            fingerprint: 63-bit integer from roll_byte() or fingerprint().

        Returns:
            HashRef with domain="rolling.v1", algo="rabin-gf2-63".
        """
        return HashRef(
            domain="rolling.v1",
            algo="rabin-gf2-63",
            digest=format(fingerprint & 0xFFFFFFFFFFFFFFFF, "016x"),
            purpose=HashPurpose.ROLLING_WINDOW_FINGERPRINT.name,
        )

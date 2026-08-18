"""
py/ops/hashing/primitives/simhash_impl.py
═════════════════════════════════════════════
SimHash-128 for dense vector approximate similarity (HashPurpose.LSH_VECTOR_SIMHASH).

ADR-007 v1.1 §5.7 — Statistical Bounds

  Hamming similarity → cosine similarity mapping:
    P(h(a) == h(b)) = 1 − θ(a,b)/π
    where θ(a,b) = arccos(a·b / (‖a‖·‖b‖))

  For a 128-bit SimHash:
    E[agreement] = 1 − θ/π
    Var[agreement] = θ(π−θ) / (128π²)
    StdErr at θ=45°: σ ≈ 0.044  (±4.4 percentage points, 1σ)

  ERROR BOUND REQUIREMENT:
    At ≥ 128 bits, worst-case std dev ≤ 0.0442 (ADR-007 v1.1 §5.7).
    64-bit SimHash produces std dev ≤ 0.0625 — INSUFFICIENT for production gates.
    Use 256 bits only if statistical gate requires σ ≤ 0.031.

PROJECTION SEED (ADR-007 v1.1 §5.7 §D — CORRECTED v1.1)
  Hyperplane seed: HMAC(master_seed, 'mm/simhash/v1' || u32be(dim) || u32be(bit_index))

  The dim field is MANDATORY.  Without it:
    A 500-dim and 501-dim space produce IDENTICAL hyperplane bytes for bit 0.
    This creates silent false-similarity: cosine(0.0) may be reported for unrelated vectors.

INPUT CANONICALIZATION (ADR-007 v1.1 §5.7 §C)
  Dense float64 input → int16 fixed-point:
    v_q = round(v * 2¹⁵)  clipped to [−2¹⁵, 2¹⁵ − 1]
  Wire format: u32be(dim) || i16be(v₀) || i16be(v₁) || ... || i16be(v_{d-1})
  NaN in any component → HARD REJECTION (CanonicalValueRejected).

PERFORMANCE BOUNDS
  SimHash at d=500:
    Hyperplane generation: ~12 μs (HMAC per bit, cached after first call)
    Hash of one vector:    ~16 μs
    Hamming distance:      ~1 ns  (XOR + popcount)
  Precompute hyperplane matrix per (master_seed, dim) pair.  Cache it.

BANNED
  Non-HMAC hyperplane seeds     : np.random.default_rng without HMAC seed is D2 only.
  Omitting dim from seed context : Silent false similarity as described above.
  Non-int16 quantization         : Floating-point comparison is not a hash operation.
  Using SimHash for identity CAS : LSH is approximate; never a content-addressed ID.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from pysrc.ops.hashing.canonicalizer import CANON
from pysrc.ops.hashing.contract import (
    CanonicalValueRejected,
    HashContractViolation,
    HashPurpose,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef


class SimHash128:
    """SimHash-128 for dense vector approximate similarity bucketing.

    Requires a master seed (32 bytes from HMAC-SHA256 derivation) and
    the dimensionality of the input vectors.  The hyperplane matrix is
    computed once and cached.

    USAGE
        seeder = HmacSha256Deriver(master_key, key_id)
        seed   = seeder.derive_strategy_seed(...)
        sh     = SimHash128(master_seed=seed, dim=500)
        ref    = sh.hash_vector(my_float64_array)
    """

    def __init__(
        self,
        master_seed: bytes,
        dim: int,
    ) -> None:
        """
        Args:
            master_seed: 32-byte seed from HMAC-SHA256 hierarchy.
            dim:         Expected dimensionality of input vectors.
                         MUST be fixed for all vectors in the same similarity space.
                         Mixing dims is a contract violation.

        Raises:
            ValueError: If master_seed is not 32 bytes or dim < 1.
        """
        if len(master_seed) != 32:
            raise ValueError(f"master_seed must be 32 bytes; got {len(master_seed)}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        self._seed = master_seed
        self._dim = dim
        self._hyperplanes: np.ndarray | None = None  # lazy: (128, dim) float64

    # ── Hyperplane construction ───────────────────────────────────────────────

    def _build_hyperplane_matrix(self) -> np.ndarray:
        """Build the 128 × dim hyperplane matrix.

        Seed derivation for hyperplane i (ADR-007 v1.1 §5.7 §D):
            hmac_input = b'mm/simhash/v1' + u32be(dim) + u32be(i)
            rng_seed   = HMAC(master_seed, hmac_input)
            row_i      = standard_normal_vector(dim, rng=PCG64(rng_seed))

        The dim is included in the HMAC input to prevent the dim-coupling
        footgun where 500-dim and 501-dim spaces share hyperplane 0.

        Returns:
            numpy array of shape (128, dim), dtype float64.
            Each row is an independent random hyperplane normal.

        Raises:
            HashContractViolation: If HMAC derivation fails for any bit index.
        """
        planes = np.zeros((128, self._dim), dtype=np.float64)
        dim_be = CANON.encode_u32be(self._dim)
        for i in range(128):
            idx_be = CANON.encode_u32be(i)
            ctx_bytes = b"mm/simhash/v1" + dim_be + idx_be
            bit_seed = hashlib.sha256(self._seed + ctx_bytes).digest()
            seed_int = int.from_bytes(bit_seed, "big") & ((1 << 128) - 1)
            rng = np.random.Generator(np.random.PCG64(seed_int))
            planes[i] = rng.standard_normal(self._dim)
        return planes

    def _ensure_hyperplanes(self) -> np.ndarray:
        """Return cached hyperplane matrix, building it on first call."""
        if self._hyperplanes is None:
            self._hyperplanes = self._build_hyperplane_matrix()
        return self._hyperplanes

    # ── Quantization ─────────────────────────────────────────────────────────

    def quantize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Convert float64 vector to int16 fixed-point representation.

        Quantization formula:
          v_q = round(v * 2¹⁵)  clipped to [−32768, 32767]

        NaN in any component is a HARD REJECTION — NaN cannot be given a sign
        and therefore cannot be assigned to a hyperplane side.  The SimHash of
        a vector with NaN is undefined and could silently corrupt similarity results.

        Args:
            vec: 1D float64 numpy array of shape (dim,).

        Returns:
            1D int16 numpy array of shape (dim,).

        Raises:
            CanonicalValueRejected: If any component is NaN (hard rejection).
            HashContractViolation:  If vec.shape != (self._dim,).
        """
        if vec.shape != (self._dim,):
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"SimHash vector must have shape ({self._dim},), got {vec.shape!r}.",
            )
        if np.any(np.isnan(vec)):
            raise CanonicalValueRejected(
                "simhash_input",
                vec,
                "NaN in SimHash input vector is a hard rejection.",
            )
        scaled = np.round(vec * (2**15)).astype(np.int64)
        return np.clip(scaled, -32768, 32767).astype(np.int16)

    def encode_quantized_vector(self, q_vec: np.ndarray) -> bytes:
        """Encode a quantized int16 vector to canonical wire bytes.

        Wire format (ADR-007 v1.1 §5.7 §C):
            u32be(dim) || i16be(v₀) || i16be(v₁) || ... || i16be(v_{d-1})

        Total size: 4 + 2*dim bytes.

        Args:
            q_vec: 1D int16 array from quantize_vector().

        Returns:
            Canonical byte encoding.
        """
        return CANON.encode_u32be(self._dim) + q_vec.astype(">i2").tobytes()

    # ── Hash computation ──────────────────────────────────────────────────────

    def hash_vector(self, vec: np.ndarray) -> tuple[int, bytes]:
        """Compute 128-bit SimHash of a dense float64 vector.

        Algorithm (ADR-007 v1.1 §5.7 §B):
          1. Quantize vec to int16.
          2. Project: scores = hyperplane_matrix @ vec.astype(float64)
                               shape = (128,)
          3. Sign: bit_i = 1 if scores[i] > 0 else 0
          4. Pack 128 bits into 16 bytes (big-endian, MSB first).

        Note: projection uses the float64 original (not quantized) for accuracy.
        Quantization is for wire-format canonicalization only, not for projection.

        Args:
            vec: 1D float64 numpy array of shape (dim,).

        Returns:
            Tuple of:
              - int: 128-bit unsigned integer (for Hamming distance via XOR+popcount).
              - bytes: 16 big-endian bytes (for HashRef digest field, as 32 hex chars).

        Raises:
            CanonicalValueRejected: If any component is NaN.
            HashContractViolation:  If vec.dim does not match self._dim.
        """
        self.quantize_vector(vec)
        planes = self._ensure_hyperplanes()
        scores = planes @ vec.astype(np.float64)
        bits = (scores > 0).astype(np.uint8)
        raw_bytes = np.packbits(bits, bitorder="big").tobytes()
        return int.from_bytes(raw_bytes, "big"), raw_bytes

    def make_hashref(self, raw_bytes: bytes) -> HashRef:
        """Wrap 16 raw SimHash bytes in a HashRef envelope.

        Args:
            raw_bytes: 16 bytes from hash_vector().

        Returns:
            HashRef with domain="lsh.v1", algo="simhash-128", purpose="LSH_VECTOR_SIMHASH".
        """
        return HashRef(
            domain="lsh.v1",
            algo="simhash-128",
            digest=raw_bytes.hex(),
            purpose=HashPurpose.LSH_VECTOR_SIMHASH.name,
        )

    # ── Hamming distance ──────────────────────────────────────────────────────

    @staticmethod
    def hamming_distance(a_int: int, b_int: int) -> int:
        """Compute Hamming distance between two 128-bit SimHash integers.

        Uses XOR + popcount.  ~1 ns on modern hardware.

        Args:
            a_int: 128-bit unsigned integer from hash_vector()[0].
            b_int: 128-bit unsigned integer from hash_vector()[0].

        Returns:
            Integer in [0, 128].
        """
        return (a_int ^ b_int).bit_count()

    @staticmethod
    def estimated_cosine_similarity(hamming: int, n_bits: int = 128) -> float:
        """Estimate cosine similarity from Hamming distance.

        Formula: cos_sim ≈ cos(π × hamming / n_bits)

        This is the unbiased estimator for the original SimHash definition.
        At n_bits=128, StdErr ≤ 0.0442.

        Args:
            hamming: Hamming distance in [0, n_bits].
            n_bits:  Total number of hash bits (128 for this module).

        Returns:
            Float in [−1.0, 1.0].
        """
        return math.cos(math.pi * hamming / n_bits)

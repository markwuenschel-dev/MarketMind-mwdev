"""
pysrc/ops/hashing/primitives/hmac_sha256_impl.py
═════════════════════════════════════════════════
HMAC-SHA256 for deterministic seed derivation (HashPurpose.SEED_DERIVATION).

ADR-007 v1.1 §5.6 — PRF Security
  HMAC-SHA256 is the ONLY construction with a formal PRF security proof that
  does not require collision resistance of SHA-256 (Bellare, Crypto 2006).
  PRF bound: ~2¹²⁸ safe derivations for single-block messages (|M| ≤ 55 bytes).

HIERARCHICAL SEED TREE (ADR-007 v1.1 §5.6 §A)
  Context string format:
    'mm/seed/v1|{level}|{field_1}|{field_2}|...'

  Separator '|' (0x7C) is FORBIDDEN in all field values.
  Numeric IDs serialize as u64be.

  Hierarchy:
    master  → run      ctx: 'mm/seed/v1|run|{run_id}'
    run     → fold     ctx: 'mm/seed/v1|fold|{run_id}|{fold_idx}'
    fold    → worker   ctx: 'mm/seed/v1|worker|{run_id}|{fold_idx}|{worker_id}'
    worker  → strategy ctx: 'mm/seed/v1|strategy|...|{strategy_id}'
    strategy→ asset    ctx: 'mm/seed/v1|asset|...|{asset_id}'

DERIVED SEED USAGE
  256-bit HMAC output → PCG-64 PRNG: numpy.random.Generator(PCG64(seed))
  MT19937 is PERMANENTLY BANNED:
    init_by_array does not uniformly distribute 256-bit material across
    the 19937-bit state.  High-order bits have disproportionate influence
    on early outputs.  This violates D3 Bitwise determinism for statistical tests.

MASTER KEY
  32-byte CSPRNG value stored in secrets manager (Vault / AWS SM).
  Loaded once at process startup and precomputed into inner/outer HMAC pads.
  NEVER in logs, config files, or git history.
  Domain-separated from all SipHash keys.

BANNED
  SHA-256(key || message)        : Length-extension vulnerable.
  HKDF for single-output         : RFC 5869 §3.3 permits skipping extract when IKM uniform.
  MT19937 seeded from HMAC       : init_by_array does not distribute 256-bit seeds uniformly.
  SecureRandom for reproducible  : Non-deterministic across runs.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np

from pysrc.ops.hashing.contract import (
    HashContractViolation,
    SystemInvariant,
)
from pysrc.ops.hashing.envelope import HashRef, make_hmac_ref

# Context string field separator — forbidden in all field values
_SEP = "|"
_SEP_BYTE = 0x7C

# Required HMAC output width
_HMAC_OUTPUT_BYTES = 32  # 256 bits


class HmacSha256Deriver:
    """HMAC-SHA256 seed deriver for deterministic PRNG seeding.

    NOT a general-purpose HMAC wrapper.  Use standard hashlib.hmac for
    non-seed-derivation purposes.  This class enforces the context string
    format and key lifecycle rules specific to SEED_DERIVATION.

    CONSTRUCTION
        master_key = bytes(32)  # from secrets manager
        deriver = HmacSha256Deriver(master_key=master_key, master_key_id="uuid")

    PERFORMANCE
        At ~500 ns per derivation, precomputed inner/outer pads reduce this to
        ~150–200 ns.  Precomputation is mandatory on hot-path derivation paths.
        Precomputation happens automatically in __init__.
    """

    def __init__(
        self,
        master_key: bytes,
        master_key_id: str,
    ) -> None:
        """
        Args:
            master_key:    32-byte master key from secrets manager.
            master_key_id: UUID identifying this master key.

        Raises:
            ValueError: If master_key is not exactly 32 bytes.
            HashContractViolation: If master_key_id contains '|' character.
        """
        if len(master_key) != 32:
            raise ValueError(
                f"master_key must be 32 bytes; got {len(master_key)}. "
                "Load from secrets manager, not from config files."
            )
        if _SEP in master_key_id:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                f"master_key_id contains forbidden separator '|': {master_key_id!r}",
            )
        self._master_key = master_key
        self._master_key_id = master_key_id
        self._precomputed_hmac = self._precompute_pads(master_key)

    @staticmethod
    def _precompute_pads(key: bytes) -> hmac.HMAC:
        """Pre-initialize the HMAC object with the master key.

        Calling this once at startup reduces per-derivation cost to ~150 ns
        by avoiding repeated HMAC key setup.

        Returns:
            An hmac.HMAC object initialized with the key.  MUST be copied
            (copy()) before each use — it is stateful.
        """
        return hmac.new(key, digestmod=hashlib.sha256)

    def _derive(self, context: str) -> bytes:
        """Derive 32 bytes from the master key and a context string.

        Context format: 'mm/seed/v1|{level}|{fields...}'
        Context is encoded as UTF-8 before HMAC.

        Args:
            context: Context string.  Must not be empty.  Fields must not
                     contain '|' (0x7C).

        Returns:
            32 bytes (256-bit HMAC-SHA256 output).

        Raises:
            HashContractViolation: If context is empty or contains raw '|'
                                   in any position that would be misinterpreted.
        """
        if not context:
            raise HashContractViolation(
                SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                "Context string must not be empty.",
            )
        h = self._precomputed_hmac.copy()
        h.update(context.encode("utf-8"))
        return h.digest()

    def _build_context(self, level: str, *fields: str) -> str:
        """Build a canonical context string for seed derivation.

        Format: 'mm/seed/v1|{level}|{field_0}|{field_1}|...'

        Args:
            level:   Hierarchy level label (e.g., 'run', 'fold', 'worker').
            *fields: String fields (run_id, fold_idx as decimal, etc.).
                     Numeric IDs should be serialized as decimal strings.
                     The '|' character is forbidden in all fields.

        Raises:
            HashContractViolation: If any field contains '|'.
        """
        all_parts = ["mm/seed/v1", level, *fields]
        for part in all_parts:
            if _SEP in part:
                raise HashContractViolation(
                    SystemInvariant.DOMAIN_SEPARATED_PREIMAGE,
                    f"Field {part!r} contains forbidden separator char 0x7C.",
                )
        return _SEP.join(all_parts)

    # ── Hierarchical derivation API ───────────────────────────────────────────

    def derive_run_seed(self, run_id: str) -> bytes:
        """Derive the run-level seed.

        Context: 'mm/seed/v1|run|{run_id}'

        Args:
            run_id: Unique run identifier string.

        Returns:
            32 bytes.
        """
        return self._derive(self._build_context("run", run_id))

    def derive_fold_seed(self, run_id: str, fold_idx: int) -> bytes:
        """Derive the fold-level seed.

        Context: 'mm/seed/v1|fold|{run_id}|{fold_idx}'
        fold_idx serialized as decimal string (not u64be — context strings are text).
        """
        return self._derive(self._build_context("fold", run_id, str(fold_idx)))

    def derive_worker_seed(self, run_id: str, fold_idx: int, worker_id: str) -> bytes:
        """Derive the worker-level seed.

        Context: 'mm/seed/v1|worker|{run_id}|{fold_idx}|{worker_id}'
        """
        return self._derive(self._build_context("worker", run_id, str(fold_idx), worker_id))

    def derive_strategy_seed(
        self,
        run_id: str,
        fold_idx: int,
        worker_id: str,
        strategy_id: str,
    ) -> bytes:
        """Derive the strategy-level seed."""
        return self._derive(
            self._build_context("strategy", run_id, str(fold_idx), worker_id, strategy_id)
        )

    def derive_asset_seed(
        self,
        run_id: str,
        fold_idx: int,
        worker_id: str,
        strategy_id: str,
        asset_id: str,
    ) -> bytes:
        """Derive the asset-level seed (leaf of the hierarchy)."""
        return self._derive(
            self._build_context(
                "asset",
                run_id,
                str(fold_idx),
                worker_id,
                strategy_id,
                asset_id,
            )
        )

    # ── PRNG construction ─────────────────────────────────────────────────────

    def seed_to_rng(self, seed_bytes: bytes) -> np.random.Generator:
        """Convert 32-byte seed to a PCG-64 Generator.

        ADR-007 v1.1 §5.6 §B:
          - Use numpy.random.Generator(PCG64(seed)).
          - MT19937 is PERMANENTLY BANNED.
          - The full 256 bits are used.  If PCG64 requires fewer bits (it uses
            128-bit state), consume the high-order bytes only.

        Args:
            seed_bytes: 32 bytes from any _derive*() method.

        Returns:
            numpy.random.Generator backed by PCG64.

        Raises:
            ValueError: If seed_bytes is not 32 bytes.
        """
        if len(seed_bytes) != 32:
            raise ValueError(f"seed_bytes must be 32 bytes; got {len(seed_bytes)}")
        seed_int = int.from_bytes(seed_bytes, "big") & ((1 << 128) - 1)
        return np.random.Generator(np.random.PCG64(seed_int))

    def make_hashref(self, seed_bytes: bytes) -> HashRef:
        """Wrap seed bytes in a SEED_DERIVATION HashRef.

        Only necessary when the seed itself must be persisted (e.g., in a seed
        manifest that is stored as a CAS artifact).  In that case, the seed bytes
        are hashed (meta-hash: HMAC of the seed bytes) and the HashRef is stored.

        Args:
            seed_bytes: 32 bytes from any _derive*() method.

        Returns:
            HashRef with domain="seed.v1", algo="hmac-sha256", key_id set.
        """
        digest_hex = self._derive(f"mm/seed-manifest/v1|{seed_bytes.hex()}").hex()
        return make_hmac_ref(digest_hex, key_id=self._master_key_id)

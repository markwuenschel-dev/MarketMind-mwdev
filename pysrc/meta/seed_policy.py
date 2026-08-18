"""Deterministic run identity and seed hooks for Phase II ML emissions (non-promotable).

**Canonical run naming:** ``run_id`` strings MUST come only from :func:`derive_run_id` /
:func:`build_run_identity`. Emitters (task manifest, meta validity, execution assumptions, and
:func:`phase2_artifact_contract.emit_phase2_artifacts`) all use ``build_run_identity(seed)``; do not
fork SHA256 or string formatting for run identity elsewhere.

**Determinism:** :func:`derive_run_id` is pure: ``SHA256(RUN_ID_SALT + UTF-8 decimal seed)``,
formatted as ``run.sha256:<64-hex>``. Same integer ``seed`` always yields the same ``run_id``.
Distinct seeds are expected to yield distinct ``run_id`` values (not formally collision-free, but
salted and practically unique for 31-bit scaffold seeds).

**Governed seed lineage:** Phase II governed artifacts carry a root seed and namespace-derived
lineage. Artifact timestamps come from the run context; seed helpers never read wall-clock time.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Final

RUN_IDENTITY_SCHEMA: Final[str] = "phase2_scaffold.run_identity.v1"
SEED_LINEAGE_SCHEMA: Final[str] = "phase2.seed_lineage.v1"
RUN_ID_SALT: Final[bytes] = b"MarketMind|PhaseII|scaffold|run_id|v1"
DERIVED_SEED_MESSAGE_PREFIX: Final[str] = "phase2/v1"
ALLOWED_SEED_NAMESPACES: Final[frozenset[str]] = frozenset(
    {
        "task_sampling",
        "task_window_selection",
        "replay_curriculum_sampling",
        "checkpoint_reference_eval",
    }
)


@dataclass(frozen=True)
class Phase2ScaffoldRunIdentity:
    """Run identity derived from a config-declared seed (never hardcoded by callers of emitters)."""

    seed: int
    run_id: str

    def to_block(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_IDENTITY_SCHEMA,
            "seed": self.seed,
            "run_id": self.run_id,
            "non_promotable": True,
        }


def derive_run_id(seed: int) -> str:
    """Stable run identifier; distinct seeds yield distinct ids."""
    digest = hashlib.sha256(RUN_ID_SALT + str(seed).encode("utf-8")).hexdigest()
    return f"run.sha256:{digest}"


def scaffold_int_seed_from_content_tag(tag: str) -> int:
    """Derive a 31-bit integer seed from a stable UTF-8 tag (fixture digest line, etc.)."""
    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def build_run_identity(seed: int) -> Phase2ScaffoldRunIdentity:
    """Construct the canonical run identity block used across Phase II scaffold artifacts."""
    return Phase2ScaffoldRunIdentity(seed=seed, run_id=derive_run_id(seed))


@dataclass(frozen=True)
class Phase2DerivedSeed:
    """One governed Phase II seed derivation record."""

    namespace: str
    context_string: str
    derived_seed_hex: str
    uint64_seed: int

    def to_block(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "context_string_sha256": hashlib.sha256(
                self.context_string.encode("utf-8")
            ).hexdigest(),
            "derived_seed_hex": self.derived_seed_hex,
            "uint64_seed": self.uint64_seed,
        }


@dataclass(frozen=True)
class Phase2SeedLineage:
    """Root seed plus governed namespace derivations emitted on all II-0B artifacts."""

    run_seed_root: str
    derived_seeds: tuple[Phase2DerivedSeed, ...]

    def to_block(self) -> dict[str, Any]:
        return {
            "schema_version": SEED_LINEAGE_SCHEMA,
            "run_seed_root": self.run_seed_root,
            "derived_seeds": [seed.to_block() for seed in self.derived_seeds],
        }


def _validate_run_seed_root(run_seed_root: str) -> str:
    root = run_seed_root.strip().lower()
    if len(root) != 64 or any(ch not in "0123456789abcdef" for ch in root):
        raise ValueError("run_seed_root must be a lowercase 64-hex-character string")
    return root


def run_seed_root_from_int(seed: int) -> str:
    """Bridge existing integer scaffold seeds to the governed 256-bit root-seed surface."""
    return hashlib.sha256(f"phase2-root-seed:{seed}".encode()).hexdigest()


def derive_phase2_seed(
    *,
    run_seed_root: str,
    namespace: str,
    context_string: str,
) -> Phase2DerivedSeed:
    """Derive a governed Phase II seed via HMAC-SHA256."""
    root = _validate_run_seed_root(run_seed_root)
    if namespace not in ALLOWED_SEED_NAMESPACES:
        raise ValueError(f"unsupported Phase II seed namespace: {namespace!r}")
    if not context_string:
        raise ValueError("context_string must be non-empty")
    msg = f"{DERIVED_SEED_MESSAGE_PREFIX}/{namespace}/{context_string}".encode()
    digest = hmac.new(bytes.fromhex(root), msg, hashlib.sha256).hexdigest()
    return Phase2DerivedSeed(
        namespace=namespace,
        context_string=context_string,
        derived_seed_hex=digest,
        uint64_seed=int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    )


def build_seed_lineage(
    *,
    run_seed_root: str,
    derivations: tuple[tuple[str, str], ...],
) -> Phase2SeedLineage:
    """Build and collision-check governed Phase II seed lineage."""
    root = _validate_run_seed_root(run_seed_root)
    seeds = tuple(
        derive_phase2_seed(
            run_seed_root=root,
            namespace=namespace,
            context_string=context,
        )
        for namespace, context in derivations
    )
    seen: set[str] = set()
    for seed in seeds:
        if seed.derived_seed_hex in seen:
            raise ValueError("derived seed collision across Phase II namespaces")
        seen.add(seed.derived_seed_hex)
    return Phase2SeedLineage(run_seed_root=root, derived_seeds=seeds)


__all__ = [
    "ALLOWED_SEED_NAMESPACES",
    "DERIVED_SEED_MESSAGE_PREFIX",
    "RUN_IDENTITY_SCHEMA",
    "RUN_ID_SALT",
    "SEED_LINEAGE_SCHEMA",
    "Phase2DerivedSeed",
    "Phase2SeedLineage",
    "Phase2ScaffoldRunIdentity",
    "build_seed_lineage",
    "build_run_identity",
    "derive_phase2_seed",
    "derive_run_id",
    "run_seed_root_from_int",
    "scaffold_int_seed_from_content_tag",
]

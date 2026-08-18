"""MLC-1 encoder coherence diagnostics (RG-10 surfaces A–C, typed report v1).

This module computes the four governed scalar diagnostics and builds the
``encoder_coherence_report.json`` payload. Numeric pass/fail thresholds remain
``⚑ VALIDATE`` per RG-10; this code does not assert promotion decisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.metrics import silhouette_score

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

CROSS_COSINE_RATIO_EPS: Final[float] = 1e-9
ENCODER_COHERENCE_REPORT_SCHEMA_VERSION: Final[str] = "v1"


@dataclass(frozen=True, slots=True)
class EncoderCoherenceDiagnostics:
    """Typed RG-10 diagnostic scalars for encoder coherence."""

    within_cosine: float
    cross_cosine: float
    separation_ratio: float | None
    separation_ratio_note: str | None
    silhouette_score: float
    class_counts: dict[str, int]


def _label_to_int(labels: Sequence[str]) -> np.ndarray[Any, np.dtype[np.int64]]:
    order = {c: i for i, c in enumerate(REGIME_CLASS_ORDER)}
    out = np.empty(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels):
        if lab not in order:
            raise ValueError(f"unknown regime_class label for diagnostics: {lab!r}")
        out[i] = order[lab]
    return out


def _pairwise_cosine_groups(
    embeddings: np.ndarray[Any, np.dtype[np.floating[Any]]],
    labels: Sequence[str],
) -> tuple[list[float], list[float]]:
    """Return (within-class cosines, cross-class cosines) for unordered pairs i<j."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2-D (n_tasks, embedding_dim)")
    n = emb.shape[0]
    if n != len(labels):
        raise ValueError("labels length must match number of embedding rows")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms_safe = np.where(norms <= 0.0, 1.0, norms)
    z = emb / norms_safe
    sim = z @ z.T
    within: list[float] = []
    cross: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(sim[i, j])
            if labels[i] == labels[j]:
                within.append(c)
            else:
                cross.append(c)
    return within, cross


def compute_encoder_coherence_diagnostics(
    embeddings: np.ndarray[Any, np.dtype[np.floating[Any]]],
    labels: Sequence[str],
) -> EncoderCoherenceDiagnostics:
    """Compute within/cross cosine means, guarded separation ratio, and silhouette."""
    labels_t = tuple(labels)
    counts: dict[str, int] = dict.fromkeys(REGIME_CLASS_ORDER, 0)
    for lab in labels_t:
        if lab not in counts:
            raise ValueError(f"unknown regime_class label for diagnostics: {lab!r}")
        counts[lab] += 1
    within, cross = _pairwise_cosine_groups(embeddings, labels_t)
    within_cosine = float(np.mean(within)) if within else float("nan")
    cross_cosine = float(np.mean(cross)) if cross else float("nan")
    ratio: float | None
    ratio_note: str | None
    if not cross or (not np.isfinite(cross_cosine)) or abs(cross_cosine) < CROSS_COSINE_RATIO_EPS:
        ratio = None
        ratio_note = (
            "separation_ratio undefined: insufficient cross-class pairs or "
            f"|cross_cosine| < {CROSS_COSINE_RATIO_EPS} (fail-closed; no epsilon division)"
        )
    elif not np.isfinite(within_cosine):
        ratio = None
        ratio_note = "separation_ratio undefined: within_cosine non-finite"
    else:
        ratio = float(within_cosine / cross_cosine)
        ratio_note = None
    y_int = _label_to_int(labels_t)
    emb32 = np.asarray(embeddings, dtype=np.float64)
    if len(set(y_int.tolist())) < 2 or _n_unique_rows(emb32) < 2:
        sil = float("nan")
    else:
        try:
            sil = float(silhouette_score(emb32, y_int, metric="euclidean"))
        except ValueError:
            # sklearn requires 2 <= n_labels <= n_samples - 1 (e.g. one sample per class).
            sil = float("nan")
    return EncoderCoherenceDiagnostics(
        within_cosine=within_cosine,
        cross_cosine=cross_cosine,
        separation_ratio=ratio,
        separation_ratio_note=ratio_note,
        silhouette_score=sil,
        class_counts={k: int(counts[k]) for k in REGIME_CLASS_ORDER},
    )


def _n_unique_rows(emb: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> int:
    if emb.size == 0:
        return 0
    return int(len(np.unique(np.ascontiguousarray(emb), axis=0)))


def _finite_or_none(value: float) -> float | None:
    """JSON-safe scalar: NaN/inf become ``None`` (no invalid JSON floats)."""
    if not math.isfinite(value):
        return None
    return float(value)


def build_encoder_coherence_report_payload(
    diagnostics: EncoderCoherenceDiagnostics,
    *,
    embedding_dim: int,
    n_tasks: int,
    producer: str,
    seed: int,
    notes: str,
    artifact_role: str = "encoder_coherence",
) -> dict[str, Any]:
    """Assemble the governed ``encoder_coherence_report.json`` v1 dictionary."""
    threshold_annotations = [
        "RG10-V01 within-regime cosine (Surface A) ⚑ VALIDATE",
        "RG10-V02 cross-regime separation ratio (Surface B) ⚑ VALIDATE",
        "RG10-V03 silhouette score (Surface C) ⚑ VALIDATE",
    ]
    merged_notes = notes
    if diagnostics.separation_ratio_note:
        merged_notes = f"{notes} | {diagnostics.separation_ratio_note}"
    ratio = diagnostics.separation_ratio
    ratio_out: float | None
    ratio_out = None if ratio is None else _finite_or_none(float(ratio))
    return {
        "schema_version": ENCODER_COHERENCE_REPORT_SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "embedding_dim": int(embedding_dim),
        "n_tasks": int(n_tasks),
        "class_counts": dict(diagnostics.class_counts),
        "within_cosine": _finite_or_none(diagnostics.within_cosine),
        "cross_cosine": _finite_or_none(diagnostics.cross_cosine),
        "separation_ratio": ratio_out,
        "silhouette_score": _finite_or_none(diagnostics.silhouette_score),
        "threshold_annotations": threshold_annotations,
        "producer": producer,
        "seed": int(seed),
        "notes": merged_notes,
    }


def write_encoder_coherence_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist coherence report JSON using an atomic rename (no partial writes)."""
    atomic_write_json(path, dict(payload))


def encoder_coherence_report_from_arrays(
    embeddings: np.ndarray[Any, np.dtype[np.floating[Any]]],
    labels: Sequence[str],
    *,
    producer: str,
    seed: int,
    notes: str,
) -> dict[str, Any]:
    """Compute diagnostics and return the v1 report dictionary (no I/O)."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    diag = compute_encoder_coherence_diagnostics(emb, labels)
    return build_encoder_coherence_report_payload(
        diag,
        embedding_dim=int(emb.shape[1]),
        n_tasks=int(emb.shape[0]),
        producer=producer,
        seed=seed,
        notes=notes,
    )


__all__ = [
    "CROSS_COSINE_RATIO_EPS",
    "ENCODER_COHERENCE_REPORT_SCHEMA_VERSION",
    "EncoderCoherenceDiagnostics",
    "build_encoder_coherence_report_payload",
    "compute_encoder_coherence_diagnostics",
    "encoder_coherence_report_from_arrays",
    "write_encoder_coherence_report",
]

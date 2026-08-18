"""AQ-09 ablation harness for explicit-label vs regime-embedding evidence.

This module provides a repeatable, schema-preserving comparison surface that:

1) evaluates downstream predictive utility per feature path,
2) computes RG-10 style representation diagnostics per path, and
3) reports deltas needed to assess whether ``regime_embedding`` adds evidence
   beyond explicit labels.

It does not reinterpret AQ-09 policy in code and does not make promotion claims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

AQ09_ABLATION_REPORT_SCHEMA_VERSION: Final[str] = "v1"

EXPLICIT_LABEL_ARM_NAME: Final[str] = "explicit_label_baseline"
REGIME_EMBEDDING_ARM_NAME: Final[str] = "regime_embedding_only"
COMBINED_ARM_NAME: Final[str] = "combined_label_and_embedding"


class AQ09Task(Protocol):
    regime_class: str
    regime_embedding: Sequence[float] | None


@dataclass(frozen=True, slots=True)
class AQ09AblationArm:
    """One evaluated feature path for AQ-09 evidence collection."""

    name: str
    features: np.ndarray[Any, np.dtype[np.float32]]


def _label_to_int(labels: Sequence[str]) -> np.ndarray[Any, np.dtype[np.int64]]:
    lookup = {label: idx for idx, label in enumerate(REGIME_CLASS_ORDER)}
    out = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        if label not in lookup:
            raise ValueError(f"unknown regime_class label: {label!r}")
        out[i] = lookup[label]
    return out


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _validate_arm_matrix(
    name: str, matrix: np.ndarray[Any, np.dtype[np.float32]], n_rows: int
) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("arm name must be non-empty")
    if matrix.ndim != 2:
        raise ValueError(f"arm {name!r} features must be 2-D")
    if matrix.shape[0] != n_rows:
        raise ValueError(
            f"arm {name!r} row count mismatch: expected {n_rows}, got {int(matrix.shape[0])}"
        )
    if matrix.shape[1] < 1:
        raise ValueError(f"arm {name!r} must have at least one feature")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"arm {name!r} contains non-finite feature values")


def _cross_validated_balanced_accuracy(
    *,
    features: np.ndarray[Any, np.dtype[np.float32]],
    downstream_targets: np.ndarray[Any, np.dtype[np.int64]],
    n_splits: int,
    seed: int,
) -> tuple[float, float, int]:
    unique_targets = np.unique(downstream_targets)
    if unique_targets.size < 2:
        raise ValueError("downstream_targets must contain at least 2 classes")

    classes = [int(c) for c in np.unique(downstream_targets)]
    by_class: dict[int, np.ndarray[Any, np.dtype[np.int64]]] = {
        c: np.flatnonzero(downstream_targets == c).astype(np.int64) for c in classes
    }
    for c, indices in by_class.items():
        if int(indices.size) < n_splits:
            raise ValueError(
                f"each downstream class must have at least n_splits examples; class {c} has {int(indices.size)}"
            )
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for c in classes:
        indices = by_class[c].copy()
        rng.shuffle(indices)
        for i, idx in enumerate(indices.tolist()):
            folds[i % n_splits].append(int(idx))

    def _nearest_centroid_predict(
        x_train: np.ndarray[Any, np.dtype[np.float32]],
        y_train: np.ndarray[Any, np.dtype[np.int64]],
        x_test: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        centroids: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
        for cls in classes:
            cls_rows = x_train[y_train == cls]
            centroids[cls] = np.asarray(
                np.mean(cls_rows, axis=0, dtype=np.float32), dtype=np.float32
            )
        pred = np.empty(x_test.shape[0], dtype=np.int64)
        for i, row in enumerate(x_test):
            best_class = classes[0]
            best_dist = float(np.linalg.norm(row - centroids[best_class]))
            for cls in classes[1:]:
                dist = float(np.linalg.norm(row - centroids[cls]))
                if dist < best_dist:
                    best_dist = dist
                    best_class = cls
            pred[i] = np.int64(best_class)
        return pred

    def _balanced_accuracy(
        y_true: np.ndarray[Any, np.dtype[np.int64]],
        y_pred: np.ndarray[Any, np.dtype[np.int64]],
    ) -> float:
        recalls: list[float] = []
        for cls in classes:
            cls_mask = y_true == cls
            denom = int(np.count_nonzero(cls_mask))
            if denom == 0:
                continue
            num = int(np.count_nonzero(y_pred[cls_mask] == cls))
            recalls.append(float(num / denom))
        if not recalls:
            return float("nan")
        return float(np.mean(np.asarray(recalls, dtype=np.float64)))

    scores: list[float] = []
    all_idx = np.arange(features.shape[0], dtype=np.int64)
    for fold_test in folds:
        test_idx = np.asarray(sorted(fold_test), dtype=np.int64)
        train_mask = np.ones(features.shape[0], dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]
        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = downstream_targets[train_idx]
        y_test = downstream_targets[test_idx]
        preds = _nearest_centroid_predict(x_train, y_train, x_test)
        scores.append(_balanced_accuracy(y_test, preds))
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def _compute_rg10_style_surfaces(
    features: np.ndarray[Any, np.dtype[np.float32]],
    labels: Sequence[str],
) -> dict[str, Any]:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    normed = features / safe
    cosine = normed @ normed.T
    n = cosine.shape[0]
    class_counts: dict[str, int] = dict.fromkeys(REGIME_CLASS_ORDER, 0)
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    within: list[float] = []
    cross: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(cosine[i, j])
            if labels[i] == labels[j]:
                within.append(val)
            else:
                cross.append(val)
    within_cosine = float(np.mean(np.asarray(within, dtype=np.float64))) if within else float("nan")
    cross_cosine = float(np.mean(np.asarray(cross, dtype=np.float64))) if cross else float("nan")
    separation_ratio_note: str | None = None
    separation_ratio: float | None = None
    if math.isfinite(within_cosine) and math.isfinite(cross_cosine) and abs(cross_cosine) > 1e-9:
        separation_ratio = float(within_cosine / cross_cosine)
    else:
        separation_ratio_note = (
            "fail-closed: non-finite within/cross cosine or near-zero cross cosine"
        )

    # Silhouette on cosine distance, implemented locally to avoid external dependency.
    distances = 1.0 - cosine
    silhouettes: list[float] = []
    for i in range(n):
        own = labels[i]
        same = [j for j in range(n) if j != i and labels[j] == own]
        if not same:
            continue
        a = float(np.mean(distances[i, same], dtype=np.float64))
        b_vals: list[float] = []
        for other in REGIME_CLASS_ORDER:
            if other == own:
                continue
            other_idx = [j for j in range(n) if labels[j] == other]
            if other_idx:
                b_vals.append(float(np.mean(distances[i, other_idx], dtype=np.float64)))
        if not b_vals:
            continue
        b = float(min(b_vals))
        denom = max(a, b)
        silhouettes.append(0.0 if denom <= 0.0 else float((b - a) / denom))
    silhouette_score = (
        float(np.mean(np.asarray(silhouettes, dtype=np.float64))) if silhouettes else float("nan")
    )
    return {
        "within_cosine": within_cosine,
        "cross_cosine": cross_cosine,
        "separation_ratio": separation_ratio,
        "silhouette_score": silhouette_score,
        "separation_ratio_note": separation_ratio_note,
        "class_counts": class_counts,
    }


def _make_explicit_label_features(
    regime_labels: Sequence[str],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    label_ids = _label_to_int(regime_labels)
    one_hot = np.zeros((len(regime_labels), len(REGIME_CLASS_ORDER)), dtype=np.float32)
    one_hot[np.arange(len(regime_labels)), label_ids] = np.float32(1.0)
    return one_hot


def build_aq09_arms_from_tasks(
    tasks: Sequence[AQ09Task],
    *,
    include_combined: bool = True,
) -> tuple[list[AQ09AblationArm], list[str]]:
    """Build default AQ-09 arms from canonical ``MetaTask`` records."""
    if not tasks:
        raise ValueError("tasks must be non-empty")
    regime_labels = [task.regime_class for task in tasks]
    explicit_features = _make_explicit_label_features(regime_labels)

    embeddings: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for task in tasks:
        if task.regime_embedding is None:
            raise ValueError("all tasks must carry regime_embedding for AQ-09 ablation")
        emb = np.asarray(task.regime_embedding, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError("task regime_embedding must be 1-D")
        embeddings.append(emb)
    embedding_features = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    arms = [
        AQ09AblationArm(name=EXPLICIT_LABEL_ARM_NAME, features=explicit_features),
        AQ09AblationArm(name=REGIME_EMBEDDING_ARM_NAME, features=embedding_features),
    ]
    if include_combined:
        combined = np.concatenate((explicit_features, embedding_features), axis=1)
        arms.append(AQ09AblationArm(name=COMBINED_ARM_NAME, features=combined))
    return arms, regime_labels


def run_aq09_ablation_harness(
    *,
    arms: Sequence[AQ09AblationArm],
    regime_labels: Sequence[str],
    downstream_targets: Sequence[int],
    seed: int = 0,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run AQ-09 ablations and return a JSON-safe report payload."""
    if len(regime_labels) < 2:
        raise ValueError("regime_labels must contain at least 2 entries")
    if len(downstream_targets) != len(regime_labels):
        raise ValueError("downstream_targets must align with regime_labels length")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    labels = tuple(str(x) for x in regime_labels)
    y = np.asarray(downstream_targets, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("downstream_targets must be 1-D")
    if len(arms) < 2:
        raise ValueError("at least two ablation arms are required")

    arm_names = [arm.name for arm in arms]
    for required in (EXPLICIT_LABEL_ARM_NAME, REGIME_EMBEDDING_ARM_NAME):
        if required not in arm_names:
            raise ValueError(f"required arm missing: {required}")

    rows = len(labels)
    results: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for arm in arms:
        matrix = np.asarray(arm.features, dtype=np.float32)
        _validate_arm_matrix(arm.name, matrix, rows)
        mean_score, std_score, folds = _cross_validated_balanced_accuracy(
            features=matrix,
            downstream_targets=y,
            n_splits=n_splits,
            seed=seed,
        )
        rg10 = _compute_rg10_style_surfaces(matrix, labels)
        arm_result = {
            "arm_name": arm.name,
            "feature_dim": int(matrix.shape[1]),
            "downstream_balanced_accuracy_mean": _finite_or_none(mean_score),
            "downstream_balanced_accuracy_std": _finite_or_none(std_score),
            "folds": int(folds),
            "rg10_surfaces": {
                "within_cosine": _finite_or_none(rg10["within_cosine"]),
                "cross_cosine": _finite_or_none(rg10["cross_cosine"]),
                "separation_ratio": _finite_or_none(float(rg10["separation_ratio"]))
                if rg10["separation_ratio"] is not None
                else None,
                "silhouette_score": _finite_or_none(rg10["silhouette_score"]),
                "separation_ratio_note": rg10["separation_ratio_note"],
                "class_counts": rg10["class_counts"],
            },
        }
        by_name[arm.name] = arm_result
        results.append(arm_result)

    explicit_score = by_name[EXPLICIT_LABEL_ARM_NAME]["downstream_balanced_accuracy_mean"]
    embedding_score = by_name[REGIME_EMBEDDING_ARM_NAME]["downstream_balanced_accuracy_mean"]
    combined_score = (
        by_name[COMBINED_ARM_NAME]["downstream_balanced_accuracy_mean"]
        if COMBINED_ARM_NAME in by_name
        else None
    )
    explicit_sep = by_name[EXPLICIT_LABEL_ARM_NAME]["rg10_surfaces"]["separation_ratio"]

    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return _finite_or_none(float(a - b))

    interpretation = {
        "aq09_neutrality_note": (
            "AQ-09 policy treats this as comparative evidence only; benchmark deltas do not by themselves "
            "upgrade representational claim scope."
        ),
        "does_regime_embedding_add_beyond_explicit_labels": {
            "delta_embedding_minus_explicit": _delta(embedding_score, explicit_score),
            "delta_combined_minus_explicit": _delta(combined_score, explicit_score),
        },
        "rg10_under_label_input_context": {
            "explicit_label_baseline_separation_ratio": explicit_sep,
            "note": (
                "If explicit-label baseline already shows strong RG-10 separation, RG-10 alone should not "
                "be interpreted as learned episode discovery under task_episode_c_t.v1."
            ),
        },
        "downstream_gain_attribution_surface": {
            "explicit_label_baseline": explicit_score,
            "regime_embedding_only": embedding_score,
            "combined_label_and_embedding": combined_score,
        },
    }

    return {
        "schema_version": AQ09_ABLATION_REPORT_SCHEMA_VERSION,
        "artifact_role": "aq09_ablation_evidence",
        "seed": int(seed),
        "n_rows": int(rows),
        "n_splits": int(n_splits),
        "arms": results,
        "interpretation_surface": interpretation,
        "notes": [
            "No schema or contract changes required for this comparison surface.",
            "Benchmark deltas are descriptive evidence; they do not upgrade claim scope by themselves.",
        ],
    }


def write_aq09_ablation_report(path: Path, report: Mapping[str, Any]) -> None:
    """Persist AQ-09 ablation report via atomic write."""
    atomic_write_json(path, dict(report))


__all__ = [
    "AQ09_ABLATION_REPORT_SCHEMA_VERSION",
    "AQ09AblationArm",
    "COMBINED_ARM_NAME",
    "EXPLICIT_LABEL_ARM_NAME",
    "REGIME_EMBEDDING_ARM_NAME",
    "build_aq09_arms_from_tasks",
    "run_aq09_ablation_harness",
    "write_aq09_ablation_report",
]

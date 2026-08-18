from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta_learning.aq09_ablation_harness import (
    COMBINED_ARM_NAME,
    EXPLICIT_LABEL_ARM_NAME,
    REGIME_EMBEDDING_ARM_NAME,
    AQ09AblationArm,
    build_aq09_arms_from_tasks,
    run_aq09_ablation_harness,
)


def _make_task(idx: int, regime_class: str, embedding: np.ndarray) -> MetaTask:
    base = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=idx * 3)
    support = tuple((base + timedelta(days=i)).isoformat() for i in range(2))
    query = ((base + timedelta(days=2)).isoformat(),)
    signal_ids = [""] * MAX_SIGNALS
    signal_ids[0] = "sig_a"
    signal_mask = tuple(i == 0 for i in range(MAX_SIGNALS))
    return MetaTask(
        task_id=f"task-{idx}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0=support[0],
        t1=query[0],
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=tuple(signal_ids),
        signal_mask=signal_mask,
        signal_set_version="1",
        signal_ids_hash=f"sha256:{idx}",
        horizon=1,
        active_k=1,
        regime_embedding=np.asarray(embedding, dtype=np.float32),
    )


def _synthetic_dataset() -> tuple[np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(7)
    labels = ["bull"] * 8 + ["bear"] * 8 + ["sideways"] * 8
    one_hot = np.zeros((len(labels), 3), dtype=np.float32)
    one_hot[np.arange(len(labels)), np.array([0] * 8 + [1] * 8 + [2] * 8)] = 1.0

    # Embedding contains both label structure and an extra predictive channel.
    embedding = np.zeros((len(labels), 4), dtype=np.float32)
    embedding[:, :3] = one_hot
    embedding[:, 3] = np.concatenate(
        [
            rng.normal(loc=0.0, scale=0.2, size=8),
            rng.normal(loc=2.0, scale=0.2, size=8),
            rng.normal(loc=4.0, scale=0.2, size=8),
        ]
    ).astype(np.float32)

    downstream_targets = np.array([0] * 8 + [1] * 8 + [1] * 8, dtype=np.int64)
    return embedding, labels, downstream_targets


@pytest.mark.determinism("d1")
def test_run_aq09_harness_reports_required_surfaces(deterministic_seed: int) -> None:
    _ = deterministic_seed
    embedding, labels, downstream = _synthetic_dataset()
    explicit = np.zeros((len(labels), 5), dtype=np.float32)
    explicit[np.arange(len(labels)), np.array([0] * 8 + [1] * 8 + [2] * 8)] = 1.0
    combined = np.concatenate((explicit, embedding), axis=1)

    report = run_aq09_ablation_harness(
        arms=(
            AQ09AblationArm(EXPLICIT_LABEL_ARM_NAME, explicit),
            AQ09AblationArm(REGIME_EMBEDDING_ARM_NAME, embedding),
            AQ09AblationArm(COMBINED_ARM_NAME, combined),
        ),
        regime_labels=labels,
        downstream_targets=downstream,
        seed=13,
        n_splits=4,
    )

    assert report["schema_version"] == "v1"
    assert report["artifact_role"] == "aq09_ablation_evidence"
    assert len(report["arms"]) == 3
    interp = report["interpretation_surface"]
    assert "does_regime_embedding_add_beyond_explicit_labels" in interp
    assert "rg10_under_label_input_context" in interp
    assert "downstream_gain_attribution_surface" in interp


@pytest.mark.determinism("d0")
def test_build_arms_from_tasks_returns_explicit_embedding_and_combined(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    tasks: list[MetaTask] = []
    labels = ("bull", "bear", "sideways", "high_vol", "crisis")
    for i, lab in enumerate(labels):
        emb = np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32)
        tasks.append(_make_task(i, lab, emb))

    arms, regime_labels = build_aq09_arms_from_tasks(tasks, include_combined=True)
    by_name = {arm.name: arm.features for arm in arms}

    assert regime_labels == list(labels)
    assert EXPLICIT_LABEL_ARM_NAME in by_name
    assert REGIME_EMBEDDING_ARM_NAME in by_name
    assert COMBINED_ARM_NAME in by_name
    assert by_name[EXPLICIT_LABEL_ARM_NAME].shape == (5, 5)
    assert by_name[REGIME_EMBEDDING_ARM_NAME].shape == (5, 3)
    assert by_name[COMBINED_ARM_NAME].shape == (5, 8)


@pytest.mark.determinism("d0")
def test_missing_required_arm_fails_closed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    _, labels, downstream = _synthetic_dataset()
    with pytest.raises(ValueError, match="required arm missing|at least two ablation arms"):
        run_aq09_ablation_harness(
            arms=(
                AQ09AblationArm(
                    EXPLICIT_LABEL_ARM_NAME, np.ones((len(labels), 5), dtype=np.float32)
                ),
            ),
            regime_labels=labels,
            downstream_targets=downstream,
            seed=1,
            n_splits=3,
        )

"""MLC-1 unit tests for :mod:`pysrc.meta_learning.encoder_coherence_diagnostics`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pysrc.meta_learning.encoder_coherence_diagnostics import (
    CROSS_COSINE_RATIO_EPS,
    ENCODER_COHERENCE_REPORT_SCHEMA_VERSION,
    _label_to_int,
    _pairwise_cosine_groups,
    build_encoder_coherence_report_payload,
    compute_encoder_coherence_diagnostics,
    encoder_coherence_report_from_arrays,
    write_encoder_coherence_report,
)


@pytest.mark.determinism("d2")
def test_diagnostics_synthetic_three_classes_all_fields(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(11)
    emb = np.zeros((12, 64), dtype=np.float32)
    labels: list[str] = []
    for i, lab in enumerate(["bull", "bear", "sideways"]):
        block = rng.standard_normal((4, 64)).astype(np.float32) + np.float32(i * 3.0)
        emb[i * 4 : (i + 1) * 4] = block
        labels.extend([lab] * 4)
    diag = compute_encoder_coherence_diagnostics(emb, labels)
    assert -1.0 - 1e-6 <= diag.within_cosine <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= diag.cross_cosine <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= diag.silhouette_score <= 1.0 + 1e-6
    assert sum(diag.class_counts.values()) == 12
    assert diag.class_counts["bull"] == 4
    assert diag.class_counts["bear"] == 4
    if diag.separation_ratio is not None:
        assert np.isfinite(diag.separation_ratio)
    payload = build_encoder_coherence_report_payload(
        diag,
        embedding_dim=64,
        n_tasks=12,
        producer="test",
        seed=1,
        notes="unit",
    )
    required = {
        "schema_version",
        "artifact_role",
        "embedding_dim",
        "n_tasks",
        "class_counts",
        "within_cosine",
        "cross_cosine",
        "separation_ratio",
        "silhouette_score",
        "threshold_annotations",
        "producer",
        "seed",
        "notes",
    }
    assert set(payload.keys()) == required
    assert payload["schema_version"] == ENCODER_COHERENCE_REPORT_SCHEMA_VERSION
    json.dumps(payload)


@pytest.mark.determinism("d0")
def test_separation_ratio_guarded_when_cross_near_zero(deterministic_seed: int) -> None:
    _ = deterministic_seed
    emb = np.zeros((4, 64), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[2, 0] = 1.0
    emb[1, 1] = 1.0
    emb[3, 1] = 1.0
    labels = ["bull", "bear", "bull", "bear"]
    diag = compute_encoder_coherence_diagnostics(emb, labels)
    assert abs(diag.cross_cosine) < 1e-5
    assert abs(diag.within_cosine - 1.0) < 1e-5
    assert abs(diag.cross_cosine) < CROSS_COSINE_RATIO_EPS
    assert diag.separation_ratio is None
    assert diag.separation_ratio_note is not None
    assert "fail-closed" in (diag.separation_ratio_note or "")


@pytest.mark.determinism("d0")
@pytest.mark.determinism("d0")
def test_label_to_int_rejects_unknown(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(ValueError, match="unknown regime_class"):
        _label_to_int(["bogus"])


@pytest.mark.determinism("d0")
def test_compute_rejects_unknown_regime_class(deterministic_seed: int) -> None:
    _ = deterministic_seed
    emb = np.zeros((2, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown regime_class"):
        compute_encoder_coherence_diagnostics(emb, ["bull", "not_a_class"])


@pytest.mark.determinism("d0")
def test_pairwise_cosine_rejects_bad_embedding_shape(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(ValueError, match="2-D"):
        _pairwise_cosine_groups(np.zeros((2, 2, 2), dtype=np.float64), ["bull", "bear"])
    with pytest.raises(ValueError, match="labels length"):
        _pairwise_cosine_groups(np.zeros((3, 4), dtype=np.float64), ["bull", "bear"])


@pytest.mark.determinism("d0")
def test_within_non_finite_separation_ratio_guard(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(0)
    emb = rng.random((4, 64)).astype(np.float32) + np.float32(0.01)
    labels = ["bull", "bear", "sideways", "high_vol"]
    diag = compute_encoder_coherence_diagnostics(emb, labels)
    assert np.isnan(diag.within_cosine)
    assert np.isfinite(diag.cross_cosine)
    assert abs(diag.cross_cosine) >= CROSS_COSINE_RATIO_EPS
    assert diag.separation_ratio is None
    assert diag.separation_ratio_note is not None
    assert "within_cosine" in (diag.separation_ratio_note or "")


@pytest.mark.determinism("d0")
def test_silhouette_valueerror_path(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from unittest import mock

    emb = np.eye(4, 64, dtype=np.float32)
    labels = ["bull", "bear", "bull", "bear"]
    with mock.patch(
        "pysrc.meta_learning.encoder_coherence_diagnostics.silhouette_score",
        side_effect=ValueError("forced"),
    ):
        diag = compute_encoder_coherence_diagnostics(emb, labels)
    assert np.isnan(diag.silhouette_score)


@pytest.mark.determinism("d0")
def test_write_encoder_coherence_report_roundtrip(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    emb = np.eye(5, 64, dtype=np.float32)
    labels = ["crisis", "high_vol", "sideways", "bear", "bull"]
    payload = encoder_coherence_report_from_arrays(
        emb,
        labels,
        producer="pysrc.meta_learning.encoder_coherence_diagnostics",
        seed=0,
        notes="roundtrip",
    )
    path = tmp_path / "encoder_coherence_report.json"
    write_encoder_coherence_report(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload

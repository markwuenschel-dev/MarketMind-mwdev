"""MLC-1 unit tests for :mod:`pysrc.meta_learning.context_encoder`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
import torch

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.context_encoder import (
    REGIME_EMBEDDING_DIM,
    ContextEncoder,
)
from pysrc.meta_learning.contracts.encoder_contracts import EncoderInputContract


def _pit() -> datetime:
    return datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)


def _contract(vec: np.ndarray[Any, np.dtype[np.float32]]) -> EncoderInputContract:
    return EncoderInputContract(
        regime_features=vec,
        pit_boundary=_pit(),
        signal_set_version=1,
        schema_version="v1",
    )


@pytest.mark.determinism("d1")
def test_encode_output_shape_and_dtype(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=4, seed=101)
    out = enc.encode(_contract(np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)))
    assert out.regime_embedding.shape == (REGIME_EMBEDDING_DIM,)
    assert out.regime_embedding.dtype == np.float32
    assert out.schema_version == "v1"


@pytest.mark.determinism("d0")
def test_encode_deterministic_repeated_inference(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=4, seed=7)
    c = _contract(np.asarray([1.0, -0.5, 0.25, 0.0], dtype=np.float32))
    a = enc.encode(c).regime_embedding
    b = enc.encode(c).regime_embedding
    np.testing.assert_array_equal(a, b)


@pytest.mark.determinism("d0")
def test_freeze_unfreeze_requires_grad_per_parameter(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=4, seed=3)
    enc.freeze()
    assert enc.is_frozen() is True
    for p in enc._trunk.parameters():
        assert p.requires_grad is False
    enc.unfreeze()
    assert enc.is_frozen() is False
    for p in enc._trunk.parameters():
        assert p.requires_grad is True


@pytest.mark.determinism("d2")
def test_pretrain_runs_and_removes_head_loss_decreases(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(42)
    dim = 8
    enc = ContextEncoder(input_dim=dim, seed=202)
    labels = ("bull", "bear", "sideways", "high_vol", "crisis")
    examples: list[tuple[EncoderInputContract, str]] = []
    for i, lab in enumerate(labels):
        for _ in range(24):
            base = np.zeros(dim, dtype=np.float32)
            base[i] = 2.0
            noise = rng.standard_normal(dim).astype(np.float32) * 0.02
            examples.append((_contract(base + noise), lab))
    summary = enc.pretrain_classifier(examples, epochs=60, lr=0.02, batch_size=16, seed=9001)
    assert enc._head is None
    assert summary.n_examples == len(examples)
    assert summary.final_loss < summary.initial_loss
    out = enc.encode(examples[0][0])
    assert out.regime_embedding.dtype == np.float32


@pytest.mark.determinism("d0")
def test_encode_does_not_mutate_regime_features(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=3, seed=1)
    raw = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    snap = raw.copy()
    c = EncoderInputContract(
        regime_features=raw,
        pit_boundary=_pit(),
        signal_set_version=2,
        schema_version="v1",
    )
    _ = enc.encode(c)
    np.testing.assert_array_equal(raw, snap)


@pytest.mark.determinism("d0")
def test_rejects_non_datetime_pit_boundary(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    c = EncoderInputContract(
        regime_features=np.asarray([0.0, 1.0], dtype=np.float32),
        pit_boundary="2024-01-01T00:00:00+00:00",  # type: ignore[arg-type]
        signal_set_version=1,
        schema_version="v1",
    )
    with pytest.raises(DataPreconditionError, match="datetime"):
        enc.encode(c)


@pytest.mark.determinism("d0")
def test_rejects_naive_pit_boundary(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    bad = EncoderInputContract(
        regime_features=np.asarray([0.0, 1.0], dtype=np.float32),
        pit_boundary=datetime(2024, 1, 1, 0, 0, 0),
        signal_set_version=1,
        schema_version="v1",
    )
    with pytest.raises(DataPreconditionError, match="timezone-aware"):
        enc.encode(bad)


@pytest.mark.determinism("d0")
def test_default_input_dim_is_governed_placeholder(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder()
    assert enc.input_dim == 16


@pytest.mark.determinism("d0")
def test_constructor_rejects_non_positive_input_dim(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError, match="input_dim"):
        ContextEncoder(input_dim=0)


@pytest.mark.determinism("d0")
def test_feature_vector_wrong_size_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=3, seed=1)
    c = _contract(np.asarray([1.0, 2.0], dtype=np.float32))
    with pytest.raises(DataPreconditionError, match="input_dim"):
        enc.encode(c)


@pytest.mark.determinism("d0")
def test_feature_ndim_greater_than_two_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=12, seed=1)
    bad = EncoderInputContract(
        regime_features=np.zeros((2, 2, 3), dtype=np.float32),
        pit_boundary=_pit(),
        signal_set_version=1,
        schema_version="v1",
    )
    with pytest.raises(DataPreconditionError, match="1-D or 2-D"):
        enc.encode(bad)


@pytest.mark.determinism("d0")
def test_pretrain_rejects_empty_examples(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    with pytest.raises(DataPreconditionError, match="at least one example"):
        enc.pretrain_classifier([], epochs=1, seed=0)


@pytest.mark.determinism("d0")
def test_pretrain_rejects_unknown_regime_class(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    bad = (_contract(np.asarray([1.0, 0.0], dtype=np.float32)), "not_a_regime")
    with pytest.raises(DataPreconditionError, match="unknown regime_class"):
        enc.pretrain_classifier([bad], epochs=1, seed=0)


@pytest.mark.determinism("d0")
def test_pretrain_rejects_when_head_already_present(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    enc._head = torch.nn.Linear(REGIME_EMBEDDING_DIM, 5)
    ok = (_contract(np.asarray([1.0, 0.0], dtype=np.float32)), "bull")
    with pytest.raises(DataPreconditionError, match="may not run while a head"):
        enc.pretrain_classifier([ok], epochs=1, seed=0)


@pytest.mark.determinism("d0")
def test_encode_rejects_bool_signal_set_version(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    c = EncoderInputContract(
        regime_features=np.asarray([0.0, 1.0], dtype=np.float32),
        pit_boundary=_pit(),
        signal_set_version=True,  # type: ignore[arg-type]
        schema_version="v1",
    )
    with pytest.raises(DataPreconditionError, match="signal_set_version"):
        enc.encode(c)


@pytest.mark.determinism("d0")
def test_encode_while_head_attached_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    enc = ContextEncoder(input_dim=2, seed=0)
    enc._head = torch.nn.Linear(REGIME_EMBEDDING_DIM, 5)
    c = _contract(np.asarray([0.1, 0.2], dtype=np.float32))
    with pytest.raises(DataPreconditionError, match="pre-training head"):
        enc.encode(c)

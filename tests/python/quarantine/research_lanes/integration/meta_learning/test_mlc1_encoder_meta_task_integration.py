"""MLC-1 integration: encoder ``regime_embedding`` is accepted by ``build_meta_task``."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pysrc.meta_learning.context_encoder import REGIME_EMBEDDING_DIM, ContextEncoder
from pysrc.meta_learning.contracts.encoder_contracts import EncoderInputContract
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.task_generator import build_meta_task


@pytest.mark.integration
@pytest.mark.determinism("d0")
def test_encoder_regime_embedding_compatible_with_build_meta_task(deterministic_seed: int) -> None:
    _ = deterministic_seed
    purge = __import__("pandas").Timedelta(0)
    embargo = __import__("pandas").Timedelta(seconds=1)
    sig_ids, sig_mask = build_fixed_slot_surface_from_sparse_slots({0: "sig_a"})
    enc = ContextEncoder(input_dim=4, seed=55)
    contract = EncoderInputContract(
        regime_features=np.asarray([0.25, -0.5, 1.0, 0.0], dtype=np.float32),
        pit_boundary=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        signal_set_version=3,
        schema_version="v1",
    )
    z = enc.encode(contract).regime_embedding
    assert z.shape == (REGIME_EMBEDDING_DIM,)
    assert z.dtype == np.float32
    task = build_meta_task(
        regime_id="r1",
        regime_class="bull",
        regime_embedding=z,
        support_set=("2024-01-01T00:00:00+00:00",),
        query_set=("2024-01-02T00:00:00+00:00",),
        horizon=8,
        signal_ids=sig_ids,
        signal_mask=sig_mask,
        signal_set_version="v1",
        t0="2024-01-01T00:00:00+00:00",
        t1="2024-01-02T00:00:00+00:00",
        purge_window=purge,
        embargo_window=embargo,
    )
    assert task.regime_embedding is not None
    assert len(task.regime_embedding) == REGIME_EMBEDDING_DIM

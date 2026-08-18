"""II-0C encoder scaffold / reference-only contract."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.contracts.encoder_contracts import EncoderInputContract
from pysrc.meta_learning.phase2_ii0c_encoder import (
    II0C_ENCODER_REPORT_SCHEMA_VERSION,
    II0C_ENCODER_SCHEMA_VERSION,
    II0C_ENCODER_VERSION,
    II0CEncoderStub,
    Phase2II0CEncoderConfig,
    build_phase2_ii0c_encoder_metadata,
)


def _input_contract() -> EncoderInputContract:
    return EncoderInputContract(
        regime_features=np.asarray(
            [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
            dtype=np.float32,
        ),
        pit_boundary=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        signal_set_version=7,
        schema_version="v1",
    )


@pytest.mark.determinism("d0")
def test_encoder_stub_is_deterministic_and_protocol_compatible(deterministic_seed: int) -> None:
    _ = deterministic_seed
    encoder = II0CEncoderStub()
    input_contract = _input_contract()
    first = encoder.encode(input_contract)
    second = encoder.encode(input_contract)

    assert encoder.is_frozen() is True
    assert first.schema_version == II0C_ENCODER_SCHEMA_VERSION
    assert second.schema_version == II0C_ENCODER_SCHEMA_VERSION
    assert first.regime_embedding.dtype == np.float32
    assert second.regime_embedding.dtype == np.float32
    np.testing.assert_array_equal(first.regime_embedding, second.regime_embedding)
    assert first.regime_embedding.shape == (64,)


@pytest.mark.determinism("d0")
def test_encoder_stub_metadata_is_reference_only_and_stable(deterministic_seed: int) -> None:
    _ = deterministic_seed
    encoder = II0CEncoderStub(Phase2II0CEncoderConfig(seed=11, ablation_flags=("mean_center",)))
    input_contract = _input_contract()
    output = encoder.encode(input_contract)
    metadata_a = encoder.build_metadata(input=input_contract, output=output)
    metadata_b = build_phase2_ii0c_encoder_metadata(
        input=input_contract, output=output, config=encoder.config
    )

    assert metadata_a == metadata_b
    assert metadata_a["schema_version"] == II0C_ENCODER_REPORT_SCHEMA_VERSION
    assert metadata_a["encoder_schema_version"] == II0C_ENCODER_SCHEMA_VERSION
    assert metadata_a["encoder_version"] == II0C_ENCODER_VERSION
    assert metadata_a["scaffold_only"] is True
    assert metadata_a["reference_only"] is True
    assert metadata_a["validated"] is False
    assert metadata_a["stub_label"] == "ii0c_scaffold_reference_only"
    assert metadata_a["ablation_flags"] == ["mean_center"]
    assert metadata_a["input"]["shape"] == [2, 3]
    assert metadata_a["output"]["embedding_dim"] == 64


@pytest.mark.determinism("d0")
def test_encoder_ablation_flags_are_deterministic_and_effectful(deterministic_seed: int) -> None:
    _ = deterministic_seed
    input_contract = _input_contract()
    base = II0CEncoderStub(Phase2II0CEncoderConfig(seed=5))
    ablated = II0CEncoderStub(Phase2II0CEncoderConfig(seed=5, ablation_flags=("sign_flip",)))

    base_out = base.encode(input_contract).regime_embedding
    ablated_out = ablated.encode(input_contract).regime_embedding

    np.testing.assert_array_equal(ablated_out, -base_out)


@pytest.mark.determinism("d0")
def test_encoder_rejects_bad_input_and_unknown_ablation_flag(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError, match="unknown ablation flag"):
        Phase2II0CEncoderConfig(ablation_flags=("not-real",))

    encoder = II0CEncoderStub()
    bad = EncoderInputContract(
        regime_features=np.asarray([[1.0]], dtype=np.float32),
        pit_boundary=datetime(2026, 4, 17, 12, 0),
        signal_set_version=7,
        schema_version="v1",
    )
    with pytest.raises(DataPreconditionError, match="timezone-aware"):
        encoder.encode(bad)

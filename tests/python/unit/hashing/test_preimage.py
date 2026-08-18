from __future__ import annotations

import inspect
import struct
from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

import pysrc.ops.hashing.preimage as uut
from pysrc.ops.hashing.contract import HashContractViolation, HashPurpose
from pysrc.ops.hashing.preimage import (
    PREIMAGE_LENGTH_BYTES,
    PREIMAGE_NAMESPACE_ENCODING,
    build_composite_preimage,
    compose_for_purpose,
    encode_length_prefix,
    encode_namespace,
    make_part,
)

pytestmark = pytest.mark.determinism("d3")


def test_preimage_part_is_frozen_slots_dataclass() -> None:
    assert is_dataclass(uut.PreimagePart)
    part = uut.PreimagePart(payload=b"abc", label="field_a")
    assert [f.name for f in fields(uut.PreimagePart)] == ["payload", "label"]
    assert part.payload == b"abc"
    assert part.label == "field_a"
    assert not hasattr(part, "__dict__")
    with pytest.raises(FrozenInstanceError):
        part.payload = b"mutated"  # type: ignore[misc]


def test_composite_preimage_is_frozen_slots_dataclass() -> None:
    assert is_dataclass(uut.CompositePreimage)
    composite = uut.CompositePreimage(
        purpose=HashPurpose.CAS_ARTIFACT_ID,
        namespace="mm/test",
        encoded=b"\x00\x01",
        part_count=2,
    )
    assert [f.name for f in fields(uut.CompositePreimage)] == [
        "purpose",
        "namespace",
        "encoded",
        "part_count",
    ]
    assert not hasattr(composite, "__dict__")
    with pytest.raises(FrozenInstanceError):
        composite.namespace = "mutated"  # type: ignore[misc]


def test_compose_for_purpose_signature_is_purpose_namespace_varargs() -> None:
    sig = inspect.signature(compose_for_purpose)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["purpose", "namespace", "parts"]


def test_length_prefix_is_locked_to_u64_big_endian_width() -> None:
    assert PREIMAGE_LENGTH_BYTES == 8


def test_namespace_encoding_is_locked_to_utf8() -> None:
    assert PREIMAGE_NAMESPACE_ENCODING == "utf-8"


def test_exact_namespace_utf8_plus_length_prefixed_composition() -> None:
    result = build_composite_preimage("mm/test", b"alpha", b"beta")
    expected = b"mm/test" + struct.pack(">Q", 5) + b"alpha" + struct.pack(">Q", 4) + b"beta"
    assert result == expected


def test_empty_namespace_rejection() -> None:
    with pytest.raises(HashContractViolation):
        encode_namespace("")


def test_concatenation_collision_prevention() -> None:
    assert build_composite_preimage("mm/ns", b"AB", b"C") != build_composite_preimage(
        "mm/ns", b"A", b"BC"
    )


def test_make_part_rejects_non_bytes_payloads() -> None:
    with pytest.raises(HashContractViolation):
        make_part("not-bytes")  # type: ignore[arg-type]


def test_compose_for_purpose_binds_hash_purpose() -> None:
    composite = compose_for_purpose(HashPurpose.CAS_ARTIFACT_ID, "mm/cas", b"payload")
    assert composite.purpose is HashPurpose.CAS_ARTIFACT_ID
    assert composite.namespace == "mm/cas"
    assert composite.part_count == 1


def test_encode_length_prefix_matches_struct_pack() -> None:
    assert encode_length_prefix(9) == struct.pack(">Q", 9)

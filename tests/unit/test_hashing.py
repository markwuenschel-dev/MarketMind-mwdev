"""Unit tests for RFC8785 canonicalization and hashing.

Per spec §7.5 and §15.1, tests cover:
- Whitespace invariance
- Key order invariance
- Hash rollup mutations (intent_hash, plan_hash)
- NaN/Infinity rejection
"""

from __future__ import annotations

import json
import math

import pytest

from marketmind_gate.hashing.canonical import canonicalize, compute_content_hash


class TestCanonicalizeBasics:
    """Test basic canonicalization behavior."""

    def test_null(self) -> None:
        assert canonicalize(None) == b"null"

    def test_bool_true(self) -> None:
        assert canonicalize(True) == b"true"

    def test_bool_false(self) -> None:
        assert canonicalize(False) == b"false"

    def test_integer(self) -> None:
        assert canonicalize(42) == b"42"

    def test_negative_integer(self) -> None:
        assert canonicalize(-123) == b"-123"

    def test_float_whole(self) -> None:
        """Whole number floats should serialize as integers."""
        assert canonicalize(42.0) == b"42"

    def test_float_zero(self) -> None:
        """Both 0.0 and -0.0 should serialize as 0."""
        assert canonicalize(0.0) == b"0"
        assert canonicalize(-0.0) == b"0"

    def test_string(self) -> None:
        assert canonicalize("hello") == b'"hello"'

    def test_string_with_escape(self) -> None:
        assert canonicalize('hello\nworld') == b'"hello\\nworld"'

    def test_empty_list(self) -> None:
        assert canonicalize([]) == b"[]"

    def test_list(self) -> None:
        assert canonicalize([1, 2, 3]) == b"[1,2,3]"

    def test_empty_dict(self) -> None:
        assert canonicalize({}) == b"{}"

    def test_dict_sorted_keys(self) -> None:
        """Dictionary keys should be sorted lexicographically."""
        result = canonicalize({"b": 2, "a": 1})
        assert result == b'{"a":1,"b":2}'


class TestHashInvariance:
    """Test hash invariance properties per spec §7.5."""

    def test_whitespace_invariant(self) -> None:
        """Hash should be invariant to whitespace formatting."""
        original = '{"a":1,"b":2}'
        formatted = '{\n  "a": 1,\n  "b": 2\n}'

        hash_original = compute_content_hash(json.loads(original))
        hash_formatted = compute_content_hash(json.loads(formatted))

        assert hash_original == hash_formatted

    def test_key_order_invariant(self) -> None:
        """Hash should be invariant to key order in source JSON."""
        json_1 = {"b": 2, "a": 1}
        json_2 = {"a": 1, "b": 2}

        assert compute_content_hash(json_1) == compute_content_hash(json_2)

    def test_nested_key_order_invariant(self) -> None:
        """Hash should be invariant to nested key order."""
        json_1 = {"outer": {"z": 26, "a": 1}, "inner": {"m": 13}}
        json_2 = {"inner": {"m": 13}, "outer": {"a": 1, "z": 26}}

        assert compute_content_hash(json_1) == compute_content_hash(json_2)

    def test_array_order_preserved(self) -> None:
        """Array order should affect hash (arrays are ordered)."""
        json_1 = {"items": [1, 2, 3]}
        json_2 = {"items": [3, 2, 1]}

        assert compute_content_hash(json_1) != compute_content_hash(json_2)


class TestHashRejection:
    """Test rejection of invalid values per RFC8785."""

    def test_nan_rejected(self) -> None:
        """NaN values should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            canonicalize(float("nan"))

    def test_positive_infinity_rejected(self) -> None:
        """Positive infinity should raise ValueError."""
        with pytest.raises(ValueError, match="Infinity"):
            canonicalize(float("inf"))

    def test_negative_infinity_rejected(self) -> None:
        """Negative infinity should raise ValueError."""
        with pytest.raises(ValueError, match="Infinity"):
            canonicalize(float("-inf"))

    def test_nan_in_nested_rejected(self) -> None:
        """NaN in nested structures should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            canonicalize({"value": float("nan")})

    def test_unsupported_type_rejected(self) -> None:
        """Unsupported types should raise TypeError."""
        with pytest.raises(TypeError):
            canonicalize({"value": object()})


class TestHashRollup:
    """Test hash rollup behavior per spec §15.1."""

    def test_intent_hash_changes_on_candidate_key_mutation(self) -> None:
        """Different candidate_key should produce different intent_hash."""
        identity_a = _make_identity(candidate_key="strat_A")
        identity_b = _make_identity(candidate_key="strat_B")

        hash_a = compute_content_hash(identity_a)
        hash_b = compute_content_hash(identity_b)

        assert hash_a != hash_b

    def test_intent_hash_changes_on_backtest_identity_mutation(self) -> None:
        """Different backtest_identity should produce different hash."""
        identity_a = _make_identity(fidelity_id="high")
        identity_b = _make_identity(fidelity_id="low")

        hash_a = compute_content_hash(identity_a)
        hash_b = compute_content_hash(identity_b)

        assert hash_a != hash_b

    def test_same_content_same_hash(self) -> None:
        """Identical content should produce identical hash."""
        identity_a = _make_identity()
        identity_b = _make_identity()

        assert compute_content_hash(identity_a) == compute_content_hash(identity_b)


class TestHashFormat:
    """Test hash output format."""

    def test_hash_format(self) -> None:
        """Hash should be in format 'sha256:<64 hex chars>'."""
        result = compute_content_hash({"test": "value"})
        assert result.startswith("sha256:")
        hex_part = result[7:]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_deterministic(self) -> None:
        """Same input should always produce same hash."""
        data = {"key": "value", "number": 42}

        hash_1 = compute_content_hash(data)
        hash_2 = compute_content_hash(data)
        hash_3 = compute_content_hash(data)

        assert hash_1 == hash_2 == hash_3


def _make_identity(
    candidate_key: str = "test_strategy",
    fidelity_id: str = "high",
    cost_model_id: str = "fixed_2bps",
    scenario_id: str = "bull_market",
    data_slice_id: str = "2020-2023",
) -> dict:
    """Helper to create test identity artifacts."""
    return {
        "schema_version": "v1",
        "candidate_key": candidate_key,
        "backtest_identity": {
            "fidelity_id": fidelity_id,
            "cost_model_id": cost_model_id,
            "scenario_id": scenario_id,
            "data_slice_id": data_slice_id,
        },
        "hash_set": {
            "intent_hash": "sha256:0" * 64,
            "plan_hash": "sha256:0" * 64,
        },
    }



from __future__ import annotations

import hashlib
import hmac
import json
import platform
from dataclasses import dataclass
from typing import Any

from tests.python._fixtures.hashing import GoldenCase, load_manifest

ADR007_SUITES: tuple[str, ...] = (
    "blake3",
    "hmac_sha256",
    "jcs_sha256",
    "minhash",
    "rabin",
    "simhash",
    "sip24",
    "xxh3",
)
_RABIN_POLY = 0x8000000000000003
_RABIN_MASK = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class ReplayResult:
    suite: str
    case_id: str
    expected_output: str
    actual_output: str

    @property
    def success(self) -> bool:
        return self.expected_output == self.actual_output


def python_toolchain() -> str:
    return f"Python {platform.python_version()}"


def _u32be(value: int) -> bytes:
    return value.to_bytes(4, byteorder="big", signed=False)


def _encode_string(value: str) -> bytes:
    return value.encode("utf-8")


def _build_composite_preimage(domain: str, *fields: bytes) -> bytes:
    payload = _encode_string(domain) + len(fields).to_bytes(8, byteorder="big", signed=False)
    for field in fields:
        payload += len(field).to_bytes(8, byteorder="big", signed=False) + field
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _build_rabin_reduce_table(poly: int) -> list[int]:
    table: list[int] = []
    for byte_value in range(256):
        entry = byte_value
        for _ in range(56):
            carry = (entry >> 62) & 1
            entry = ((entry << 1) & _RABIN_MASK) ^ (poly & _RABIN_MASK if carry else 0)
        table.append(entry & _RABIN_MASK)
    return table


def _build_rabin_pop_table(poly: int, window_size: int) -> list[int]:
    table: list[int] = []
    shift_count = max(window_size * 8, 1)
    for byte_value in range(256):
        entry = byte_value
        for _ in range(shift_count):
            carry = (entry >> 62) & 1
            entry = ((entry << 1) & _RABIN_MASK) ^ (poly & _RABIN_MASK if carry else 0)
        table.append(entry & _RABIN_MASK)
    return table


def _rabin_fingerprint(data: bytes, *, window_size: int) -> str:
    reduce_table = _build_rabin_reduce_table(_RABIN_POLY)
    pop_table = _build_rabin_pop_table(_RABIN_POLY, window_size)
    state = 0
    ring = bytearray(window_size)
    position = 0
    for byte_value in data:
        outgoing = ring[position]
        ring[position] = byte_value & 0xFF
        position = (position + 1) % window_size
        high_byte = (state >> 55) & 0xFF
        state = (
            reduce_table[high_byte]
            ^ ((state << 8) & _RABIN_MASK)
            ^ (byte_value & 0xFF)
            ^ pop_table[outgoing]
        ) & _RABIN_MASK
    return format(state & 0xFFFFFFFFFFFFFFFF, "016x")


def case_by_id(suite: str, case_id: str) -> GoldenCase:
    manifest = load_manifest(suite)
    for case in manifest.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Missing case {case_id!r} in suite {suite!r}.")


def expected_output_for_case(suite: str, case: GoldenCase) -> str:
    if suite in {"blake3", "hmac_sha256", "rabin", "sip24", "xxh3"}:
        key = "expected_digest_hex" if suite != "rabin" else "expected_fingerprint_hex"
        return str(case.metadata[key])
    payload = case.read_json()
    if suite == "jcs_sha256":
        return str(payload["expected_digest_hex"])
    if suite == "minhash":
        return str(payload["expected_sip_key_hex"])
    if suite == "simhash":
        return str(payload["expected_projection_seed_hex"])
    raise ValueError(f"Unsupported suite {suite!r}.")


def replay_case(suite: str, case: GoldenCase) -> str:
    if suite == "blake3":
        return hashlib.blake2b(
            case.read_bytes(),
            digest_size=32,
            person=b"mm-b3-fallback",
        ).hexdigest()
    if suite == "hmac_sha256":
        master_seed = bytes.fromhex(str(case.metadata["master_seed_hex"]))
        return hmac.new(master_seed, case.read_bytes(), hashlib.sha256).hexdigest()
    if suite == "jcs_sha256":
        payload = case.read_json()["payload"]
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if suite == "minhash":
        payload = case.read_json()
        master_seed = bytes.fromhex(payload["master_seed_hex"])
        ctx = b"mm/minhash/v1" + _u32be(int(payload["hash_family_index"]))
        return hmac.new(master_seed, ctx, hashlib.sha256).digest()[:16].hex()
    if suite == "rabin":
        return _rabin_fingerprint(case.read_bytes(), window_size=int(case.metadata["window_size"]))
    if suite == "simhash":
        payload = case.read_json()
        master_seed = bytes.fromhex(payload["master_seed_hex"])
        ctx = b"mm/simhash/v1" + _u32be(int(payload["dim"])) + _u32be(int(payload["bit_index"]))
        return hashlib.sha256(master_seed + ctx).hexdigest()
    if suite == "sip24":
        namespace_bytes = _encode_string(str(case.metadata["namespace"]))
        preimage = _build_composite_preimage("mm/sip/v1", namespace_bytes, case.read_bytes())
        return hashlib.blake2b(
            preimage,
            key=bytes.fromhex(str(case.metadata["key_hex"])),
            digest_size=8,
            person=b"mm/sip24",
        ).hexdigest()
    if suite == "xxh3":
        return hashlib.blake2b(
            case.read_bytes(),
            digest_size=16,
            person=b"mm-xxh3-128",
        ).hexdigest()
    raise ValueError(f"Unsupported suite {suite!r}.")


def replay_suite(suite: str) -> tuple[ReplayResult, ...]:
    manifest = load_manifest(suite)
    results: list[ReplayResult] = []
    for case in manifest.cases:
        results.append(
            ReplayResult(
                suite=suite,
                case_id=case.case_id,
                expected_output=expected_output_for_case(suite, case),
                actual_output=replay_case(suite, case),
            )
        )
    return tuple(results)


def replay_all_suites() -> tuple[ReplayResult, ...]:
    results: list[ReplayResult] = []
    for suite in ADR007_SUITES:
        results.extend(replay_suite(suite))
    return tuple(results)


def replay_summary() -> dict[str, Any]:
    results = replay_all_suites()
    suites = sorted({result.suite for result in results})
    version_info = platform.python_version_tuple()
    return {
        "language": f"python-{version_info[0]}.{version_info[1]}",
        "toolchain": python_toolchain(),
        "suite_count": len(suites),
        "case_count": len(results),
        "success": all(result.success for result in results),
        "suites": suites,
        "cases": [
            {
                "suite": result.suite,
                "case_id": result.case_id,
                "expected_output": result.expected_output,
                "actual_output": result.actual_output,
                "success": result.success,
            }
            for result in results
        ],
    }

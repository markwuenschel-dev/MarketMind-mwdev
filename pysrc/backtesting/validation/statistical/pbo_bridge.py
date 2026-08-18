from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pysrc.backtesting.validation.statistical.pbo import PBODataError

CANONICAL_PBO_MODE = "path_pairs"
_IN_SAMPLE_SCORE_KEYS = ("in_sample_score", "in_sample_net_sharpe")
_OUT_OF_SAMPLE_SCORE_KEYS = ("out_of_sample_score", "out_of_sample_net_sharpe")


def build_pbo_path_pairs(source: Any) -> list[dict[str, Any]]:
    """Normalize evaluated CPCV outputs into canonical validator-side path_pairs."""
    if isinstance(source, Mapping):
        for key in ("path_pairs", "cpcv_evaluations", "evaluations", "records"):
            if key in source:
                return build_pbo_path_pairs(source[key])
        raise PBODataError(
            "PBO source mapping must contain path_pairs, cpcv_evaluations, evaluations, or records"
        )

    if not isinstance(source, Iterable) or isinstance(source, (str, bytes)):
        raise PBODataError("PBO source must be an iterable of mappings or a mapping wrapper")

    items = list(source)
    if not items:
        raise PBODataError("PBO source must be non-empty")
    if not all(isinstance(item, Mapping) for item in items):
        raise PBODataError("PBO source items must be mappings")

    if _looks_like_path_pairs(items):
        return [_normalize_path_pair(item, index) for index, item in enumerate(items)]
    return _path_pairs_from_records(items)


def _looks_like_path_pairs(items: Sequence[Mapping[str, Any]]) -> bool:
    return all("in_sample_scores" in item and "out_of_sample_scores" in item for item in items)


def _normalize_path_pair(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    in_sample_scores = item.get("in_sample_scores")
    out_of_sample_scores = item.get("out_of_sample_scores")
    if in_sample_scores is None or out_of_sample_scores is None:
        raise PBODataError("Each path pair must contain in_sample_scores and out_of_sample_scores")
    return {
        "path_id": str(item.get("path_id", index)),
        "in_sample_scores": list(in_sample_scores),
        "out_of_sample_scores": list(out_of_sample_scores),
    }


def _path_pairs_from_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()

    for record in records:
        path_id = _require_identifier(record, "path_id")
        trial_id = _require_identifier(record, "trial_id")
        pair_id = (trial_id, path_id)
        if pair_id in seen_pairs:
            raise PBODataError(f"Duplicate (trial_id, path_id) pair encountered: {pair_id!r}")
        seen_pairs.add(pair_id)
        grouped[path_id].append(
            (
                trial_id,
                _extract_score(record, _IN_SAMPLE_SCORE_KEYS, "in-sample"),
                _extract_score(record, _OUT_OF_SAMPLE_SCORE_KEYS, "out-of-sample"),
            )
        )

    expected_trial_ids: list[str] | None = None
    path_pairs: list[dict[str, Any]] = []
    for path_id in sorted(grouped):
        rows = sorted(grouped[path_id], key=lambda row: row[0])
        trial_ids = [trial_id for trial_id, _, _ in rows]
        if expected_trial_ids is None:
            expected_trial_ids = trial_ids
        elif trial_ids != expected_trial_ids:
            raise PBODataError(
                "CPCV evaluations must form a complete rectangular trial/path score surface"
            )
        path_pairs.append(
            {
                "path_id": path_id,
                "in_sample_scores": [in_sample_score for _, in_sample_score, _ in rows],
                "out_of_sample_scores": [out_of_sample_score for _, _, out_of_sample_score in rows],
            }
        )

    return path_pairs


def _require_identifier(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if value is None:
        raise PBODataError(f"CPCV evaluation record missing required field: {field_name}")
    return str(value)


def _extract_score(record: Mapping[str, Any], keys: tuple[str, ...], label: str) -> float:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return float(value)

    net_sharpe = record.get("net_sharpe")
    if isinstance(net_sharpe, Mapping):
        if label == "in-sample" and "in_sample" in net_sharpe:
            return float(net_sharpe["in_sample"])
        if label == "out-of-sample" and "out_of_sample" in net_sharpe:
            return float(net_sharpe["out_of_sample"])

    raise PBODataError(
        f"CPCV evaluation record missing {label} net_sharpe score; "
        f"expected one of {keys!r} or a net_sharpe mapping"
    )

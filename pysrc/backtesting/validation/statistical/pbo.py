from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None


class PBOError(Exception):
    pass


class PBODataError(PBOError):
    pass


class PBOComputationError(PBOError):
    pass


@dataclass(frozen=True)
class PBOConfig:
    warn_threshold: float = 0.40
    fail_threshold: float = 0.50
    min_trials: int = 2
    min_paths: int = 2
    include_diagnostics: bool = False
    score_basis: str = "net_sharpe"

    def validate(self) -> None:
        if not (0.0 <= self.warn_threshold <= 1.0):
            raise PBODataError("warn_threshold must be in [0, 1]")
        if not (0.0 <= self.fail_threshold <= 1.0):
            raise PBODataError("fail_threshold must be in [0, 1]")
        if self.warn_threshold > self.fail_threshold:
            raise PBODataError("warn_threshold cannot exceed fail_threshold")
        if self.min_trials < 2:
            raise PBODataError("min_trials must be >= 2")
        if self.min_paths < 2:
            raise PBODataError("min_paths must be >= 2")
        if not self.score_basis or not isinstance(self.score_basis, str):
            raise PBODataError("score_basis must be a non-empty string")


def _as_2d_float64(x: Any, *, name: str) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float64, copy=False)
    elif isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=np.float64, copy=False)
    elif pl is not None and isinstance(x, pl.DataFrame):
        arr = x.to_numpy().astype(np.float64, copy=False)
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=np.float64)
    else:
        raise PBODataError(f"Unsupported type for {name}: {type(x).__name__}")

    if arr.ndim != 2:
        raise PBODataError(f"{name} must be 2-D")
    if arr.size == 0:
        raise PBODataError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise PBODataError(f"{name} contains NaN or Inf")
    return arr


def _average_rank_percentile(values: np.ndarray, idx: int) -> float:
    n = int(values.size)
    if n < 2:
        raise PBODataError("At least 2 trials are required per path")

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        v = values[order[i]]
        while j < n and values[order[j]] == v:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return float(ranks[idx] / (n + 1.0))


def _safe_logit(p: float, eps: float = 1e-12) -> float:
    p_clipped = min(max(p, eps), 1.0 - eps)
    return float(math.log(p_clipped / (1.0 - p_clipped)))


def _gate_result_from_value(value: float, cfg: PBOConfig) -> str:
    if value > cfg.fail_threshold:
        return "FAIL"
    if value > cfg.warn_threshold:
        return "WARN"
    return "PASS"


def _coerce_records_df(records: Any) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif pl is not None and isinstance(records, pl.DataFrame):
        df = records.to_pandas()
    elif isinstance(records, Iterable) and not isinstance(records, (str, bytes, dict)):
        df = pd.DataFrame(list(records))
    else:
        raise PBODataError("records must be a DataFrame or iterable of mappings")

    required = {"trial_id", "path_id", "in_sample_score", "out_of_sample_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise PBODataError(f"records missing required columns: {missing}")
    if df.empty:
        raise PBODataError("records must be non-empty")

    for col in ("trial_id", "path_id"):
        if df[col].isna().any():
            raise PBODataError(f"{col} contains nulls")

    for col in ("in_sample_score", "out_of_sample_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise PBODataError(f"{col} contains non-numeric or null values")

    if df.duplicated(subset=["trial_id", "path_id"]).any():
        raise PBODataError("records contain duplicate (trial_id, path_id) pairs")

    return df


def _matrices_from_records(records: Any) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    df = _coerce_records_df(records)
    trial_ids = [str(x) for x in sorted(df["trial_id"].unique().tolist())]
    path_ids = [str(x) for x in sorted(df["path_id"].unique().tolist())]

    is_pivot = df.pivot(index="trial_id", columns="path_id", values="in_sample_score").reindex(
        index=trial_ids, columns=path_ids
    )
    oos_pivot = df.pivot(index="trial_id", columns="path_id", values="out_of_sample_score").reindex(
        index=trial_ids, columns=path_ids
    )

    if is_pivot.isna().any().any() or oos_pivot.isna().any().any():
        raise PBODataError("records do not form a complete rectangular trial/path matrix")

    return (
        is_pivot.to_numpy(dtype=np.float64),
        oos_pivot.to_numpy(dtype=np.float64),
        trial_ids,
        path_ids,
    )


def _coerce_path_pairs(path_pairs: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not isinstance(path_pairs, Iterable) or isinstance(path_pairs, (str, bytes, dict)):
        raise PBODataError("path_pairs must be an iterable of mappings")

    pairs = list(path_pairs)
    if not pairs:
        raise PBODataError("path_pairs must be non-empty")

    path_ids: list[str] = []
    is_cols: list[np.ndarray] = []
    oos_cols: list[np.ndarray] = []
    expected_n_trials: int | None = None

    for i, item in enumerate(pairs):
        if not isinstance(item, Mapping):
            raise PBODataError("each path pair must be a mapping")

        if "in_sample_scores" not in item or "out_of_sample_scores" not in item:
            raise PBODataError(
                "each path pair must contain in_sample_scores and out_of_sample_scores"
            )

        path_id = str(item.get("path_id", i))
        is_col = np.asarray(item["in_sample_scores"], dtype=np.float64)
        oos_col = np.asarray(item["out_of_sample_scores"], dtype=np.float64)

        if is_col.ndim != 1 or oos_col.ndim != 1:
            raise PBODataError("path pair score vectors must be 1-D")
        if is_col.size == 0 or oos_col.size == 0:
            raise PBODataError("path pair score vectors must be non-empty")
        if is_col.size != oos_col.size:
            raise PBODataError("in_sample_scores and out_of_sample_scores must have equal length")
        if not np.isfinite(is_col).all() or not np.isfinite(oos_col).all():
            raise PBODataError("path pair score vectors contain NaN or Inf")

        if expected_n_trials is None:
            expected_n_trials = int(is_col.size)
        elif int(is_col.size) != expected_n_trials:
            raise PBODataError("all path pairs must have the same trial count")

        path_ids.append(path_id)
        is_cols.append(is_col)
        oos_cols.append(oos_col)

    in_sample = np.column_stack(is_cols)
    out_of_sample = np.column_stack(oos_cols)
    return in_sample, out_of_sample, path_ids


def compute_pbo_from_matrices(
    in_sample_scores: Any,
    out_of_sample_scores: Any,
    *,
    config: PBOConfig | None = None,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    cfg = config or PBOConfig()
    cfg.validate()

    in_sample = _as_2d_float64(in_sample_scores, name="in_sample_scores")
    out_of_sample = _as_2d_float64(out_of_sample_scores, name="out_of_sample_scores")

    if in_sample.shape != out_of_sample.shape:
        raise PBODataError(
            f"in_sample_scores shape {in_sample.shape} must match "
            f"out_of_sample_scores shape {out_of_sample.shape}"
        )

    n_trials, n_paths = (int(in_sample.shape[0]), int(in_sample.shape[1]))
    if n_trials < cfg.min_trials:
        raise PBODataError(f"n_trials={n_trials} is below min_trials={cfg.min_trials}")
    if n_paths < cfg.min_paths:
        raise PBODataError(f"n_paths={n_paths} is below min_paths={cfg.min_paths}")

    selected_trial_indices = np.empty(n_paths, dtype=np.int64)
    selected_in_sample_scores = np.empty(n_paths, dtype=np.float64)
    selected_out_of_sample_scores = np.empty(n_paths, dtype=np.float64)
    relative_rank_percentiles = np.empty(n_paths, dtype=np.float64)
    overfit_flags = np.empty(n_paths, dtype=bool)

    for path_idx in range(n_paths):
        is_col = in_sample[:, path_idx]
        oos_col = out_of_sample[:, path_idx]

        selected_idx = int(np.argmax(is_col) if score_higher_is_better else np.argmin(is_col))
        selected_trial_indices[path_idx] = selected_idx
        selected_in_sample_scores[path_idx] = float(is_col[selected_idx])
        selected_out_of_sample_scores[path_idx] = float(oos_col[selected_idx])

        rank_values = oos_col if score_higher_is_better else -oos_col
        rank_pct = _average_rank_percentile(rank_values, selected_idx)
        relative_rank_percentiles[path_idx] = rank_pct
        overfit_flags[path_idx] = bool(rank_pct < 0.5)

    value = float(np.mean(overfit_flags))
    gate_result = _gate_result_from_value(value, cfg)

    payload: dict[str, Any] = {
        "value": value,
        "threshold": cfg.fail_threshold,
        "warn_threshold": cfg.warn_threshold,
        "gate_result": gate_result,
        "method": "p_oos_rank_lt_0_5",
        "score_basis": cfg.score_basis,
        "score_direction": "higher_is_better" if score_higher_is_better else "lower_is_better",
        "n_trials": n_trials,
        "n_paths": n_paths,
        "share_oos_rank_below_half": value,
        "median_oos_rank_percentile": float(np.median(relative_rank_percentiles)),
        "mean_oos_rank_percentile": float(np.mean(relative_rank_percentiles)),
    }

    if cfg.include_diagnostics:
        payload["relative_rank_percentiles"] = relative_rank_percentiles.tolist()
        payload["selected_trial_indices"] = selected_trial_indices.tolist()
        payload["selected_in_sample_scores"] = selected_in_sample_scores.tolist()
        payload["selected_out_of_sample_scores"] = selected_out_of_sample_scores.tolist()
        payload["logits"] = [_safe_logit(float(p)) for p in relative_rank_percentiles]

    return payload


def compute_pbo_from_records(
    records: Any,
    *,
    config: PBOConfig | None = None,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    in_sample, out_of_sample, trial_ids, path_ids = _matrices_from_records(records)
    payload = compute_pbo_from_matrices(
        in_sample,
        out_of_sample,
        config=config,
        score_higher_is_better=score_higher_is_better,
    )
    if (config or PBOConfig()).include_diagnostics:
        payload["trial_ids"] = trial_ids
        payload["path_ids"] = path_ids
    return payload


def compute_pbo_from_path_pairs(
    path_pairs: Any,
    *,
    config: PBOConfig | None = None,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    in_sample, out_of_sample, path_ids = _coerce_path_pairs(path_pairs)
    payload = compute_pbo_from_matrices(
        in_sample,
        out_of_sample,
        config=config,
        score_higher_is_better=score_higher_is_better,
    )
    if (config or PBOConfig()).include_diagnostics:
        payload["path_ids"] = path_ids
    return payload


def compute_pbo(
    data: Any,
    *,
    mode: str = "records",
    config: PBOConfig | None = None,
    score_higher_is_better: bool = True,
    out_of_sample_scores: Any | None = None,
) -> dict[str, Any]:
    if mode == "matrices":
        if out_of_sample_scores is None:
            raise PBODataError("out_of_sample_scores is required when mode='matrices'")
        return compute_pbo_from_matrices(
            data,
            out_of_sample_scores,
            config=config,
            score_higher_is_better=score_higher_is_better,
        )
    if mode == "records":
        return compute_pbo_from_records(
            data,
            config=config,
            score_higher_is_better=score_higher_is_better,
        )
    if mode == "path_pairs":
        return compute_pbo_from_path_pairs(
            data,
            config=config,
            score_higher_is_better=score_higher_is_better,
        )
    raise PBODataError("mode must be one of {'matrices', 'records', 'path_pairs'}")

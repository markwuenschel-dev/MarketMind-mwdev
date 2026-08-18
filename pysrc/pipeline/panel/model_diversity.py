"""Model-matrix diversity diagnostics for Gate 2 stop/go decisions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pysrc.pipeline.panel.panel_keys import normalize_key_columns
from pysrc.pipeline.panel.panel_targets import CANONICAL_PANEL_KEYS

_REDUNDANCY_CORR_THRESHOLD = 0.99
_REDUNDANCY_RANK_THRESHOLD = 0.99
_REDUNDANCY_TOPK_OVERLAP_THRESHOLD = 0.95
_SOFT_CLUSTER_THRESHOLDS = (0.80, 0.90)
_ACTIVE_PREDICTION_STD_EPS = 1e-8
ALIGNMENT_KEY_COLS = ("date", "instrument", "interval", "fold_id")
DEFAULT_MAX_SAMPLE_ROWS = 2_000_000
_KEY_FIELD_SEP = "\x1f"


def _pack_alignment_key(date: str, instrument: str, interval: str, fold_id: str) -> str:
    return f"{date}{_KEY_FIELD_SEP}{instrument}{_KEY_FIELD_SEP}{interval}{_KEY_FIELD_SEP}{fold_id}"


def _rank_alignment_key(packed_key: str, *, random_seed: int) -> int:
    digest = hashlib.sha256(f"{random_seed}:aligned:{packed_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True, slots=True)
class AlignedKeySelection:
    selected_keys: frozenset[str]
    total_aligned_keys: int
    target_aligned_keys: int
    stratum_count: int


def select_aligned_prediction_keys(
    predictions_path: Path,
    *,
    random_seed: int,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    expected_model_count: int | None = None,
    batch_size: int = 250_000,
) -> AlignedKeySelection:
    """Deterministically select shared date/instrument/interval/fold keys for all models."""
    import pyarrow.parquet as pq

    columns = ["date", "instrument", "interval", "fold_id", "model_id"]
    parquet = pq.ParquetFile(predictions_path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    stratum_keys: dict[tuple[str, str, str, str], dict[str, None]] = defaultdict(dict)
    observed_models: set[str] = set()
    seen_aligned_keys: set[str] = set()

    for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
        frame = batch.to_pandas()
        if "interval" not in frame.columns:
            frame = frame.copy()
            frame["interval"] = "1d"
        frame["date"] = frame["date"].astype(str)
        frame["instrument"] = frame["instrument"].astype(str)
        frame["interval"] = frame["interval"].astype(str)
        frame["fold_id"] = frame["fold_id"].astype(str)
        observed_models.update(frame["model_id"].astype(str).unique().tolist())
        for date, instrument, interval, fold_id in zip(
            frame["date"].to_numpy(),
            frame["instrument"].to_numpy(),
            frame["interval"].to_numpy(),
            frame["fold_id"].to_numpy(),
            strict=True,
        ):
            packed = _pack_alignment_key(str(date), str(instrument), str(interval), str(fold_id))
            if packed in seen_aligned_keys:
                continue
            seen_aligned_keys.add(packed)
            stratum = (str(date), str(interval), str(fold_id), "all")
            stratum_keys[stratum][packed] = None

    model_count = expected_model_count or max(len(observed_models), 1)
    total_aligned_keys = sum(len(keys) for keys in stratum_keys.values())
    target_aligned_keys = max(1, min(total_aligned_keys, max_sample_rows // model_count))

    selected: set[str] = set()
    for stratum in sorted(stratum_keys):
        key_map = stratum_keys[stratum]
        stratum_quota = max(
            1, int(round(target_aligned_keys * len(key_map) / max(total_aligned_keys, 1)))
        )
        ranked = sorted(
            key_map,
            key=lambda packed: (_rank_alignment_key(packed, random_seed=random_seed), packed),
        )
        selected.update(ranked[:stratum_quota])

    return AlignedKeySelection(
        selected_keys=frozenset(selected),
        total_aligned_keys=total_aligned_keys,
        target_aligned_keys=target_aligned_keys,
        stratum_count=len(stratum_keys),
    )


def build_diagnostic_coverage_report(
    frame: pd.DataFrame,
    *,
    expected_model_count: int,
    expected_fold_count: int,
) -> dict[str, object]:
    """Assert explicit diagnostic coverage across models and folds."""
    observed_models = sorted(frame["model_id"].astype(str).unique().tolist())
    observed_folds = (
        sorted(frame["fold_id"].astype(str).unique().tolist()) if "fold_id" in frame.columns else []
    )
    per_model = {
        str(model_id): int(count)
        for model_id, count in frame.groupby("model_id", sort=True).size().items()
    }
    per_fold = (
        {
            str(fold_id): int(count)
            for fold_id, count in frame.groupby("fold_id", sort=True).size().items()
        }
        if "fold_id" in frame.columns
        else {}
    )
    combo_frame = (
        frame.groupby(["model_id", "fold_id"], sort=True).size().reset_index(name="count")
        if "fold_id" in frame.columns
        else frame.groupby(["model_id"], sort=True).size().reset_index(name="count")
    )
    observed_combo_count = int(len(combo_frame))
    expected_combo_count = int(expected_model_count * expected_fold_count)
    row_counts_equal = len(set(per_model.values())) == 1 if per_model else False
    coverage_satisfied = (
        len(observed_models) == expected_model_count
        and observed_combo_count == expected_combo_count
        and row_counts_equal
        and all(count > 0 for count in per_model.values())
    )
    return {
        "schema_version": "diagnostic_coverage.v1",
        "expected_model_count": expected_model_count,
        "observed_model_count": len(observed_models),
        "expected_model_fold_combinations": expected_combo_count,
        "observed_model_fold_combinations": observed_combo_count,
        "expected_fold_count": expected_fold_count,
        "observed_fold_count": len(observed_folds),
        "observed_models": observed_models,
        "observed_folds": observed_folds,
        "per_model_sampled_row_count": per_model,
        "per_fold_sampled_row_count": per_fold,
        "per_model_fold_sampled_row_count": [
            {
                "model_id": str(row["model_id"]),
                "fold_id": str(row["fold_id"]),
                "row_count": int(row["count"]),
            }
            for _, row in combo_frame.iterrows()
        ],
        "coverage_satisfied": coverage_satisfied,
    }


def _distribution_summary(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _positive_connected_clusters(
    models: list[str],
    pairs: list[dict[str, object]],
    *,
    threshold: float,
    metric_key: str = "value",
) -> list[list[str]]:
    parent = {model: model for model in models}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for item in pairs:
        value = float(item[metric_key])
        if value >= threshold:
            union(str(item["model_a"]), str(item["model_b"]))

    clusters: dict[str, list[str]] = {}
    for model in models:
        root = find(model)
        clusters.setdefault(root, []).append(model)
    return [sorted(cluster) for cluster in clusters.values() if len(cluster) > 1]


def _anti_correlated_pairs(
    pairs: list[dict[str, object]],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    anti: list[dict[str, object]] = []
    for item in pairs:
        value = float(item["value"])
        if value <= -threshold:
            anti.append(
                {
                    "model_a": str(item["model_a"]),
                    "model_b": str(item["model_b"]),
                    "value": value,
                    "metric": "prediction_correlation",
                }
            )
    return anti


def _router_representative_children(
    models: list[str],
    clusters: list[list[str]],
    *,
    fold_spearman_ic: dict[str, dict[str, float]],
) -> list[str]:
    clustered = {model for cluster in clusters for model in cluster}
    representatives: list[str] = []
    overall_ic = {
        model_id: float(
            np.mean([fold_map.get(model_id, 0.0) for fold_map in fold_spearman_ic.values()])
        )
        if fold_spearman_ic
        else 0.0
        for model_id in models
    }
    for cluster in clusters:
        representatives.append(max(cluster, key=lambda model_id: overall_ic.get(model_id, 0.0)))
    for model_id in models:
        if model_id not in clustered:
            representatives.append(model_id)
    return sorted(set(representatives))


def _connected_clusters(
    models: list[str],
    pairs: list[dict[str, object]],
    *,
    threshold: float,
    metric_key: str = "value",
) -> list[list[str]]:
    parent = {model: model for model in models}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for item in pairs:
        value = float(item[metric_key])
        if abs(value) >= threshold:
            union(str(item["model_a"]), str(item["model_b"]))

    clusters: dict[str, list[str]] = {}
    for model in models:
        root = find(model)
        clusters.setdefault(root, []).append(model)
    return [sorted(cluster) for cluster in clusters.values() if len(cluster) > 1]


def _soft_redundancy_report(
    *,
    pred_pairs: list[dict[str, object]],
    rank_pairs: list[dict[str, object]],
    models: list[str],
) -> dict[str, object]:
    clusters: dict[str, object] = {}
    for threshold in _SOFT_CLUSTER_THRESHOLDS:
        key = f"threshold_{int(threshold * 100)}"
        clusters[key] = {
            "positive_prediction_correlation_clusters": _positive_connected_clusters(
                models,
                pred_pairs,
                threshold=threshold,
            ),
            "positive_rank_correlation_clusters": _positive_connected_clusters(
                models,
                rank_pairs,
                threshold=threshold,
            ),
            "anti_correlated_prediction_pairs": _anti_correlated_pairs(
                pred_pairs,
                threshold=threshold,
            ),
        }
    return clusters


def _per_date_metric_summaries(
    frame: pd.DataFrame,
    *,
    group_col: str | None = None,
    top_k: int = 20,
) -> dict[str, object]:
    """Summarize cross-sectional prediction/rank/top-k metrics by date."""
    models = sorted(frame["model_id"].astype(str).unique())
    empty_summary = {
        "prediction_correlation": _distribution_summary([]),
        "rank_correlation": _distribution_summary([]),
        "top_k_jaccard": _distribution_summary([]),
    }
    if len(models) < 2:
        if group_col is None or group_col not in frame.columns:
            return {"overall": empty_summary, "by_group": {}}
        by_group = {
            str(group_value): dict(empty_summary)
            for group_value, _ in frame.groupby(group_col, sort=True, observed=True)
        }
        return {"overall": empty_summary, "by_group": by_group}

    def summarize(subframe: pd.DataFrame) -> dict[str, object]:
        pred_values: list[float] = []
        rank_values: list[float] = []
        topk_values: list[float] = []
        for _, day_frame in subframe.groupby("date", sort=True):
            wide = day_frame.pivot_table(
                index="instrument",
                columns="model_id",
                values="prediction",
                aggfunc="mean",
            )
            if wide.shape[0] < 3 or wide.shape[1] < 2:
                continue
            pred_corr = wide.corr(method="pearson")
            rank_corr = wide.corr(method="spearman")
            pred_mean = _mean_off_diagonal(pred_corr)
            rank_mean = _mean_off_diagonal(rank_corr)
            if pred_mean is not None:
                pred_values.append(pred_mean)
            if rank_mean is not None:
                rank_values.append(rank_mean)
            top_sets: dict[str, set[str]] = {}
            for model_id, model_rows in day_frame.groupby("model_id", sort=True):
                ranked = model_rows.sort_values("prediction", ascending=False).head(top_k)
                top_sets[str(model_id)] = set(ranked["instrument"].astype(str))
            model_list = sorted(top_sets)
            for i, a in enumerate(model_list):
                for b in model_list[i + 1 :]:
                    sa, sb = top_sets[a], top_sets[b]
                    if not sa and not sb:
                        continue
                    union = sa | sb
                    topk_values.append(len(sa & sb) / max(len(union), 1))
        return {
            "prediction_correlation": _distribution_summary(pred_values),
            "rank_correlation": _distribution_summary(rank_values),
            "top_k_jaccard": _distribution_summary(topk_values),
        }

    overall = summarize(frame)
    if group_col is None or group_col not in frame.columns:
        return {"overall": overall, "by_group": {}}
    by_group = {
        str(group_value): summarize(group)
        for group_value, group in frame.groupby(group_col, sort=True, observed=True)
    }
    return {"overall": overall, "by_group": by_group}


def _regime_documentation(
    panel: pd.DataFrame,
    target_column: str,
    regimes: pd.DataFrame,
) -> dict[str, object]:
    frame = panel.copy()
    frame["date"] = frame["date"].astype(str)
    date_counts = regimes["regime"].value_counts().to_dict() if not regimes.empty else {}
    row_counts: dict[str, int] = {}
    if not regimes.empty and target_column in frame.columns:
        merged = frame.merge(regimes, on="date", how="left")
        for regime_value, group in merged.groupby("regime", dropna=False, observed=True):
            row_counts[str(regime_value)] = int(len(group))
    return {
        "classification": "ex_post_realized_return_diagnostic",
        "eligible_as_router_feature": False,
        "pit_status": "not_point_in_time_state",
        "definition": (
            "Ex-post diagnostic only: per-date cross-sectional mean of realized target; "
            "down < -1e-4, flat within [-1e-4, 1e-4], up > 1e-4."
        ),
        "bins": ["down", "flat", "up"],
        "date_counts": {str(k): int(v) for k, v in date_counts.items()},
        "row_counts": row_counts,
    }


def _child_count_summary(
    *,
    models: list[str],
    redundant_near_duplicate: list[dict[str, object]],
    dispersion: pd.DataFrame,
    fold_spearman_ic: dict[str, dict[str, float]],
    near_duplicate_clusters: list[list[str]],
) -> dict[str, int | list[str]]:
    representatives = _router_representative_children(
        models,
        near_duplicate_clusters,
        fold_spearman_ic=fold_spearman_ic,
    )

    active: list[str] = []
    for model_id in models:
        row = dispersion.loc[dispersion["model_id"].astype(str) == model_id]
        if row.empty:
            continue
        std = float(row.iloc[0]["prediction_std"])
        if np.isfinite(std) and std > _ACTIVE_PREDICTION_STD_EPS:
            active.append(str(model_id))

    positive_all: list[str] = []
    positive_any: list[str] = []
    if fold_spearman_ic:
        for model_id in models:
            fold_values = [
                float(fold_map.get(str(model_id), 0.0)) for fold_map in fold_spearman_ic.values()
            ]
            if fold_values and all(value > 0.0 for value in fold_values):
                positive_all.append(str(model_id))
            if any(value > 0.0 for value in fold_values):
                positive_any.append(str(model_id))

    eligible = [m for m in representatives if m in active and m in positive_any]
    return {
        "nonredundant_child_count": len(representatives),
        "active_prediction_child_count": len(active),
        "positive_all_folds_count": len(positive_all),
        "positive_any_fold_count": len(positive_any),
        "eligible_router_child_count": len(eligible),
        "router_representative_children": representatives,
        "meaningful_child_count": len(representatives),
    }


def _wide_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    index_cols = [column for column in (*CANONICAL_PANEL_KEYS,) if column in frame.columns]
    if len(index_cols) < 2:
        index_cols = ["date", "instrument"]
    return frame.pivot_table(
        index=index_cols,
        columns="model_id",
        values="prediction",
        aggfunc="mean",
    )


def _pairwise_matrix(corr: pd.DataFrame) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    models = list(corr.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            value = float(corr.loc[a, b])
            if np.isfinite(value):
                pairs.append({"model_a": a, "model_b": b, "value": value})
    return pairs


def _mean_off_diagonal(corr: pd.DataFrame) -> float | None:
    n = corr.shape[0]
    if n < 2:
        return None
    values = corr.to_numpy()
    return float((values.sum() - n) / max(n * (n - 1), 1))


def _top_k_overlap(
    frame: pd.DataFrame,
    *,
    top_k: int,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    models = sorted(frame["model_id"].astype(str).unique())
    if len(models) < 2:
        return pairs

    top_sets: dict[tuple[str, str], set[str]] = {}
    for (date, model_id), group in frame.groupby(["date", "model_id"], sort=True):
        ranked = group.sort_values("prediction", ascending=False).head(top_k)
        top_sets[(str(date), str(model_id))] = set(ranked["instrument"].astype(str))

    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            overlaps: list[float] = []
            for date in sorted(frame["date"].astype(str).unique()):
                key_a = (date, a)
                key_b = (date, b)
                if key_a not in top_sets or key_b not in top_sets:
                    continue
                sa, sb = top_sets[key_a], top_sets[key_b]
                if not sa and not sb:
                    continue
                union = sa | sb
                overlaps.append(len(sa & sb) / max(len(union), 1))
            if overlaps:
                pairs.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "mean_topk_jaccard": float(np.mean(overlaps)),
                        "top_k": top_k,
                    }
                )
    return pairs


def _residual_frame(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    panel_frame = panel.copy()
    pred_frame = predictions.copy()
    pred_frame = pred_frame.drop(columns=["realized"], errors="ignore")
    if "interval" not in panel_frame.columns:
        panel_frame["interval"] = "1d"
    if "interval" not in pred_frame.columns:
        pred_frame["interval"] = "1d"
    keys = [
        column
        for column in CANONICAL_PANEL_KEYS
        if column in pred_frame.columns and column in panel_frame.columns
    ]
    if len(keys) < 2:
        keys = ["date", "instrument"]
    realized = normalize_key_columns(panel_frame[[*keys, target_column]])
    merged = normalize_key_columns(pred_frame).merge(realized, on=keys, how="inner")
    merged["residual"] = merged["prediction"].astype(float) - merged[target_column].astype(float)
    return merged


def _regime_labels(panel: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Date-level coarse regime from cross-sectional mean realized return."""

    frame = panel.copy()
    frame["date"] = frame["date"].astype(str)
    if target_column not in frame.columns:
        return pd.DataFrame(columns=["date", "regime"])
    daily = frame.groupby("date")[target_column].mean().reset_index()
    daily["regime"] = pd.cut(
        daily[target_column].astype(float),
        bins=[-float("inf"), -1e-4, 1e-4, float("inf")],
        labels=["down", "flat", "up"],
    )
    return daily[["date", "regime"]]


_MIN_CROSS_SECTIONAL_INSTRUMENTS = 3


def _utility_by_group(
    frame: pd.DataFrame,
    group_col: str,
) -> dict[str, dict[str, float]]:
    """Pooled Spearman IC within each group — retained for regression tests."""
    if group_col not in frame.columns or "realized" not in frame.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for group_value, group in frame.groupby(group_col, sort=True, observed=True):
        part: dict[str, float] = {}
        for model_id, model_rows in group.groupby("model_id"):
            pred = model_rows["prediction"].astype(float)
            realized = model_rows["realized"].astype(float)
            if len(pred) < 2:
                part[str(model_id)] = 0.0
                continue
            ic = float(pred.corr(realized, method="spearman"))
            part[str(model_id)] = ic if np.isfinite(ic) else 0.0
        if part:
            out[str(group_value)] = part
    return out


def _cross_sectional_ic_by_group(
    frame: pd.DataFrame,
    group_col: str,
) -> dict[str, dict[str, float]]:
    """Mean per-date cross-sectional Spearman IC within each group (fold/regime)."""
    if group_col not in frame.columns or "realized" not in frame.columns:
        return {}
    work = frame[[group_col, "model_id", "date", "prediction", "realized"]].copy()
    work["prediction"] = work["prediction"].astype(float)
    work["realized"] = work["realized"].astype(float)
    work = work.loc[
        work["prediction"].notna()
        & work["realized"].notna()
        & np.isfinite(work["prediction"])
        & np.isfinite(work["realized"])
    ]
    if work.empty:
        return {}

    def _daily_ic(day_rows: pd.DataFrame) -> float:
        pred = day_rows["prediction"]
        realized = day_rows["realized"]
        if len(pred) < _MIN_CROSS_SECTIONAL_INSTRUMENTS:
            return np.nan
        if pred.nunique() < 2 or realized.nunique() < 2:
            return np.nan
        ic = float(pred.corr(realized, method="spearman"))
        return ic if np.isfinite(ic) else np.nan

    daily = (
        work.groupby([group_col, "model_id", "date"], sort=True, observed=True)
        .apply(_daily_ic, include_groups=False)
        .reset_index(name="ic")
    )
    daily = daily.loc[daily["ic"].notna()]
    if daily.empty:
        return {}

    means = daily.groupby([group_col, "model_id"], sort=True, observed=True)["ic"].mean()
    out: dict[str, dict[str, float]] = {}
    for (group_value, model_id), ic in means.items():
        out.setdefault(str(group_value), {})[str(model_id)] = float(ic)
    return out


def _flag_redundant_pairs(
    *,
    pred_corr: list[dict[str, object]],
    rank_corr: list[dict[str, object]],
    topk_overlap: list[dict[str, object]],
) -> list[dict[str, object]]:
    redundant: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add_pair(a: str, b: str, reason: str, metric: str, value: float) -> None:
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        redundant.append(
            {
                "model_a": key[0],
                "model_b": key[1],
                "reason": reason,
                "metric": metric,
                "value": value,
            }
        )

    for item in pred_corr:
        value = float(item["value"])
        if abs(value) >= _REDUNDANCY_CORR_THRESHOLD:
            add_pair(
                str(item["model_a"]),
                str(item["model_b"]),
                "effectively_identical_predictions",
                "prediction_correlation",
                value,
            )

    for item in rank_corr:
        value = float(item["value"])
        if abs(value) >= _REDUNDANCY_RANK_THRESHOLD:
            add_pair(
                str(item["model_a"]),
                str(item["model_b"]),
                "effectively_identical_ranks",
                "rank_correlation",
                value,
            )

    for item in topk_overlap:
        value = float(item["mean_topk_jaccard"])
        if value >= _REDUNDANCY_TOPK_OVERLAP_THRESHOLD:
            add_pair(
                str(item["model_a"]),
                str(item["model_b"]),
                "effectively_identical_top_selection",
                "topk_jaccard",
                value,
            )

    return redundant


def materialize_realized_panel_from_scratch(
    *,
    target_column: str,
    n_rows: int,
    unique_dates: tuple[str, ...],
    unique_instruments: tuple[str, ...],
    date_code_path: Path,
    instrument_code_path: Path,
    target_path: Path,
) -> pd.DataFrame:
    """Rebuild date/instrument/target columns from low-memory scratch memmaps."""
    if n_rows <= 0:
        raise ValueError("Cannot materialize realized panel from empty scratch")
    date_codes = np.memmap(date_code_path, dtype=np.int32, mode="r", shape=(n_rows,))
    instrument_codes = np.memmap(instrument_code_path, dtype=np.int32, mode="r", shape=(n_rows,))
    targets = np.memmap(target_path, dtype=np.float32, mode="r", shape=(n_rows,))
    date_labels = np.asarray(unique_dates, dtype=object)[date_codes]
    instrument_labels = np.asarray(unique_instruments, dtype=object)[instrument_codes]
    return pd.DataFrame(
        {
            "date": date_labels,
            "instrument": instrument_labels,
            target_column: targets.astype(np.float64, copy=False),
        }
    )


def _iter_parquet_batches(path: Path, *, columns: list[str], batch_size: int = 250_000):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
        yield batch.to_pandas()


def build_panel_target_lookup(
    panel_path: Path,
    *,
    target_column: str,
    batch_size: int = 250_000,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Build a target lookup from the canonical feature panel."""
    lookup: dict[tuple[str, str, str], float] = {}
    columns = ["date", "instrument", "interval", target_column]
    for batch in _iter_parquet_batches(panel_path, columns=columns, batch_size=batch_size):
        batch = normalize_key_columns(batch)
        values = pd.to_numeric(batch[target_column], errors="coerce")
        for date, instrument, interval, value in zip(
            batch["date"].to_numpy(),
            batch["instrument"].to_numpy(),
            batch["interval"].to_numpy(),
            values.to_numpy(),
            strict=True,
        ):
            if np.isfinite(value):
                lookup[(str(date), str(instrument), str(interval))] = float(value)

    def target_lookup(frame: pd.DataFrame) -> pd.Series:
        normalized = normalize_key_columns(frame)
        dates = normalized["date"].to_numpy()
        instruments = normalized["instrument"].to_numpy()
        intervals = normalized["interval"].to_numpy()
        out = np.fromiter(
            (
                lookup.get((str(dates[i]), str(instruments[i]), str(intervals[i])), np.nan)
                for i in range(len(normalized))
            ),
            dtype=np.float64,
            count=len(normalized),
        )
        return pd.Series(out, index=frame.index, dtype=float)

    return target_lookup


def _attach_targets_from_lookup(
    frame: pd.DataFrame, target_lookup: Callable[[pd.DataFrame], pd.Series]
) -> pd.DataFrame:
    out = frame.copy()
    out["realized"] = target_lookup(out)
    return out


def build_streaming_model_diversity_report(
    predictions_path: Path,
    *,
    target_column: str,
    target_lookup: Callable[[pd.DataFrame], pd.Series],
    top_k: int = 20,
    batch_size: int = 250_000,
    random_seed: int = 42,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    expected_model_count: int | None = None,
    expected_fold_count: int = 3,
) -> dict[str, Any]:
    """Stream prediction parquet with deterministic aligned-key stratified sampling."""
    selection = select_aligned_prediction_keys(
        predictions_path,
        random_seed=random_seed,
        max_sample_rows=max_sample_rows,
        expected_model_count=expected_model_count,
        batch_size=batch_size,
    )
    chunks: list[pd.DataFrame] = []
    total_rows_scanned = 0
    total_rows_kept = 0
    columns = [
        "date",
        "instrument",
        "interval",
        "model_id",
        "fold_id",
        "prediction",
    ]
    for batch in _iter_parquet_batches(predictions_path, columns=columns, batch_size=batch_size):
        batch = batch.copy()
        total_rows_scanned += len(batch)
        if "interval" not in batch.columns:
            batch["interval"] = "1d"
        batch["date"] = batch["date"].astype(str)
        batch["instrument"] = batch["instrument"].astype(str)
        batch["interval"] = batch["interval"].astype(str)
        batch["model_id"] = batch["model_id"].astype(str)
        batch["fold_id"] = batch["fold_id"].astype(str)
        packed = [
            _pack_alignment_key(d, i, iv, f)
            for d, i, iv, f in zip(
                batch["date"].to_numpy(),
                batch["instrument"].to_numpy(),
                batch["interval"].to_numpy(),
                batch["fold_id"].to_numpy(),
                strict=True,
            )
        ]
        keep_mask = pd.Series(packed, index=batch.index).isin(selection.selected_keys)
        kept = batch.loc[keep_mask]
        if kept.empty:
            continue
        kept = normalize_key_columns(kept)
        kept["realized"] = target_lookup(kept)
        chunks.append(kept)
        total_rows_kept += len(kept)
    if not chunks:
        raise ValueError(
            "Streaming diversity report found zero prediction rows for selected aligned keys"
        )
    frame = pd.concat(chunks, ignore_index=True)
    finite_realized = frame["realized"].notna() & np.isfinite(frame["realized"].astype(float))
    panel = frame.loc[:, [*CANONICAL_PANEL_KEYS, "realized"]].rename(
        columns={"realized": target_column}
    )
    report = build_model_diversity_report(frame, panel, target_column, top_k=top_k)
    report["realized_lookup"] = {
        "hit_rate": float(finite_realized.mean()) if len(frame) else 0.0,
        "finite_count": int(finite_realized.sum()),
        "total_rows": int(len(frame)),
    }
    model_count = expected_model_count or int(report.get("model_count", 0) or 0)
    coverage = build_diagnostic_coverage_report(
        frame,
        expected_model_count=model_count,
        expected_fold_count=expected_fold_count,
    )
    report["diagnostic_coverage"] = coverage
    report["sampling"] = {
        "schema_version": "aligned_key_stratified_sampling.v1",
        "method": "deterministic_stratified_by_date_interval_fold",
        "random_seed": random_seed,
        "max_sample_rows": max_sample_rows,
        "total_aligned_keys": selection.total_aligned_keys,
        "target_aligned_keys": selection.target_aligned_keys,
        "selected_aligned_keys": len(selection.selected_keys),
        "stratum_count": selection.stratum_count,
        "rows_scanned": int(total_rows_scanned),
        "rows_sampled": int(total_rows_kept),
        "aligned_keys_shared_across_models": True,
    }
    report["low_memory_source"] = "streaming_prediction_parquet"
    report["streaming_rows_processed"] = int(total_rows_kept)
    return report


def build_low_memory_model_diversity_report(
    predictions_path: Path,
    *,
    target_column: str,
    n_rows: int,
    unique_dates: tuple[str, ...],
    unique_instruments: tuple[str, ...],
    unique_intervals: tuple[str, ...],
    date_code_path: Path,
    instrument_code_path: Path,
    interval_code_path: Path,
    target_path: Path,
    top_k: int = 20,
    random_seed: int = 42,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    expected_model_count: int | None = None,
    expected_fold_count: int = 3,
) -> dict[str, Any]:
    """Full model-matrix diversity report for low-memory train-matrix outputs."""
    date_codes = np.memmap(date_code_path, dtype=np.int32, mode="r", shape=(n_rows,))
    instrument_codes = np.memmap(instrument_code_path, dtype=np.int32, mode="r", shape=(n_rows,))
    interval_codes = np.memmap(interval_code_path, dtype=np.int32, mode="r", shape=(n_rows,))
    targets = np.memmap(target_path, dtype=np.float32, mode="r", shape=(n_rows,))
    lookup: dict[tuple[str, str, str], float] = {}
    chunk = 1_000_000
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        date_labels = np.asarray(unique_dates, dtype=object)[date_codes[start:end]]
        instrument_labels = np.asarray(unique_instruments, dtype=object)[
            instrument_codes[start:end]
        ]
        interval_labels = np.asarray(unique_intervals, dtype=object)[interval_codes[start:end]]
        target_slice = targets[start:end].astype(np.float64, copy=False)
        for date, instrument, interval, value in zip(
            date_labels,
            instrument_labels,
            interval_labels,
            target_slice,
            strict=True,
        ):
            if np.isfinite(value):
                lookup[(str(date), str(instrument), str(interval))] = float(value)

    def target_lookup(batch: pd.DataFrame) -> pd.Series:
        if "interval" not in batch.columns:
            intervals = np.full(len(batch), str(unique_intervals[0]), dtype=object)
        else:
            intervals = batch["interval"].astype(str).to_numpy()
        dates = batch["date"].astype(str).to_numpy()
        instruments = batch["instrument"].astype(str).to_numpy()
        out = np.fromiter(
            (
                lookup.get((str(dates[i]), str(instruments[i]), str(intervals[i])), np.nan)
                for i in range(len(batch))
            ),
            dtype=np.float64,
            count=len(batch),
        )
        return pd.Series(out, index=batch.index, dtype=float)

    return build_streaming_model_diversity_report(
        predictions_path,
        target_column=target_column,
        target_lookup=target_lookup,
        top_k=top_k,
        random_seed=random_seed,
        max_sample_rows=max_sample_rows,
        expected_model_count=expected_model_count,
        expected_fold_count=expected_fold_count,
    )


def build_model_diversity_report(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_column: str,
    *,
    top_k: int = 20,
    redundancy_threshold: float = _REDUNDANCY_CORR_THRESHOLD,
) -> dict[str, Any]:
    """Full model-matrix diversity report for Gate 2."""

    required = {"date", "instrument", "model_id", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError(
            f"predictions missing columns: {sorted(required - set(predictions.columns))}"
        )

    frame = predictions.copy()
    frame = frame.drop(columns=["realized"], errors="ignore")
    frame["date"] = frame["date"].astype(str)
    frame["instrument"] = frame["instrument"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)

    wide = _wide_predictions(frame)
    models = list(wide.columns)
    pred_corr = wide.corr(method="pearson")
    rank_corr = wide.corr(method="spearman")

    dispersion = (
        frame.groupby("model_id")["prediction"]
        .agg(["std", "mean", "count"])
        .reset_index()
        .rename(columns={"std": "prediction_std", "mean": "prediction_mean"})
    )

    residuals = _residual_frame(frame, panel, target_column)
    residuals = residuals.rename(columns={target_column: "realized"})
    residual_wide = residuals.pivot_table(
        index=["date", "instrument"],
        columns="model_id",
        values="residual",
        aggfunc="mean",
    )
    residual_corr = (
        residual_wide.corr(method="pearson") if residual_wide.shape[1] >= 2 else pd.DataFrame()
    )

    topk_overlap = _top_k_overlap(frame, top_k=top_k)

    fold_spearman_ic: dict[str, dict[str, float]] = {}
    if "fold_id" in frame.columns:
        merged = frame.merge(
            residuals[[*CANONICAL_PANEL_KEYS, "model_id", "realized"]],
            on=[*CANONICAL_PANEL_KEYS, "model_id"],
            how="left",
        )
        fold_spearman_ic = _cross_sectional_ic_by_group(merged, "fold_id")

    regimes = _regime_labels(panel, target_column)
    ex_post_regime_spearman_ic: dict[str, dict[str, float]] = {}
    merged_for_regime = frame
    if not regimes.empty:
        merged = frame.merge(
            residuals[[*CANONICAL_PANEL_KEYS, "model_id", "realized"]],
            on=[*CANONICAL_PANEL_KEYS, "model_id"],
            how="left",
        ).merge(regimes, on="date", how="left")
        merged_for_regime = merged
        ex_post_regime_spearman_ic = _cross_sectional_ic_by_group(merged, "regime")

    pred_pairs = _pairwise_matrix(pred_corr)
    rank_pairs = _pairwise_matrix(rank_corr)
    residual_pairs = _pairwise_matrix(residual_corr) if not residual_corr.empty else []

    redundant = _flag_redundant_pairs(
        pred_corr=pred_pairs,
        rank_corr=rank_pairs,
        topk_overlap=topk_overlap,
    )
    near_duplicate_clusters = _positive_connected_clusters(
        models,
        pred_pairs,
        threshold=redundancy_threshold,
    )

    mean_pred_corr = _mean_off_diagonal(pred_corr)
    low_diversity = mean_pred_corr is not None and abs(mean_pred_corr) > 0.85

    cross_sectional = _per_date_metric_summaries(frame, top_k=top_k)
    if "fold_id" in frame.columns:
        cross_sectional["by_fold"] = _per_date_metric_summaries(
            frame,
            group_col="fold_id",
            top_k=top_k,
        )["by_group"]
    if not regimes.empty:
        cross_sectional["by_ex_post_regime"] = _per_date_metric_summaries(
            merged_for_regime,
            group_col="regime",
            top_k=top_k,
        )["by_group"]

    child_counts = _child_count_summary(
        models=models,
        redundant_near_duplicate=redundant,
        dispersion=dispersion,
        fold_spearman_ic=fold_spearman_ic,
        near_duplicate_clusters=near_duplicate_clusters,
    )
    regime_docs = _regime_documentation(panel, target_column, regimes)

    return {
        "schema_version": "model_diversity_report.v2",
        "model_count": len(models),
        "models": models,
        "prediction_correlation": {
            "mean_pairwise": mean_pred_corr,
            "pairs": pred_pairs[:100],
            "note": "Pooled-row correlation across all date/instrument/interval keys.",
        },
        "rank_correlation": {
            "mean_pairwise": _mean_off_diagonal(rank_corr),
            "pairs": rank_pairs[:100],
            "note": "Pooled-row Spearman correlation across all date/instrument/interval keys.",
        },
        "residual_correlation": {
            "mean_pairwise": _mean_off_diagonal(residual_corr) if not residual_corr.empty else None,
            "pairs": residual_pairs[:100],
            "note": (
                "Shared target variance can dominate residual correlation; "
                "use cross_sectional metrics for redundancy decisions."
            ),
        },
        "cross_sectional_correlation": cross_sectional,
        "top_k_overlap": topk_overlap[:100],
        "prediction_dispersion": dispersion.to_dict(orient="records"),
        "fold_by_fold_spearman_ic": fold_spearman_ic,
        "ex_post_regime_spearman_ic": ex_post_regime_spearman_ic,
        "ex_post_regime_documentation": regime_docs,
        "redundant_pairs": redundant,
        "redundant_model_ids": sorted(
            {p["model_a"] for p in redundant} | {p["model_b"] for p in redundant}
        ),
        "soft_redundancy_clusters": _soft_redundancy_report(
            pred_pairs=pred_pairs,
            rank_pairs=rank_pairs,
            models=models,
        ),
        "low_diversity_warning": bool(low_diversity),
        "redundancy_threshold": redundancy_threshold,
        **child_counts,
    }


__all__ = [
    "AlignedKeySelection",
    "build_diagnostic_coverage_report",
    "build_low_memory_model_diversity_report",
    "build_model_diversity_report",
    "build_panel_target_lookup",
    "build_streaming_model_diversity_report",
    "materialize_realized_panel_from_scratch",
    "select_aligned_prediction_keys",
]

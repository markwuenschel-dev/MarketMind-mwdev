from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

TargetType = Literal["regression", "multi_horizon", "classification"]


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        seq_len: int,
        horizon: int,
        target_col: str = "close",
        ticker_col: str | None = "ticker",
        target_type: TargetType = "regression",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        # basic validation -------------------------------------------------
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in DataFrame")
        if seq_len < 1 or horizon < 1:
            raise ValueError("seq_len and horizon must be ≥ 1")

        # keep order consistent, reset index to 0..N‑1 so position math works
        df = df.reset_index(drop=True)

        # store numeric tensor copy on requested device -------------------
        df_numeric = df.select_dtypes(include=[np.number])  # drops ticker str col
        if df_numeric.empty:
            raise ValueError("DataFrame contains no numeric columns")
        self.data = torch.as_tensor(df_numeric.values, dtype=dtype, device=device)
        self.target = torch.as_tensor(df_numeric[target_col].values, dtype=dtype, device=device)

        self.seq_len = seq_len
        self.horizon = horizon
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        # pre‑compute valid window start indices --------------------------
        self.starts: list[int] = []
        if ticker_col and ticker_col in df.columns:
            tickers = df[ticker_col].values  # ndarray[str]
            n = len(df)
            i = 0
            while i < n:
                cur = tickers[i]
                j = i
                # advance to end of this ticker block
                while j < n and tickers[j] == cur:
                    j += 1
                block_len = j - i
                # add starts within [i, j)
                max_start = block_len - seq_len - horizon + 1
                if max_start > 0:
                    self.starts.extend(range(i, i + max_start))
                i = j  # next block
        else:
            # single‑ticker – contiguous data, no grouping needed
            total = len(df) - seq_len - horizon + 1
            if total > 0:
                self.starts = list(range(total))

        if not self.starts:
            raise ValueError("No valid windows for given seq_len/horizon")

    # dataset protocol ----------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.seq_len
        X = self.data[start:end]  # shape (seq_len, n_features)

        # target construction --------------------------------------------
        if self.target_type == "regression":
            y = self.target[end + self.horizon - 1]  # scalar
        elif self.target_type == "multi_horizon":
            y = self.target[end : end + self.horizon]  # vector
        elif self.target_type == "classification":
            future = self.target[end + self.horizon - 1]
            current = self.target[end - 1]
            y = (future > current).long()
        else:  # pragma: no cover – exhaustive guard
            raise RuntimeError(f"Unknown target_type {self.target_type}")

        # optional transforms -------------------------------------------
        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return X, y

    # helpful when printing dataset objects ------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"TimeSeriesDataset(samples={len(self)}, seq_len={self.seq_len}, "
            f"horizon={self.horizon}, target='{self.target_type}')"
        )


# helper to create a DistributedSampler when distributed=True -------------


def _make_sampler(dataset: Dataset, *, distributed: bool, rank: int | None, world_size: int | None):
    if not distributed:
        return None
    if world_size is None or rank is None:
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed requested but torch.distributed not initialised")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)


# -------------------------------------------------------------------------
# Public loaders
# -------------------------------------------------------------------------


def build_loader(
    df: pd.DataFrame,
    *,
    seq_len: int,
    horizon: int,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    target_col: str = "close",
    ticker_col: str | None = "ticker",
    target_type: TargetType = "regression",
    distributed: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    **loader_kwargs,
) -> DataLoader:
    ds = TimeSeriesDataset(
        df,
        seq_len=seq_len,
        horizon=horizon,
        target_col=target_col,
        ticker_col=ticker_col,
        target_type=target_type,
        transform=transform,
        target_transform=target_transform,
        dtype=dtype,
        device=device,
    )

    sampler = _make_sampler(ds, distributed=distributed, rank=rank, world_size=world_size)
    if distributed and shuffle:
        shuffle = False  # DistributedSampler already shuffles.

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        **loader_kwargs,
    )


def build_train_val_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    seq_len: int,
    horizon: int,
    batch_size: int = 128,
    num_workers: int = 4,
    **common_kwargs,
) -> tuple[DataLoader, DataLoader]:
    # shared hyper‑parameters gathered once for both loaders -------------
    common = dict(
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        num_workers=num_workers,
        **common_kwargs,
    )
    return (
        build_loader(train_df, shuffle=True, **common),
        build_loader(val_df, shuffle=False, **common),
    )

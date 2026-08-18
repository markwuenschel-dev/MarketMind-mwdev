"""MLC-1 governed Phase II context encoder (2-layer MLP, D=64 ``regime_embedding``).

Trains only on contract-shaped :class:`EncoderInputContract` feature tensors (PIT metadata
is validated but not concatenated into the vector unless callers place PIT-safe features
inside ``regime_features`` per upstream policy).

**Contract-lowering shim (Phase II interface):** ``regime_features`` may be 1-D
``(input_dim,)`` or 2-D ``(T, F)``; the latter is flattened in C order to a length
``T * F`` vector that must equal ``input_dim``. This is not a second feature spec; it only
lowers the contract's ndarray into the fixed-width MLP input expected by the baseline
architecture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    from torch import Tensor
except ImportError as _torch_err:  # pragma: no cover
    raise ImportError(
        "torch is required for ContextEncoder; install pytorch: pip install torch"
    ) from _torch_err

from datetime import UTC

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.contracts.encoder_contracts import (
    EncoderInputContract,
    EncoderOutputContract,
)
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

# Governed baseline embedding width (AQ-01 / RG-10); not an empirical optimality claim.
REGIME_EMBEDDING_DIM: Final[int] = 64  # ⚑ VALIDATE
HIDDEN_DIM: Final[int] = 128
NUM_REGIME_CLASSES: Final[int] = 5

# Default flattened regime feature width for the 2-layer MLP input (contract must match).
CONTEXT_ENCODER_DEFAULT_INPUT_DIM: Final[int] = 16  # ⚑ VALIDATE


@dataclass(frozen=True, slots=True)
class ContextEncoderPretrainSummary:
    """Minimal pre-training smoke summary (loss trajectory only)."""

    initial_loss: float
    final_loss: float
    epochs: int
    n_examples: int


def _normalize_pit_boundary(value: object) -> None:
    from datetime import datetime

    if not isinstance(value, datetime):
        raise DataPreconditionError(
            "EncoderInputContract.pit_boundary must be datetime",
            details={"type": type(value).__name__},
        )
    if value.tzinfo is None:
        raise DataPreconditionError(
            "pit_boundary must be timezone-aware (governed PIT discipline)",
            details={"pit_boundary": value.isoformat()},
        )
    _ = value.astimezone(UTC)


def _encoder_feature_vector(
    inp: EncoderInputContract,
    *,
    input_dim: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    arr = np.asarray(inp.regime_features, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr.reshape(-1)
    elif arr.ndim == 2:
        vec = arr.reshape(-1, order="C")
    else:
        raise DataPreconditionError(
            "regime_features must be 1-D or 2-D for the MLC-1 encoder",
            details={"ndim": int(arr.ndim)},
        )
    if vec.size != input_dim:
        raise DataPreconditionError(
            "regime_features size must match encoder input_dim (contract-lowering shim)",
            details={"expected": int(input_dim), "actual": int(vec.size)},
        )
    return np.ascontiguousarray(vec, dtype=np.float32)


def _xavier_init_linear(module: nn.Module, gen: torch.Generator) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, generator=gen)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ContextEncoder:
    """Two-layer MLP encoder: Linear(D→128), ReLU, Linear(128→64); inference emits float32 z.

    Implements :class:`ContextEncoderProtocol` shape/dtype semantics via
    :meth:`encode` → :class:`EncoderOutputContract`. ``freeze`` / ``unfreeze`` toggle
    ``requires_grad`` on **all** trunk parameters per MLC-1; :meth:`is_frozen` mirrors the
    protocol's inner-loop expectation.
    """

    def __init__(
        self,
        input_dim: int = CONTEXT_ENCODER_DEFAULT_INPUT_DIM,
        *,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if input_dim < 1:
            raise DataPreconditionError(
                "input_dim must be positive", details={"input_dim": input_dim}
            )
        self._input_dim = int(input_dim)
        self._device = torch.device(device or "cpu")
        self._frozen = False
        gen = torch.Generator(device=self._device)
        gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        trunk = nn.Sequential(
            nn.Linear(self._input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, REGIME_EMBEDDING_DIM),
        ).to(self._device)
        _xavier_init_linear(trunk, gen)
        self._trunk = trunk
        self._head: nn.Linear | None = None

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._set_trainable(False)

    def unfreeze(self) -> None:
        self._set_trainable(True)

    def _set_trainable(self, trainable: bool) -> None:
        self._frozen = not trainable
        for p in self._trunk.parameters():
            p.requires_grad_(trainable)
        if self._head is not None:
            for p in self._head.parameters():
                p.requires_grad_(trainable)

    def encode(self, input: EncoderInputContract) -> EncoderOutputContract:
        if self._head is not None:
            raise DataPreconditionError(
                "encoder has a pre-training head attached; finish pretrain_classifier first",
                details={},
            )
        if not isinstance(input.signal_set_version, int) or isinstance(
            input.signal_set_version, bool
        ):
            raise DataPreconditionError(
                "signal_set_version must be int (contract)",
                details={"signal_set_version": input.signal_set_version},
            )
        _normalize_pit_boundary(input.pit_boundary)
        vec = _encoder_feature_vector(input, input_dim=self._input_dim)
        x = torch.tensor(vec, dtype=torch.float32, device=self._device, requires_grad=False)
        self._trunk.eval()
        with torch.no_grad():
            z = self._trunk(x)
        out = np.asarray(z.detach().cpu().numpy(), dtype=np.float32).reshape(REGIME_EMBEDDING_DIM)
        out = np.ascontiguousarray(out, dtype=np.float32)
        return EncoderOutputContract(regime_embedding=out, schema_version="v1")

    def pretrain_classifier(
        self,
        examples: Sequence[tuple[EncoderInputContract, str]],
        *,
        epochs: int = 40,
        lr: float = 0.05,
        batch_size: int = 8,
        seed: int = 0,
    ) -> ContextEncoderPretrainSummary:
        """Fit a 5-class head with cross-entropy; head is removed before return."""
        if not examples:
            raise DataPreconditionError(
                "pretrain_classifier requires at least one example", details={}
            )
        if self._head is not None:
            raise DataPreconditionError(
                "pretrain_classifier may not run while a head is attached", details={}
            )
        gen = torch.Generator(device=self._device)
        gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        label_index = {c: i for i, c in enumerate(REGIME_CLASS_ORDER)}
        self._head = nn.Linear(REGIME_EMBEDDING_DIM, NUM_REGIME_CLASSES).to(self._device)
        _xavier_init_linear(self._head, gen)
        self.unfreeze()
        params: list[nn.Parameter] = []
        params.extend(self._trunk.parameters())
        params.extend(self._head.parameters())
        opt = torch.optim.Adam(params, lr=float(lr))
        loss_fn = nn.CrossEntropyLoss()
        tensors_x: list[Tensor] = []
        tensors_y: list[Tensor] = []
        for ex, lab in examples:
            if lab not in label_index:
                raise DataPreconditionError("unknown regime_class label", details={"label": lab})
            _normalize_pit_boundary(ex.pit_boundary)
            if not isinstance(ex.signal_set_version, int) or isinstance(
                ex.signal_set_version, bool
            ):
                raise DataPreconditionError(
                    "signal_set_version must be int (contract)",
                    details={"signal_set_version": ex.signal_set_version},
                )
            vec = _encoder_feature_vector(ex, input_dim=self._input_dim)
            tensors_x.append(
                torch.tensor(vec, dtype=torch.float32, device=self._device, requires_grad=False)
            )
            tensors_y.append(torch.tensor(label_index[lab], dtype=torch.long, device=self._device))
        n = len(tensors_x)
        initial_loss = float("nan")
        final_loss = float("nan")
        for ep in range(int(epochs)):
            perm = torch.randperm(n, generator=gen, device=self._device)
            epoch_losses: list[float] = []
            for start in range(0, n, int(batch_size)):
                idx = perm[start : start + int(batch_size)]
                xb = torch.stack([tensors_x[int(i)] for i in idx.tolist()])
                yb = torch.stack([tensors_y[int(i)] for i in idx.tolist()])
                self._trunk.train()
                self._head.train()
                opt.zero_grad(set_to_none=True)
                logits = self._head(self._trunk(xb))
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            mean_ep = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if ep == 0:
                initial_loss = mean_ep
            if ep == int(epochs) - 1:
                final_loss = mean_ep
        self._trunk.eval()
        self._head = None
        return ContextEncoderPretrainSummary(
            initial_loss=initial_loss,
            final_loss=final_loss,
            epochs=int(epochs),
            n_examples=n,
        )


__all__ = [
    "CONTEXT_ENCODER_DEFAULT_INPUT_DIM",
    "NUM_REGIME_CLASSES",
    "REGIME_EMBEDDING_DIM",
    "ContextEncoder",
    "ContextEncoderPretrainSummary",
]

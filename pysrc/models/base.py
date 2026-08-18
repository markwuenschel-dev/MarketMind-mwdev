"""Panel model protocol for ticker-date experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PanelModel(Protocol):
    model_id: str

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, *, fold_id: str) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def predict_confidence(self, x: np.ndarray) -> np.ndarray: ...

    def feature_usage(self) -> list[str]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> PanelModel: ...


__all__ = ["PanelModel"]

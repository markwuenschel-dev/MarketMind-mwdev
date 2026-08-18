"""LSTM architecture and classifier modules for sequence models."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Sampler, default_collate

from pysrc.core.errors import (
    DataValidationError,
    InvalidInputError,
    ModelCheckpointError,
    ModelInferenceError,
    ModelTrainingError,
)
from pysrc.core.validation import validate_tensor
from pysrc.models.runtime.torch import init_weights, seed_everything
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.pipeline_config import get_config

# Initialise logger (per-model seeds set in LstmPanelModel / LSTMConfig)
log = get_logger(__name__)


@dataclass(kw_only=True)
class LSTMConfig:
    # Core sizes
    input_dim: int  # feature dimension per time step
    units: int = 128  # hidden size of (bi-)LSTM
    num_layers: int = 2  # stacked LSTM blocks

    # Regularisation
    zoneout_rate: float = 0.1  # zone-out prob for custom cell
    input_dropout_rate: float = 0.1  # dropout mask shared across time steps
    dropout: float = 0.0  # native LSTM dropout between stacked layers

    # Architecture toggles
    bidirectional: bool = True
    return_sequences: bool = False  # expose full sequence instead of last step
    residual: bool = True  # enable layer-wise residuals
    pooling_type: str | None = "attention"  # None | "attention" | "gated" | "mean" | "max"
    use_custom_cell: bool = False  # NormLSTM vs standard nn.LSTM

    # Reproducibility
    seed: int | None = None  # pass to torch_utils.seed_everything

    # ---------------------------------------------------------
    def to_dict(self):
        return asdict(self)


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: list[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        boundaries: list[int],
    ) -> None:
        super().__init__(dataset)
        self.batch_size = batch_size
        self.buckets: list[list[int]] = [[] for _ in range(len(boundaries) + 1)]

        # Partition indices by length bucket
        for idx, (seq, _) in enumerate(dataset):
            length = seq.size(0)
            for bucket_id, bound in enumerate(boundaries):
                if length <= bound:
                    self.buckets[bucket_id].append(idx)
                    break
            else:
                self.buckets[-1].append(idx)

        # Shuffle each bucket once per sampler instantiation
        for b in self.buckets:
            random.shuffle(b)
        log.debug("BucketBatchSampler created", n_buckets=len(self.buckets))

    def __iter__(self):  # type: ignore[override]
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)


# ---------------------------------------------------------------------------


def collate_fn(batch):

    xs, ys = zip(*batch, strict=False)
    lengths = torch.as_tensor([x.size(0) for x in xs])
    x_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    y = torch.stack(ys)
    return default_collate((x_pad, y, lengths))


class SharedDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self._mask: torch.Tensor | None = None

    def train(self, mode: bool = True):  # noqa: D401 override strictly
        self._mask = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if not (self.training and self.p > 0):
            return x
        # Re-create mask if batch size has changed
        if self._mask is None or self._mask.size(0) != x.size(0):
            m = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli_(1 - self.p)
            self._mask = m / (1 - self.p)
        return x * self._mask


class NormLSTMCell(nn.Module):
    def __init__(self, inp: int, hid: int, zoneout: float):
        super().__init__()
        self.hid, self.zoneout = hid, zoneout

        self.W = nn.Parameter(torch.empty(inp, 4 * hid))
        self.U = nn.Parameter(torch.empty(hid, 4 * hid))
        self.b = nn.Parameter(torch.empty(4 * hid))

        self.ln_i = nn.LayerNorm(hid)
        self.ln_f = nn.LayerNorm(hid)
        self.ln_g = nn.LayerNorm(hid)
        self.ln_o = nn.LayerNorm(hid)

        # Init params
        nn.init.xavier_uniform_(self.W)
        nn.init.orthogonal_(self.U)
        nn.init.zeros_(self.b)
        self.b.data[hid : 2 * hid] = 1.0  # forget-gate bias

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        z = x @ self.W + h_prev @ self.U + self.b
        i, f, g, o = z.chunk(4, dim=1)
        i = torch.sigmoid(self.ln_i(i))
        f = torch.sigmoid(self.ln_f(f))
        g = torch.tanh(self.ln_g(g))
        o = torch.sigmoid(self.ln_o(o))

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        if self.training and self.zoneout > 0:
            mask_h = torch.rand_like(h) < self.zoneout
            mask_c = torch.rand_like(c) < self.zoneout
            h = torch.where(mask_h, h_prev, h)
            c = torch.where(mask_c, c_prev, c)
        return h, (h, c)


class NormLSTM(nn.Module):
    def __init__(self, inp: int, hid: int, zoneout: float, *, return_seq: bool):
        super().__init__()
        self.cell = NormLSTMCell(inp, hid, zoneout)
        # Optional compile for CUDA speedup in PyTorch 2.x+
        if hasattr(torch, "compile") and torch.cuda.is_available():
            self.cell = torch.compile(self.cell)
        self.return_seq = return_seq

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.cell.hid)
        c = x.new_zeros_like(h)

        if lengths is not None and lengths.device != x.device:
            lengths = lengths.to(x.device)
        mask = (
            torch.arange(T, device=x.device)[None] < lengths[:, None]
            if lengths is not None
            else torch.ones(B, T, dtype=torch.bool, device=x.device)
        )

        # Collect all output hidden states
        hs = []
        for t in range(T):
            x_t = x[:, t]
            valid = mask[:, t]
            if not valid.any():
                hs.append(h)
                continue
            h_prop, (n_h, n_c) = self.cell(x_t, (h, c))
            h = torch.where(valid.unsqueeze(1), h_prop, h)
            c = torch.where(valid.unsqueeze(1), n_c, c)
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, T, H)
        return hs if self.return_seq else h  # <- return only h


class BidirectionalNormLSTM(nn.Module):
    def __init__(self, inp: int, hid: int, zoneout: float, *, return_seq: bool):
        super().__init__()
        self.fwd = NormLSTM(inp, hid, zoneout, return_seq=return_seq)
        self.bwd = NormLSTM(inp, hid, zoneout, return_seq=return_seq)
        self.return_seq = return_seq

    @staticmethod
    def _rev(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        idx = torch.arange(T, device=x.device).expand(B, T)
        rev_idx = (lengths[:, None] - 1 - idx).clamp(min=0)
        rev_idx = torch.where(idx < lengths[:, None], rev_idx, idx)
        return x.gather(1, rev_idx.unsqueeze(-1).expand(-1, -1, D))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        f = self.fwd(x, lengths)
        x_rev = self._rev(x, lengths) if lengths is not None else torch.flip(x, [1])
        b = self.bwd(x_rev, lengths)
        if self.return_seq:
            b = torch.flip(b, [1]) if lengths is None else self._rev(b, lengths)
        return torch.cat([f, b], dim=-1)


class LSTMBlock(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.input_dim

        if cfg.return_sequences and cfg.pooling_type is not None:
            raise ValueError("Pooling is not applicable when return_sequences=True")

        self.layers = nn.ModuleList()
        self.dropouts = (
            nn.ModuleList([SharedDropout(cfg.input_dropout_rate) for _ in range(cfg.num_layers)])
            if cfg.input_dropout_rate > 0
            else None
        )

        keep_seq_flags = [True] * (cfg.num_layers - 1) + [
            cfg.return_sequences or cfg.pooling_type is not None
        ]
        self.keep_seq_flags = keep_seq_flags

        for i in range(cfg.num_layers):
            if cfg.use_custom_cell:
                lstm_cls = BidirectionalNormLSTM if cfg.bidirectional else NormLSTM
                lstm = lstm_cls(dim, cfg.units, cfg.zoneout_rate, return_seq=keep_seq_flags[i])
            else:
                lstm = nn.LSTM(
                    dim,
                    cfg.units,
                    num_layers=1,
                    bidirectional=cfg.bidirectional,
                    batch_first=True,
                    dropout=cfg.dropout,
                )
            self.layers.append(lstm)
            dim = cfg.units * 2 if cfg.bidirectional else cfg.units

        if not cfg.return_sequences:
            if cfg.pooling_type == "attention":
                self.attention = nn.Linear(dim, 1)
            elif cfg.pooling_type == "gated":
                self.gate = nn.Linear(2 * dim, dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if lengths is not None:
            lengths_cpu = lengths.cpu()

        for i, lstm in enumerate(self.layers):
            if self.dropouts is not None:
                x = self.dropouts[i](x)

            if self.cfg.use_custom_cell:
                out = lstm(x, lengths)
            else:
                if lengths is not None:
                    packed = pack_padded_sequence(
                        x, lengths_cpu, batch_first=True, enforce_sorted=False
                    )
                    packed_out, (h_n, _) = lstm(packed)
                    if self.keep_seq_flags[i]:
                        out, _ = pad_packed_sequence(packed_out, batch_first=True)
                    else:
                        if self.cfg.bidirectional:
                            out = torch.cat([h_n[0], h_n[1]], dim=-1)
                        else:
                            out = h_n[0]
                else:
                    out, (h_n, _) = lstm(x)
                    if not self.keep_seq_flags[i]:
                        if self.cfg.bidirectional:
                            out = torch.cat([h_n[0], h_n[1]], dim=-1)
                        else:
                            out = h_n[0]

            # Residuals if shape matches
            if self.cfg.residual and i > 0 and out.dim() == x.dim() and out.shape == x.shape:
                out = out + x
            x = out

        # Pooling / sequence return
        if self.cfg.return_sequences or self.cfg.pooling_type is None:
            return x
        if self.cfg.pooling_type == "attention":
            w = F.softmax(self.attention(x).squeeze(-1), dim=1).unsqueeze(-1)
            return (w * x).sum(1)
        if self.cfg.pooling_type == "gated":
            avg = x.mean(1)
            mx, _ = x.max(1)
            g = torch.sigmoid(self.gate(torch.cat([avg, mx], 1)))
            return g * avg + (1 - g) * mx
        if self.cfg.pooling_type == "mean":
            return x.mean(1)
        if self.cfg.pooling_type == "max":
            return x.max(1)[0]
        raise ValueError(f"Unsupported pooling_type: {self.cfg.pooling_type}")

    def get_config(self):
        return asdict(self.cfg)

    @classmethod
    def from_config(cls, config):
        return cls(LSTMConfig(**config))


class Model(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg

        # honour per-instance seed if supplied
        if cfg.seed is not None:
            seed_everything(cfg.seed)

        self.backbone = LSTMBlock(cfg)
        out_dim = cfg.units * 2 if cfg.bidirectional else cfg.units
        self.head = nn.Linear(out_dim, 1)

        log.debug(
            "Model initialised",
            **cfg.to_dict(),
            n_params=sum(p.numel() for p in self.parameters()),
        )

    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # 1) Fail-fast validation -------------------------------------------------
        try:
            validate_tensor(x, name="inputs", min_dims=3)
            if lengths is not None:
                validate_tensor(lengths, name="lengths", min_dims=1)
        except DataValidationError as exc:
            log.error(
                "Tensor validation failed",
                error=str(exc),
                details=getattr(exc, "details", {}),
            )
            raise ModelTrainingError(str(exc)) from exc

        # 2) Forward pass ---------------------------------------------------------
        try:
            feats = self.backbone(x, lengths)  # (B, D) or (B, T, D)

            # Apply linear head; handles both (B, D) and (B, T, D) transparently
            logits = self.head(feats).squeeze(-1)  # (B) or (B, T)

            return logits
        except Exception as exc:
            log.exception("Forward pass crashed", error=str(exc))
            raise ModelInferenceError(
                "Forward pass crashed",
                details={"error": str(exc)},
            ) from exc


class ClassifierConfig(LSTMConfig):
    num_classes: int = 1  # 1 ⇒ binary, >1 ⇒ multi-class
    projection_dim: int | None = None  # bottleneck after LSTM (# params)
    pooling_type: str | None = "mean"  # default safer for unpadded seqs

    @classmethod
    def from_marketmind(cls, overrides: Mapping[str, Any] | None = None) -> ClassifierConfig:
        mm_cfg = get_config()  # may raise ConfigValidationError upstream
        log.debug("Loaded MarketMind pipeline_config", version=mm_cfg.version)

        base = {
            "input_dim": mm_cfg.model.architecture.input_size
            or 10,  # Default if not in pipeline_config
            "units": mm_cfg.model.architecture.hidden_size,
            "num_layers": mm_cfg.model.architecture.num_layers,
            "dropout": mm_cfg.model.architecture.dropout,
            "bidirectional": True,  # MarketMind favours bi-dir encoders
            "sequence_length": mm_cfg.model.sequence_length,
            "num_classes": getattr(mm_cfg.model, "num_classes", 1),
        }
        if overrides:
            base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig | None = None):
        super().__init__()

        # Fallback to YAML-driven defaults when caller passes nothing
        if config is None:
            try:
                config = ClassifierConfig.from_marketmind()
                log.info("ClassifierConfig auto-built from MarketMind YAML")
            except Exception as exc:  # pragma: no-cover – rare in prod
                log.warning("Falling back to hard-coded defaults", error=str(exc))
                config = ClassifierConfig(input_dim=10, units=128)
        self.config = config

        # Honour per-instance seed so two classifiers can be independent
        if config.seed is not None:
            seed_everything(config.seed, deterministic=True)

        # Backbone encoder
        self.lstm_block = LSTMBlock(config)
        d_model = config.units * (2 if config.bidirectional else 1)

        # Optional projection (parameter saver)
        if config.projection_dim:
            self.projection = nn.Linear(d_model, config.projection_dim)
            d_model = config.projection_dim
        else:
            self.projection = None

        # Final classifier layer
        self.classifier = nn.Linear(d_model, config.num_classes)

        # Standardise weight init *before* any manual tweaks
        self.apply(init_weights)

        # Forget-gate bias trick (helps long-range gradients)
        for name, p in self.lstm_block.named_parameters():
            if "bias_ih" in name or "bias_hh" in name:
                n = p.numel()
                p.data[n // 4 : n // 2].fill_(1.0)

        log.debug(
            "LSTMClassifier initialised",
            params=self.num_parameters(),
            n_classes=config.num_classes,
            seed=config.seed,
        )

    # -----------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        try:
            validate_tensor(x, name="inputs", min_dims=3)
            feats = self.lstm_block(x, lengths=lengths)
            if self.projection is not None:
                feats = F.relu(self.projection(feats))
            return self.classifier(feats)
        except DataValidationError as exc:
            raise InvalidInputError(str(exc), details=getattr(exc, "details", {})) from exc
        except Exception as exc:
            raise ModelInferenceError("Forward pass crashed", details={"error": str(exc)}) from exc

    # -----------------------------------------------------------------
    @torch.inference_mode()
    def predict_proba(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(x, lengths)
        if self.config.num_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    # -----------------------------------------------------------------
    def save(self, path: str) -> None:  # UPDATED
        try:
            torch.save(
                {"pipeline_config": asdict(self.config), "state_dict": self.state_dict()},
                path,
            )
            log.info("Checkpoint saved", path=path)
        except Exception as exc:
            raise ModelCheckpointError(
                "Failed to save checkpoint",
                details={"path": path, "error": str(exc)},
            ) from exc

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> LSTMClassifier:  # UPDATED
        try:
            ckpt = torch.load(path, map_location=map_location)
            config = ClassifierConfig(**ckpt["pipeline_config"])
            model = cls(config)
            model.load_state_dict(ckpt["state_dict"])
            log.info("Checkpoint loaded", path=path)
            return model
        except Exception as exc:
            raise ModelCheckpointError(
                "Failed to load checkpoint",
                details={"path": path, "error": str(exc)},
            ) from exc


class LstmPanelModel:
    """Sequence PanelModel wrapper over the LSTM regression backbone."""

    def __init__(
        self,
        *,
        model_id: str = "lstm",
        sequence_length: int = 20,
        random_seed: int = 42,
        units: int = 32,
        num_layers: int = 1,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> None:
        self.model_id = model_id
        self.sequence_length = int(sequence_length)
        self.random_seed = int(random_seed)
        self.units = int(units)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self._feature_names: list[str] = []
        self._residual_std: float = 1.0
        self._model: Model | None = None
        self._scaler = StandardScaler()

    def set_feature_names(self, names: list[str]) -> None:
        self._feature_names = list(names)

    def feature_usage(self) -> list[str]:
        return list(self._feature_names)

    def _scale(self, x: np.ndarray, *, fit: bool) -> np.ndarray:
        flat = x.reshape(-1, x.shape[-1])
        scaled = self._scaler.fit_transform(flat) if fit else self._scaler.transform(flat)
        return scaled.reshape(x.shape).astype(np.float32)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, *, fold_id: str) -> None:
        del fold_id
        if x_train.ndim != 3:
            raise ValueError(f"LstmPanelModel expects (n, seq_len, features); got {x_train.shape}")
        seed_everything(self.random_seed, deterministic=True)
        x_scaled = self._scale(x_train, fit=True)
        cfg = LSTMConfig(
            input_dim=x_scaled.shape[-1],
            units=self.units,
            num_layers=self.num_layers,
            bidirectional=False,
            seed=self.random_seed,
        )
        self._model = Model(cfg)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_train.reshape(-1), dtype=torch.float32)
        self._model.train()
        for _ in range(self.epochs):
            for start in range(0, len(x_tensor), self.batch_size):
                batch_x = x_tensor[start : start + self.batch_size]
                batch_y = y_tensor[start : start + self.batch_size]
                optimizer.zero_grad()
                preds = self._model(batch_x).reshape(-1)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
        self._model.eval()
        with torch.no_grad():
            preds = self._model(x_tensor).reshape(-1).cpu().numpy()
        resid = y_train.reshape(-1) - preds
        self._residual_std = float(max(np.std(resid), 1e-6))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelInferenceError("LstmPanelModel is not fitted")
        x_scaled = self._scale(x, fit=False)
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            return self._model(x_tensor).reshape(-1).cpu().numpy()

    def predict_confidence(self, x: np.ndarray) -> np.ndarray:
        from pysrc.models.tabular import _confidence_from_predictions

        preds = self.predict(x).reshape(-1)
        return _confidence_from_predictions(preds, self._residual_std)

    def predict_with_confidence(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self.predict(x).reshape(-1)
        return preds, self.predict_confidence(x)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ModelCheckpointError("Cannot save unfitted LstmPanelModel")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "feature_names": self._feature_names,
                "residual_std": self._residual_std,
                "scaler_mean": self._scaler.mean_,
                "scaler_scale": self._scaler.scale_,
                "config": self._model.get_config(),
                "state_dict": self._model.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> LstmPanelModel:
        payload = torch.load(path, map_location="cpu")
        cfg = LSTMConfig(**payload["config"])
        self.model_id = str(payload.get("model_id", self.model_id))
        self._feature_names = list(payload.get("feature_names", []))
        self._residual_std = float(payload.get("residual_std", 1.0))
        mean = payload.get("scaler_mean", payload.get("feature_means"))
        scale = payload.get("scaler_scale", payload.get("feature_stds"))
        if mean is not None and scale is not None:
            self._scaler.mean_ = np.asarray(mean, dtype=np.float64)
            self._scaler.scale_ = np.asarray(scale, dtype=np.float64)
            self._scaler.n_features_in_ = int(len(self._scaler.mean_))
        self._model = Model(cfg)
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        return self


def create_lstm_panel_model(
    *,
    model_id: str = "lstm",
    sequence_length: int = 20,
    random_seed: int = 42,
    params: dict[str, object] | None = None,
) -> LstmPanelModel:
    hp = params or {}
    return LstmPanelModel(
        model_id=model_id,
        sequence_length=sequence_length,
        random_seed=random_seed,
        units=int(hp.get("units", 8)),
        num_layers=int(hp.get("num_layers", 1)),
        epochs=int(hp.get("epochs", 2)),
        batch_size=int(hp.get("batch_size", 8)),
        learning_rate=float(hp.get("learning_rate", 1e-3)),
    )

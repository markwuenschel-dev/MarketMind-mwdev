from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Protocol

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import nn

from pysrc.ops.mm_logkit import get_logger

__all__ = [
    "seed_everything",
    "init_weights",
    "autocast",
    "get_device",
    "set_perf_flags",
    "set_logging_level",
]


class _LoggerLike(Protocol):
    def setLevel(self, level: int | str) -> None: ...
    def getEffectiveLevel(self) -> int: ...
    def isEnabledFor(self, level: int | str) -> bool: ...
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> Any: ...
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> Any: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> Any: ...


_logger: _LoggerLike | None = None
_logger_configured: bool = False


def _get_logger() -> _LoggerLike:
    class _LoggerAdapter:
        def __init__(self, base: Any) -> None:
            self._base = base
            # Do NOT call methods on structlog logger during init; keep our own level.
            self._level = logging.INFO

        # Pure internal level management to avoid structlog proxies
        def setLevel(self, level: int | str) -> None:
            try:
                self._level = int(level)
            except Exception:
                # Support strings like "DEBUG"
                self._level = getattr(logging, str(level).upper(), logging.INFO)

        def getEffectiveLevel(self) -> int:
            return self._level

        def isEnabledFor(self, level: int | str) -> bool:
            try:
                lvl = int(level)
            except Exception:
                lvl = getattr(logging, str(level).upper(), logging.INFO)
            return lvl >= self._level

        # Logging: delegate only to benign methods on base; these are safe on structlog
        def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> Any:
            # structlog usually lacks .log; route by level
            if level >= logging.INFO and hasattr(self._base, "info"):
                return self._base.info(msg, *args, **kwargs)
            if hasattr(self._base, "debug"):
                return self._base.debug(msg, *args, **kwargs)
            if hasattr(self._base, "d"):
                return self._base.d(msg, *args, **kwargs)
            return None

        def debug(self, msg: str, *args: Any, **kwargs: Any) -> Any:
            if hasattr(self._base, "debug"):
                return self._base.debug(msg, *args, **kwargs)
            if hasattr(self._base, "d"):
                return self._base.d(msg, *args, **kwargs)
            if hasattr(self._base, "info"):
                return self._base.info(msg, *args, **kwargs)
            return None

        def info(self, msg: str, *args: Any, **kwargs: Any) -> Any:
            if hasattr(self._base, "info"):
                return self._base.info(msg, *args, **kwargs)
            return self.debug(msg, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

    global _logger, _logger_configured
    if _logger is None:
        base = get_logger(__name__)
        _logger = _LoggerAdapter(base)
    if not _logger_configured:
        level_name = os.environ.get("TORCH_UTILS_LOG_LEVEL", "INFO")
        _logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
        _logger_configured = True
    return _logger


# Function to set the logging level dynamically for controlling verbosity
def set_logging_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]) -> None:
    log_level = getattr(logging, level.upper(), None)
    if log_level is None:
        raise ValueError(
            f"Invalid logging level: {level}. Valid: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    lg = _get_logger()
    lg.setLevel(log_level)
    lg.log(log_level, f"Logging level set to {level.upper()}.")


# Self-contained infra for setting specific performance flags.


def _set_tf32_flag(value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"allow_tf32 must be bool; got {type(value)}.")
    torch.backends.cuda.matmul.allow_tf32 = value
    torch.backends.cudnn.allow_tf32 = value
    _get_logger().debug(f"Set allow_tf32 to {value}.")


def _set_matmul_precision(value: str) -> None:
    valid_matmul = {"lowest", "medium", "high", "highest"}
    if not isinstance(value, str) or value not in valid_matmul:
        raise ValueError(f"Invalid matmul_precision: {value}. Valid: {valid_matmul}.")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(value)
        _get_logger().debug(f"Set matmul_precision to {value}.")
    else:
        _get_logger().debug("torch.set_float32_matmul_precision not available; skipping.")


def _set_reduced_precision_flag(flag_name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{flag_name} must be bool; got {type(value)}.")
    if flag_name == "allow_fp16_reduced_precision_reduction":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = value
        _get_logger().debug(f"Set allow_fp16_reduced_precision_reduction to {value}.")
    elif flag_name == "allow_bf16_reduced_precision_reduction":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = value
        _get_logger().debug(f"Set allow_bf16_reduced_precision_reduction to {value}.")
    elif flag_name == "allow_fp16_accumulation":
        torch.backends.cuda.matmul.allow_fp16_accumulation = value
        _get_logger().debug(f"Set allow_fp16_accumulation to {value}.")
    else:
        raise ValueError(f"Unknown reduced precision flag: {flag_name}.")


# Main function calls the infra for modularity and clarity
def set_perf_flags(
    precision_flags: dict[str, bool | str] | None = None,
) -> None:
    if not torch.cuda.is_available():
        lg = _get_logger()
        if lg.isEnabledFor(logging.DEBUG):
            lg.debug("CUDA not available; performance flags unchanged.")
        return

    if precision_flags is None:
        precision_flags = {}

    for key, value in precision_flags.items():
        if key == "allow_tf32":
            if not isinstance(value, bool):
                raise ValueError(f"allow_tf32 must be bool; got {type(value)}.")
            _set_tf32_flag(value)
        elif key == "matmul_precision":
            if not isinstance(value, str):
                raise ValueError(f"matmul_precision must be str; got {type(value)}.")
            _set_matmul_precision(value)
        elif key in [
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
            "allow_fp16_accumulation",
        ]:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be bool; got {type(value)}.")
            _set_reduced_precision_flag(key, value)
        else:
            raise ValueError(f"Unknown precision flag: {key}.")


# Seed every RNG we care about. When deterministic=True we also ask PyTorch for
# algorithm-level determinism, disable benchmarking, and fill uninitialized memory.
# Optional return of a seeded generator for advanced reproducibility.
def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
    precision_flags: dict[str, bool | str] | None = None,
    return_generator: bool = False,
) -> torch.Generator | None:
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer; got {type(seed)}.")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        # Fill uninitialized memory for added determinism (PyTorch 2.0+)
        if hasattr(torch.utils, "deterministic") and hasattr(
            torch.utils.deterministic, "fill_uninitialized_memory"
        ):
            torch.utils.deterministic.fill_uninitialized_memory = True
        # CUBLAS pipeline_config for reproducibility
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _get_logger().debug("Enabled full deterministic mode.")

    set_perf_flags(precision_flags)

    if return_generator:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen
    return None


# Self-contained infra for initializing specific layer types.


def _init_conv_linear_weight(
    m: nn.Module,
    mode: str,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize Conv/Linear weights with specified mode."""
    if mode == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight, generator=generator)
    elif mode == "xavier_normal":
        nn.init.xavier_normal_(m.weight, generator=generator)
    elif mode == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu", generator=generator)
    elif mode == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu", generator=generator)
    elif mode == "orthogonal":
        nn.init.orthogonal_(m.weight, generator=generator)
    elif mode == "trunc_normal":
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, generator=generator)
    else:
        raise ValueError(f"Unknown init mode for Conv/Linear: {mode}")
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_norm_weight(
    m: nn.Module,
) -> None:
    nn.init.ones_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_embedding_weight(
    m: nn.Module,
    mode: str,
    generator: torch.Generator | None = None,
) -> None:
    if mode in ("trunc_normal", "normal"):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, generator=generator)
    else:
        nn.init.normal_(m.weight, mean=0.0, std=0.02, generator=generator)


def _init_rnn_weight(
    m: nn.Module,
    mode: str,
    generator: torch.Generator | None = None,
) -> None:
    for name, param in m.named_parameters():
        if "weight" in name:
            if mode in ("xavier_uniform", "orthogonal"):
                nn.init.xavier_uniform_(param, generator=generator)
            else:
                nn.init.xavier_uniform_(param, generator=generator)
        elif "bias" in name:
            nn.init.zeros_(param)


# Recursively apply a chosen weight initialisation strategy using modular infra.
def init_weights(
    module: nn.Module,
    mode: Literal[
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
        "orthogonal",
        "trunc_normal",
    ] = "xavier_uniform",
    generator: torch.Generator | None = None,
) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            _init_conv_linear_weight(m, mode, generator)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            _init_norm_weight(m)
        elif isinstance(m, nn.Embedding):
            _init_embedding_weight(m, mode, generator)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            _init_rnn_weight(m, mode, generator)


# Auto-select a compute device. Honors a preferred CUDA index and falls back to MPS or CPU.
def get_device(
    preferred_idx: int | None = None,
) -> torch.device:
    """Selects the best available device based on availability and preference."""
    if preferred_idx is not None and not isinstance(preferred_idx, int):
        raise TypeError(f"preferred_idx must be int or None; got {type(preferred_idx)}.")

    if torch.cuda.is_available():
        if torch.cuda.device_count() == 0:
            _get_logger().debug(
                f"No CUDA devices available; falling back to {'MPS device.' if torch.backends.mps.is_available() else 'CPU.'}"
            )
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if preferred_idx is not None:
            if 0 <= preferred_idx < torch.cuda.device_count():
                _get_logger().debug(f"Using preferred CUDA device: {preferred_idx}")
                return torch.device(f"cuda:{preferred_idx}")
            else:
                _get_logger().debug(
                    f"Invalid preferred_idx {preferred_idx}; falling back to default CUDA."
                )
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        _get_logger().debug("No CUDA devices available; falling back to MPS device.")
        return torch.device("mps")
    _get_logger().debug("No CUDA devices available; falling back to CPU.")
    return torch.device("cpu")


# Context manager that enables Automatic Mixed Precision flexibly across environments.
@contextmanager
def autocast(
    enabled: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_type: Literal["cuda", "cpu", "mps"] | None = None,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"

    supported = False
    if device_type == "cuda" and torch.cuda.is_available():
        supported = True
    elif device_type == "cpu":
        if dtype == torch.bfloat16:
            supported = True
        else:
            _get_logger().debug(
                f"CPU AMP supports only bfloat16; got {dtype}. Falling back to no-op."
            )
    elif device_type == "mps" and torch.backends.mps.is_available():
        supported = True
    else:
        _get_logger().debug(
            f"AMP not supported for device_type '{device_type}'. Falling back to no-op."
        )

    if supported:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield

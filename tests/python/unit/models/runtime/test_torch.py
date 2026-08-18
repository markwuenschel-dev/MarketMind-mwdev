# tests/python/unit/models/runtime/test_torch.py
import logging
import random
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pysrc.models.runtime.torch import (
    autocast,
    get_device,
    init_weights,
    seed_everything,
    set_perf_flags,
)

pytestmark = [pytest.mark.determinism("d0"), pytest.mark.usefixtures("deterministic_seed")]


@pytest.fixture
def sample_net():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 3)
            self.linear = nn.Linear(10, 5)
            self.bn = nn.BatchNorm1d(10)
            self.ln = nn.LayerNorm(10)
            self.embed = nn.Embedding(100, 10)
            self.lstm = nn.LSTM(10, 20)
            self.gru = nn.GRU(10, 20)

        def forward(self, x):
            return x

    return Net()


@pytest.mark.parametrize(
    "flags",
    [
        {
            "allow_tf32": True,
            "matmul_precision": "high",
            "allow_fp16_reduced_precision_reduction": False,
        },
        {"allow_tf32": False},
    ],
)
def test_set_perf_flags(flags, caplog):
    caplog.set_level(logging.DEBUG)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.set_float32_matmul_precision") as sp,
    ):
        set_perf_flags(flags)
        if "matmul_precision" in flags:
            sp.assert_called()


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("deterministic", [True, False])
def test_seed_everything_repro(seed, deterministic):
    def draw():
        return (random.random(), np.random.rand(), torch.rand(1).item())

    seed_everything(seed, deterministic=deterministic)
    s1 = draw()
    seed_everything(seed, deterministic=deterministic)
    s2 = draw()
    assert s1 == s2


@pytest.mark.parametrize("mode", ["xavier_uniform", "kaiming_normal", "orthogonal"])
def test_init_weights(sample_net, mode):
    before = sample_net.conv.weight.clone()
    init_weights(sample_net, mode=mode)
    assert not torch.allclose(sample_net.conv.weight, before)


@pytest.mark.parametrize(
    ("has_cuda", "cuda_n", "has_mps"),
    [(True, 1, False), (False, 0, True), (False, 0, False)],
)
def test_get_device(has_cuda, cuda_n, has_mps, caplog):
    caplog.set_level(logging.INFO)
    with (
        patch("torch.cuda.is_available", return_value=has_cuda),
        patch("torch.cuda.device_count", return_value=cuda_n),
        patch("torch.backends.mps.is_available", return_value=has_mps),
    ):
        dev = get_device(None)
        assert dev.type in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize(
    ("enabled", "dtype", "device"),
    [(True, torch.float16, "cpu"), (False, torch.bfloat16, None)],
)
def test_autocast_ctx(enabled, dtype, device):
    with autocast(enabled=enabled, dtype=dtype, device_type=device):
        x = torch.ones(1)
        assert x.numel() == 1

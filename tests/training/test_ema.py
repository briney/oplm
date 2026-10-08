"""EMA tracker semantics and config validation (docs/TRAIN.md §16, `train.ema_decay`)."""

from __future__ import annotations

import pytest
import torch

from oplm.config import TrainConfig
from oplm.training.ema import EMA_HF_DIRNAME, EMA_SIDECAR_NAME, build_ema, sync_ema_buffers


def test_build_ema_first_update_copies_then_lerps() -> None:
    torch.manual_seed(0)
    live = torch.nn.Linear(4, 2)
    ema = build_ema(live, decay=0.5)
    assert int(ema.n_averaged) == 0

    ema.update_parameters(live)
    assert int(ema.n_averaged) == 1
    assert torch.equal(ema.module.weight, live.weight)
    assert ema.module.weight.data_ptr() != live.weight.data_ptr()  # independent copy

    with torch.no_grad():
        live.weight.add_(1.0)
    ema.update_parameters(live)
    assert int(ema.n_averaged) == 2
    assert torch.allclose(ema.module.weight, live.weight - 0.5)

    assert set(ema.state_dict()) == {"n_averaged", "module.weight", "module.bias"}
    assert (EMA_SIDECAR_NAME, EMA_HF_DIRNAME) == ("ema.pt", "hf_ema")


def test_sync_ema_buffers_copies_live_buffers() -> None:
    live = torch.nn.BatchNorm1d(3)
    ema = build_ema(live, decay=0.9)
    with torch.no_grad():
        live.running_mean.fill_(7.0)
    sync_ema_buffers(ema, live)
    assert torch.equal(ema.module.running_mean, live.running_mean)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_ema_decay_must_be_in_open_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(ema_decay=bad)


def test_ema_refused_under_hsdp() -> None:
    with pytest.raises(ValueError, match="hsdp"):
        TrainConfig(parallelism="hsdp", ema_decay=0.999)
    TrainConfig(ema_decay=0.999)  # ddp default: accepted

"""DeviceDataLoader moves dict batches field by field and object batches through their ``.to``."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from oplm.data import DeviceDataLoader


class _Batch:
    def __init__(self, x: torch.Tensor) -> None:
        self.x = x
        self.moved_to: torch.device | None = None

    def to(self, device: torch.device, *, non_blocking: bool = False) -> _Batch:
        out = _Batch(self.x.to(device, non_blocking=non_blocking))
        out.moved_to = device
        return out


def test_dict_batches_move_tensors_only() -> None:
    loader = DataLoader([{"a": torch.zeros(2), "ids": ["x"]}], batch_size=None)
    (batch,) = list(DeviceDataLoader(loader, torch.device("cpu")))
    assert batch["a"].tolist() == [0.0, 0.0] and batch["ids"] == ["x"]


def test_object_batches_use_their_to_method() -> None:
    loader = DataLoader([_Batch(torch.ones(1)), _Batch(torch.zeros(1))], batch_size=None)
    out = list(DeviceDataLoader(loader, torch.device("cpu")))
    assert [b.moved_to for b in out] == [torch.device("cpu")] * 2
    assert [b.x.item() for b in out] == [1.0, 0.0]

"""End-to-end training tests with torch.compile enabled."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@_REQUIRES_CUDA
def test_compile_default_mode_trains(
    tmp_path: Path, training_parquet: Path, reset_dynamo: None
) -> None:
    """compile=True, mode='default': loss is finite across all logged steps."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        output_dir=tmp_path,
        train_data=training_parquet,
        max_steps=5,
        log_every=1,
        compile=True,
        compile_mode="default",
    )
    cb = FullRecordingCallback()
    Trainer(cfg, callbacks=[cb]).train()

    assert len(cb.train_logs) == 5
    for _, metrics in cb.train_logs:
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))


@_REQUIRES_CUDA
def test_compile_checkpoint_hf_export(
    tmp_path: Path, training_parquet: Path, reset_dynamo: None
) -> None:
    """compile=True: the HF export under checkpoint-N/hf/ loads via from_pretrained."""
    from oplm.model import OplmForMaskedLM
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        output_dir=tmp_path,
        train_data=training_parquet,
        max_steps=3,
        save_every=3,
        compile=True,
    )
    Trainer(cfg).train()

    hf_dir = tmp_path / "checkpoint-3" / "hf"
    assert hf_dir.exists(), "HF export directory missing"
    model = OplmForMaskedLM.from_pretrained(str(hf_dir))
    assert model is not None

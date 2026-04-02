"""Tests for inference config resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator

from oplm.config import load_config
from oplm.inference import resolve_inference_config
from oplm.training.checkpoint import save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def test_checkpoint_inference_uses_unified_sequence_length(tmp_path: Path) -> None:
    """Checkpoint-backed inference should see the same sequence length as training."""
    cfg = load_config(
        [
            "model.hidden_dim=64",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.num_kv_heads=2",
            "model.max_seq_len=96",
            f"train.output_dir={tmp_path}",
            "train.wandb_enabled=false",
            "train.mixed_precision=\"no\"",
        ]
    )

    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=1,
        epoch=0,
        samples_seen=0,
        tokens_seen=0,
        save_total_limit=1,
    )

    config_text = (tmp_path / "checkpoint-1" / "config.yaml").read_text()
    assert "max_length" not in config_text

    resolved = resolve_inference_config(tmp_path / "checkpoint-1")

    assert resolved.model.max_seq_len == 96

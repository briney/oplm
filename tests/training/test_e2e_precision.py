"""G5 — BF16 mixed-precision training on real FSDP2 (docs/TESTING_E2E.md §5). CUDA-only.

In the native stack, mixed precision is supplied by FSDP2's
``MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`` rather than an
Accelerate autocast/GradScaler. This guards that path end to end with real
sharding (``fsdp_sharding_strategy="full"``): a bf16 run must complete with finite
loss, actually reduce eval loss over the run, and write a checkpoint whose ``hf/``
export reloads via ``from_pretrained`` and produces finite logits. (The legacy
fp16 path is gone — ``precision`` is ``bf16`` or ``fp8``; fp8 lives in the
Blackwell-gated ``test_e2e_fsdp2`` suite.)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 mixed precision requires CUDA"),
]


def test_bf16_fsdp2_run_learns_and_checkpoints(training_parquet: Path, tmp_path: Path) -> None:
    """A bf16 FSDP2 run stays finite, reduces eval loss, and writes a loadable checkpoint."""
    from oplm.model import OplmForMaskedLM
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=40,
        batch_size=8,
        lr=1e-3,
        precision="bf16",
        fsdp_sharding_strategy="full",
        max_grad_norm=1.0,  # realistic setup; also exercises clipping under bf16 params
        log_every=10,
        eval={
            "hd": {
                "path": str(training_parquet),
                "type": "sequence",
                "every": {"steps": 1000, "at_start": True, "at_end": True},
            }
        },
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    # Every logged train loss is finite (no NaN from bf16 compute).
    train_losses = [m["train/loss"] for _, m in callback.train_logs]
    assert train_losses and all(math.isfinite(v) for v in train_losses)

    # Eval loss decreased from its at_start value to its at_end value.
    eval_losses = [m["eval/hd/loss"] for _, m in callback.evals]
    assert len(eval_losses) >= 2
    assert all(math.isfinite(v) for v in eval_losses)
    assert eval_losses[-1] < eval_losses[0]

    # The final checkpoint's HF export reloads and runs a finite forward.
    hf_dir = tmp_path / "checkpoint-40" / "hf"
    assert (hf_dir / "model.safetensors").exists()
    reloaded = OplmForMaskedLM.from_pretrained(str(hf_dir)).eval()
    with torch.no_grad():
        input_ids = torch.randint(0, reloaded.config.vocab_size, (2, 8))
        logits = reloaded(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    assert logits.shape == (2, 8, reloaded.config.vocab_size)
    assert torch.isfinite(logits).all()

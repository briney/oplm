"""Bounded two-rank looping parity and real Trainer lifecycle worker."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from oplm.training.trainer import Trainer


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _parity(trainer: Trainer) -> dict[str, object]:
    from oplm.model import OplmForMaskedLM, OplmTokenizerFast
    from oplm.training.optim import build_optimizers

    raw = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    reference = OplmForMaskedLM(copy.deepcopy(raw.config)).to(trainer.accelerator.device)
    reference.gradient_checkpointing_disable()
    reference.load_state_dict(
        get_model_state_dict(raw, options=StateDictOptions(full_state_dict=True))
    )
    reference_optimizers = build_optimizers(reference, trainer.cfg.train)
    params = [
        p for opt in trainer.optimizers for group in opt.param_groups for p in group["params"]
    ]
    assert len(params) == len({id(p) for p in params})
    batch = OplmTokenizerFast()(["MEEPQ", "LAGVS"], padding=True, return_tensors="pt")
    batch = {key: value.to(trainer.accelerator.device) for key, value in batch.items()}
    batch["labels"] = batch["input_ids"].clone()
    accumulation = trainer.cfg.train.gradient_accumulation_steps
    for _ in range(accumulation):
        (reference(**batch).loss / accumulation).backward()
        with trainer.accelerator.accumulate(trainer.model):
            actual = trainer.model(**batch)
            trainer.accelerator.backward(actual.loss)
    max_grad_diff = 0.0
    for name, expected in reference.named_parameters():
        grad = raw.get_parameter(name).grad
        assert (grad is None) == (expected.grad is None), name
        if grad is not None:
            actual_grad = _full(grad)
            torch.testing.assert_close(actual_grad, expected.grad, rtol=1e-4, atol=1e-5, msg=name)
            max_grad_diff = max(max_grad_diff, (actual_grad - expected.grad).abs().max().item())
    for opt in reference_optimizers:
        opt.step()
    for opt in trainer.optimizers:
        opt.step()
    for name, expected in reference.named_parameters():
        torch.testing.assert_close(
            _full(raw.get_parameter(name)), expected, rtol=1e-4, atol=1e-5, msg=name
        )
    return {
        "max_gradient_difference": max_grad_diff,
        "unique_optimizer_parameters": len(params),
        "accumulation": accumulation,
    }


def main(config_path: str, result_dir: str) -> None:
    """Train from a resolved YAML, or verify gradients when --parity is supplied."""
    from oplm.config import load_config
    from oplm.training.trainer import Trainer
    from tests.training.conftest import FullRecordingCallback

    cfg = load_config(["--config", config_path])
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    raw = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    rank = trainer.accelerator.process_index
    payload = {
        "resumed_from_step": trainer.global_step,
        "num_unique_layers": len(raw.oplm.backbone.layers),
        "effective_depth": len(raw.oplm.backbone.layer_execution_order),
        "is_sharded": isinstance(next(raw.parameters()), DTensor),
    }
    if "--parity" in sys.argv:
        payload.update(_parity(trainer))
        torch.distributed.barrier()
        trainer.accelerator.end_training()
    else:
        state = get_model_state_dict(
            raw, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if rank == 0:
            torch.save(state, Path(result_dir, "initial.pt"))
        trainer.train()
        payload["global_step"] = trainer.global_step
        losses = [m["train/loss"] for _, m in callback.train_logs]
        payload["loss_count"] = len(losses)
        payload["all_losses_finite"] = all(math.isfinite(loss) for loss in losses)
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except (ValueError, RuntimeError) as exc:
        import os

        Path(sys.argv[2], f"rank{os.environ.get('RANK', '0')}.error").write_text(str(exc))
        raise

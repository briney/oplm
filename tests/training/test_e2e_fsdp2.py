"""Single-rank FSDP2 end-to-end smoke tests (TODOS.md Phase 6.5).

These drive the real :class:`~oplm.training.trainer.Trainer` with
``fsdp_sharding_strategy="full"`` so the ``fully_shard`` per-block-then-root path,
the ``MixedPrecisionPolicy``, and (for FP8) the torchao conversion + post-step
dynamic-scale precompute all execute. A single rank still exercises every code
path — the all-gather / reduce-scatter just operate over a one-rank mesh. The BF16
rows require CUDA (the FSDP2 device mesh is CUDA-only); the FP8 row additionally
requires sm90+ via ``@pytest.mark.blackwell``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FSDP2 device mesh requires CUDA"
)


@pytest.mark.slow
@_REQUIRES_CUDA
def test_single_rank_bf16_fsdp2(training_parquet: Path, tmp_path: Path) -> None:
    """Single-rank FSDP2 + BF16: a few steps complete with finite loss."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        log_every=1,
        precision="bf16",
        fsdp_sharding_strategy="full",
    )
    cb = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[cb])
    trainer.train()

    assert trainer.global_step == 3
    losses = [m["train/loss"] for _, m in cb.train_logs]
    assert len(losses) == 3
    assert all(math.isfinite(v) for v in losses)


@pytest.mark.slow
@_REQUIRES_CUDA
def test_single_rank_bf16_fsdp2_compile(
    training_parquet: Path, tmp_path: Path, reset_dynamo: None
) -> None:
    """Single-rank FSDP2 + BF16 + compile: compilation does not break training."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        log_every=1,
        precision="bf16",
        fsdp_sharding_strategy="full",
        compile=True,
    )
    cb = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[cb])
    trainer.train()

    assert trainer.global_step == 3
    losses = [m["train/loss"] for _, m in cb.train_logs]
    assert len(losses) == 3
    assert all(math.isfinite(v) for v in losses)


@pytest.mark.slow
@pytest.mark.blackwell
def test_single_rank_fp8_fsdp2(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_dynamo: None
) -> None:
    """Single-rank FSDP2 + FP8: loss stays finite and the post-step FP8 sync runs."""
    import oplm.training.precision as precision_mod
    from oplm.training.trainer import Trainer

    # Count the dynamic-scale precompute calls the trainer makes after optimizer.step().
    # The trainer does `from oplm.training.precision import sync_fp8_history` each step,
    # which re-binds from this module, so patching the attribute here intercepts it.
    sync_calls = {"n": 0}
    original_sync = precision_mod.sync_fp8_history

    def counting_sync(model: object) -> None:
        sync_calls["n"] += 1
        original_sync(model)

    monkeypatch.setattr(precision_mod, "sync_fp8_history", counting_sync)

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        log_every=1,
        precision="fp8",
        fsdp_sharding_strategy="full",
        compile=True,
    )
    cb = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[cb])
    trainer.train()

    assert trainer.global_step == 3
    losses = [m["train/loss"] for _, m in cb.train_logs]
    assert len(losses) == 3
    assert all(math.isfinite(v) for v in losses)
    # FP8 sync fires once per optimizer step.
    assert sync_calls["n"] == 3

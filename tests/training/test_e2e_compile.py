"""End-to-end training tests with torch.compile enabled."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Regression test: alpha must be a tensor buffer for DDP+compile compatibility
# ---------------------------------------------------------------------------


def test_compile_aot_eager_forward_cpu(reset_dynamo: None) -> None:
    """torch.compile with aot_eager backend must trace OplmForMaskedLM without error.

    The aot_eager backend exercises the AOT autograd code path (runtime_wrappers)
    that fails when any graph output is a plain Python float instead of a tensor.
    This previously broke with DDP+compile because OplmBlock.alpha was a plain float
    that got lifted as a graph input and placed in subgraph outputs by DDPOptimizer.
    """
    from oplm.model import OplmConfig as OplmModelConfig
    from oplm.model import OplmForMaskedLM

    config = OplmModelConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    model = OplmForMaskedLM(config)
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (2, 8))

    output = compiled(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_compile_selective_checkpointing_aot_eager_forward_cpu(reset_dynamo: None) -> None:
    """torch.compile must trace a model using selective activation checkpointing.

    Selective checkpointing inserts a create_selective_checkpoint_contexts
    context_fn into torch.utils.checkpoint; this exercises that the SAC policy
    composes with AOT autograd (aot_eager) and that no graph output regresses to
    a plain Python float (the alpha-buffer invariant).
    """
    from oplm.model import OplmConfig as OplmModelConfig
    from oplm.model import OplmForMaskedLM

    config = OplmModelConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=64,
        gradient_checkpointing=True,
        gradient_checkpointing_mode="selective",
    )
    model = OplmForMaskedLM(config).train()
    # Checkpointing only fires in training mode; make the mode explicit in case the
    # HF auto-enable path defaulted it to "full" during __init__.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": "selective"})
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (2, 8))

    output = compiled(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    output.loss.backward()
    assert output.loss is not None
    assert torch.isfinite(output.loss)


_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


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

    # The compile path disables Inductor's mix-order-reduction pass to dodge the
    # FusedMixOrderReductions backward-compile crash (pytorch/pytorch#169811).
    from torch._inductor import config as inductor_config

    if hasattr(inductor_config.triton, "mix_order_reduction"):
        assert inductor_config.triton.mix_order_reduction is False

    assert len(cb.train_logs) == 5
    for _, metrics in cb.train_logs:
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))


@_REQUIRES_CUDA
def test_compile_with_stability_diagnostics_trains(
    tmp_path: Path, training_parquet: Path, reset_dynamo: None
) -> None:
    """compile=True + stability diagnostics: training runs and emits finite ``diag/*``.

    Confirms the hook-free diagnostics coexist with ``torch.compile`` — the
    compiled training step carries no forward hooks, and the periodic probe runs
    eagerly on the ``unwrap_model``-peeled module. Grad norm (per step), residual
    RMS, logit RMS, and attention entropy (per probe) must all be finite.
    """
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        output_dir=tmp_path,
        train_data=training_parquet,
        max_steps=5,
        log_every=1,
        max_grad_norm=1.0,  # enable clipping so diag/grad_norm is captured
        compile=True,
        compile_mode="default",
    )
    cfg.train.stability_diagnostics = True
    cfg.train.stability_probe_every = 1

    trainer = Trainer(cfg)
    logged: list[dict[str, float]] = []
    real_log = trainer.accelerator.log

    def _spy_log(metrics: dict[str, float], step: int | None = None, **kwargs: object) -> None:
        logged.append(dict(metrics))
        return real_log(metrics, step=step, **kwargs)

    trainer.accelerator.log = _spy_log  # type: ignore[method-assign]  # test spy
    trainer.train()

    diag = {k: v for metrics in logged for k, v in metrics.items() if k.startswith("diag/")}
    for key in ("diag/grad_norm", "diag/residual_rms/max", "diag/logit_rms"):
        assert key in diag, f"missing {key}"
    assert all(math.isfinite(v) for v in diag.values())


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


@_REQUIRES_CUDA
def test_compile_dynamic_false_pad_to_multiple_runs(
    tmp_path: Path, training_parquet: Path, reset_dynamo: None
) -> None:
    """compile=True + compile_dynamic=False + pad_to_multiple_of=16: static-bucket path runs.

    With compile_dynamic=False the trainer specializes a static graph per concrete
    sequence length.  pad_to_multiple_of=16 bounds the shape space to a small
    number of buckets so the Dynamo cache_size_limit rail does not trip.  This test
    proves the end-to-end path — static-bucket compile + bucketed collation — runs
    without error and produces finite losses.
    """
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        output_dir=tmp_path,
        train_data=training_parquet,
        max_steps=3,
        log_every=1,
        compile=True,
        compile_mode="default",
        compile_dynamic=False,
        pad_to_multiple_of=16,
    )
    cb = FullRecordingCallback()
    Trainer(cfg, callbacks=[cb]).train()

    assert len(cb.train_logs) == 3
    for _, metrics in cb.train_logs:
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))

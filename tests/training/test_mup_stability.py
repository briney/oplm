"""Unit tests for :class:`StabilityDiagnosticsCallback`.

These exercise the callback directly against a tiny model and a stub trainer
(no Accelerate). The design is hook-free: the grad norm is logged every training
log, and a periodic main-process-only eager probe forward supplies the per-depth
residual RMS and logit RMS. Tests assert those two paths, the main-process
gating (distributed-safety), the eval-log skip, the probe cadence, and the config
defaults/validation.
"""

from __future__ import annotations

import pytest
import torch

from oplm.config import TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.mup import StabilityDiagnosticsCallback

_WIDTH = 128
_HEADS = _WIDTH // 64  # head_dim held at 64
_LAYERS = 3
_VOCAB = 33
_SEQ = 16

_PROBE_KEYS = (
    "diag/residual_rms/max",
    "diag/residual_rms/mean",
    "diag/residual_rms/argmax_layer",
    "diag/residual_rms/final_layer",
    "diag/logit_rms",
)


def _model() -> OplmForMaskedLM:
    torch.manual_seed(0)
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=_WIDTH,
            num_attention_heads=_HEADS,
            num_hidden_layers=_LAYERS,
            vocab_size=_VOCAB,
            max_position_embeddings=64,
        )
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "input_ids": torch.randint(0, _VOCAB, (2, _SEQ)),
        "attention_mask": torch.ones(2, _SEQ, dtype=torch.long),
    }


class _StubAccelerator:
    def __init__(self, *, is_main_process: bool = True) -> None:
        self.device = torch.device("cpu")
        self.is_main_process = is_main_process
        self.logged: list[tuple[int, dict[str, float]]] = []

    def unwrap_model(self, model: OplmForMaskedLM) -> OplmForMaskedLM:
        return model

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((step, dict(metrics)))


class _StubTrainer:
    def __init__(
        self,
        model: OplmForMaskedLM,
        dataloader: object | None = None,
        *,
        is_main_process: bool = True,
    ) -> None:
        self.model = model
        self.accelerator = _StubAccelerator(is_main_process=is_main_process)
        self.dataloader = dataloader
        self._last_grad_norm = torch.tensor(1.23)


def test_grad_norm_logged_every_train_log() -> None:
    """probe_every=0 logs the grad norm and nothing else (no probe forward)."""
    trainer = _StubTrainer(_model())
    cb = StabilityDiagnosticsCallback(probe_every=0)

    cb.on_train_start(trainer)
    cb.on_log(trainer, {"train/loss": 2.0}, step=10)

    step, metrics = trainer.accelerator.logged[-1]
    assert step == 10
    assert metrics["diag/grad_norm"] == torch.tensor(1.23).item()
    assert not any(key in metrics for key in _PROBE_KEYS)


def test_probe_emits_residual_and_logit_signals() -> None:
    """A fired probe emits per-depth residual RMS and logit RMS (plus grad norm)."""
    batch = _batch()
    trainer = _StubTrainer(_model(), dataloader=[batch])
    cb = StabilityDiagnosticsCallback(probe_every=1)

    cb.on_train_start(trainer)  # captures the probe batch
    cb.on_log(trainer, {"train/loss": 2.0}, step=1)

    _step, metrics = trainer.accelerator.logged[-1]
    for key in _PROBE_KEYS:
        assert key in metrics, f"missing {key}"
    assert "diag/grad_norm" in metrics
    assert 0 <= metrics["diag/residual_rms/argmax_layer"] <= _LAYERS  # index into (L+1)-tuple
    # Attention entropy was dropped (output_attentions manual path was unsafe under DDP).
    assert not any(key.startswith("diag/attn_") for key in metrics)


def test_probe_skipped_on_non_main_process() -> None:
    """On a non-main rank the probe never runs (distributed-safety: no extra forward)."""
    trainer = _StubTrainer(_model(), dataloader=[_batch()], is_main_process=False)
    cb = StabilityDiagnosticsCallback(probe_every=1)

    cb.on_train_start(trainer)  # must not capture a probe batch off-main
    assert cb._probe_batch is None
    cb.on_log(trainer, {"train/loss": 2.0}, step=1)

    _step, metrics = trainer.accelerator.logged[-1]
    assert not any(key in metrics for key in _PROBE_KEYS)  # no probe metrics
    assert "diag/grad_norm" in metrics  # grad norm still emitted


def test_registers_no_forward_hooks() -> None:
    """The callback attaches zero forward hooks, so a compiled training step is untouched.

    This is the invariant that lets ``torch.compile`` stay on: if diagnostics were
    ever reimplemented with module hooks they would graph-break the compiled
    forward, and this guard would fail.
    """
    model = _model()
    trainer = _StubTrainer(model, dataloader=[_batch()])
    cb = StabilityDiagnosticsCallback(probe_every=1)

    cb.on_train_start(trainer)
    cb.on_log(trainer, {"train/loss": 2.0}, step=1)  # runs a probe forward

    total_hooks = sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
        for m in model.modules()
    )
    assert total_hooks == 0


def test_eval_only_log_is_skipped() -> None:
    """on_log with no ``train/loss`` key emits nothing (eval-only cadence)."""
    trainer = _StubTrainer(_model(), dataloader=[_batch()])
    cb = StabilityDiagnosticsCallback(probe_every=1)

    cb.on_train_start(trainer)
    cb.on_log(trainer, {"eval/heldout/loss": 1.5}, step=5)

    assert trainer.accelerator.logged == []


def test_probe_respects_cadence() -> None:
    """With probe_every=2, the probe fires on the 2nd train log, not the 1st."""
    trainer = _StubTrainer(_model(), dataloader=[_batch()])
    cb = StabilityDiagnosticsCallback(probe_every=2)
    cb.on_train_start(trainer)

    cb.on_log(trainer, {"train/loss": 2.0}, step=1)
    _s1, first = trainer.accelerator.logged[-1]
    assert "diag/logit_rms" not in first  # only grad norm on the 1st log

    cb.on_log(trainer, {"train/loss": 2.0}, step=2)
    _s2, second = trainer.accelerator.logged[-1]
    assert "diag/logit_rms" in second  # probe fired on the 2nd log


def test_missing_grad_norm_is_omitted() -> None:
    """When the trainer never captured a grad norm, the key is simply absent."""
    trainer = _StubTrainer(_model())
    trainer._last_grad_norm = None  # e.g. max_grad_norm <= 0
    cb = StabilityDiagnosticsCallback(probe_every=0)

    cb.on_train_start(trainer)
    cb.on_log(trainer, {"train/loss": 2.0}, step=1)

    # No grad norm and no probe → nothing to log.
    assert trainer.accelerator.logged == []


def test_negative_probe_every_rejected() -> None:
    with pytest.raises(ValueError, match="probe_every"):
        StabilityDiagnosticsCallback(probe_every=-1)


def test_config_defaults_off_and_validated() -> None:
    cfg = TrainConfig()
    assert cfg.stability_diagnostics is False
    assert cfg.stability_probe_every == 25
    with pytest.raises(ValueError, match="stability_probe_every"):
        TrainConfig(stability_probe_every=-1)

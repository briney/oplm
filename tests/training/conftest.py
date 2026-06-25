"""Shared harness for the end-to-end training tests (docs/TESTING_E2E.md §4).

These helpers drive the *real* :class:`~oplm.training.trainer.Trainer` over a tiny
model and real data and assert observable contracts. The pieces are deliberately
importable (``from tests.training.conftest import ...``) as well as available as
fixtures, because :data:`DEVICE_PRECISION_PARAMS` is referenced at decoration time
by ``@pytest.mark.parametrize`` and :class:`FullRecordingCallback` is instantiated
multiple times per test (e.g. one per trainer in resume-equivalence runs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.training import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from oplm.training.trainer import Trainer


class FullRecordingCallback(TrainerCallback):
    """Record every trainer event so one run can be asserted against all cadences.

    Stores ``(step, metrics)`` for the metric-bearing events and a running count
    for the bracketing lifecycle events. ``on_log`` fires for *both* train-metric
    payloads (``train/*`` keys) and eval payloads (``eval/*`` keys), since the
    trainer routes eval metrics through ``_log_metrics`` too — use
    :attr:`train_logs` to isolate the train-metric emissions.
    """

    def __init__(self) -> None:
        self.train_start_count = 0
        self.train_end_count = 0
        self.logs: list[tuple[int, dict[str, float]]] = []
        self.evals: list[tuple[int, dict[str, float]]] = []
        self.checkpoints: list[tuple[int, Path]] = []

    def on_train_start(self, trainer: Trainer) -> None:
        self.train_start_count += 1

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        self.logs.append((step, dict(metrics)))

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        self.evals.append((step, dict(metrics)))

    def on_checkpoint_saved(self, trainer: Trainer, checkpoint_dir: Path, step: int) -> None:
        self.checkpoints.append((step, checkpoint_dir))

    def on_train_end(self, trainer: Trainer) -> None:
        self.train_end_count += 1

    # -- convenience views -------------------------------------------------

    @property
    def train_logs(self) -> list[tuple[int, dict[str, float]]]:
        """Only the ``on_log`` calls that carry a train-metric payload."""
        return [(step, m) for step, m in self.logs if "train/loss" in m]

    @property
    def train_log_steps(self) -> list[int]:
        """The steps at which a train-metric payload was logged, in order."""
        return [step for step, _ in self.train_logs]

    @property
    def eval_steps(self) -> list[int]:
        """The steps at which an eval pass emitted metrics, in order."""
        return [step for step, _ in self.evals]

    @property
    def checkpoint_steps(self) -> list[int]:
        """The steps at which a checkpoint was saved, in order."""
        return [step for step, _ in self.checkpoints]


def tiny_train_cfg(
    output_dir: Path,
    train_data: Path,
    *,
    max_steps: int = 6,
    max_epochs: int | None = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    optimizer: str = "adamw",
    lr: float = 1e-3,
    min_lr: float = 0.0,
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.0,
    scheduler: str = "warmup_linear",
    warmup_steps: int = 0,
    stable_steps: int = 0,
    log_every: int = 1,
    save_every: int = 10_000,
    save_total_limit: int = 3,
    save_final: bool = True,
    eval: dict[str, object] | None = None,
    eval_every: object = None,
    resume_from: str | None = None,
    seed: int = 42,
    mixed_precision: str = "no",
    compile: bool = False,
    compile_mode: str = "default",
    compile_dynamic: bool | None = True,
    pad_to_multiple_of: int | None = None,
    wandb_enabled: bool = False,
    wandb_project: str = "oplm",
    wandb_run_name: str | None = None,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    num_hidden_layers: int = 2,
    max_position_embeddings: int = 64,
    gradient_checkpointing: bool = False,
    gradient_checkpointing_mode: str = "full",
    num_workers: int = 0,
    pin_memory: bool = False,
    mask_prob: float = 0.15,
) -> OplmConfig:
    """Build a tiny end-to-end :class:`OplmConfig` for a real trainer run.

    Defaults are chosen for fast, isolated assertions: a 2-layer/32-hidden model,
    ``wandb`` off, single-process data loading, fp32, clipping off, and a
    save cadence effectively disabled (only the final checkpoint is written) so a
    test opts into checkpointing explicitly. ``output_dir`` and ``train_data``
    are paths; everything else mirrors the like-named config field.
    """
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_mode=gradient_checkpointing_mode,
        ),
        train=TrainConfig(
            max_steps=max_steps,
            max_epochs=max_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            lr=lr,
            min_lr=min_lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            log_every=log_every,
            eval_every=eval_every,
            save_every=save_every,
            save_total_limit=save_total_limit,
            save_final=save_final,
            resume_from=resume_from,
            seed=seed,
            output_dir=str(output_dir),
            mixed_precision=mixed_precision,
            compile=compile,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        ),
        data=DataConfig(
            train=str(train_data),
            eval=eval,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mask_prob=mask_prob,
            pad_to_multiple_of=pad_to_multiple_of,
        ),
    )


def configure_accelerator_device(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the trainer's Accelerator onto ``device`` for this test.

    The :class:`~oplm.training.trainer.Trainer` constructs its own ``Accelerator``
    without a ``cpu=`` argument, so CPU is forced through the ``ACCELERATE_USE_CPU``
    environment variable (the same knob ``Accelerator(cpu=True)`` sets). The
    autouse ``_reset_accelerator_state`` fixture clears the global accelerate state
    between tests, so toggling this per-parametrization is safe.
    """
    if device == "cpu":
        monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    else:
        monkeypatch.delenv("ACCELERATE_USE_CPU", raising=False)


# (device, precision) matrix. CUDA rows skip on CPU-only machines; the cpu row is
# forced onto CPU via configure_accelerator_device even on a GPU box, so both
# code paths are exercised wherever the suite runs.
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
DEVICE_PRECISION_PARAMS = [
    pytest.param("cpu", "no", id="cpu-no"),
    pytest.param("cuda", "no", id="cuda-no", marks=_REQUIRES_CUDA),
    pytest.param("cuda", "bf16", id="cuda-bf16", marks=_REQUIRES_CUDA),
]


@pytest.fixture
def make_tiny_cfg() -> Callable[..., OplmConfig]:
    """Fixture wrapper around :func:`tiny_train_cfg` for tests that prefer injection."""
    return tiny_train_cfg


@pytest.fixture
def reset_dynamo() -> Generator[None, None, None]:
    """Clear torch.compile cache before and after a test."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def restore_optimize_ddp() -> Generator[None, None, None]:
    """Save/restore the process-global ``torch._dynamo.config.optimize_ddp`` flag.

    The trainer flips ``optimize_ddp`` off for ``selective`` checkpointing + compile.
    It is process-global, so restore the original value afterward to keep the change
    from leaking into later compile tests.
    """
    import torch._dynamo

    original = torch._dynamo.config.optimize_ddp
    try:
        yield
    finally:
        torch._dynamo.config.optimize_ddp = original

"""Shared harness for the end-to-end training tests (docs/TESTING_E2E.md §4).

These helpers drive the *real* :class:`~oplm.training.trainer.Trainer` over a tiny
model and real data and assert observable contracts. The pieces are deliberately
importable (``from tests.training.conftest import ...``) as well as available as
fixtures, because :data:`DEVICE_PARAMS` is referenced at decoration time
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
    eval: dict[str, object] | None = None,
    eval_every: object = None,
    resume_from: str | None = None,
    seed: int = 42,
    mixed_precision: str = "bf16",
    precision: str = "bf16",
    fsdp_sharding_strategy: str = "none",
    compile: bool = False,
    compile_mode: str = "default",
    wandb_enabled: bool = False,
    wandb_project: str = "oplm",
    wandb_run_name: str | None = None,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    num_hidden_layers: int = 2,
    max_position_embeddings: int = 64,
    gradient_checkpointing: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    mask_prob: float = 0.15,
) -> OplmConfig:
    """Build a tiny end-to-end :class:`OplmConfig` for a real trainer run.

    Defaults are chosen for fast, isolated assertions: a 2-layer/32-hidden model,
    ``wandb`` off, single-process data loading, fp32, clipping off, and a
    save cadence effectively disabled (only the final checkpoint is written) so a
    test opts into checkpointing explicitly. ``fsdp_sharding_strategy`` defaults to
    ``"none"`` so the run forms a single-rank process group and moves the model to
    the device without real FSDP2 sharding — the path that works on both CPU-only
    CI and a single GPU. ``output_dir`` and ``train_data`` are paths; everything
    else mirrors the like-named config field.
    """
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            gradient_checkpointing=gradient_checkpointing,
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
            resume_from=resume_from,
            seed=seed,
            output_dir=str(output_dir),
            mixed_precision=mixed_precision,
            precision=precision,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            compile=compile,
            compile_mode=compile_mode,
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
        ),
    )


def force_device(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the native Trainer onto ``device`` for this test.

    The native-FSDP2 :class:`~oplm.training.trainer.Trainer` selects its device from
    ``torch.cuda.is_available()`` (CUDA + NCCL when true, CPU + Gloo otherwise) — it
    no longer reads any Accelerate env var. To exercise the CPU/Gloo path on a GPU
    box we patch ``torch.cuda.is_available`` to ``False``; the ``"cuda"`` case leaves
    detection intact (those rows are CUDA-gated by :data:`DEVICE_PARAMS`). Pair only
    with ``fsdp_sharding_strategy="none"`` — the ``"full"`` path builds a CUDA device
    mesh and cannot run on CPU.
    """
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# Device matrix for the single-rank (``fsdp_sharding_strategy="none"``) e2e runs.
# The cpu row is forced onto CPU/Gloo via force_device even on a GPU box, and the
# cuda row is gated so it skips on CPU-only machines — both paths run wherever
# possible.
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
DEVICE_PARAMS = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=_REQUIRES_CUDA),
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

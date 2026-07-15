"""End-to-end pilot training run (Phase 5/6 trainer + checkpoint integration).

Runs the live :class:`~oplm.training.trainer.Trainer` over a tiny model and the
real ``training_parquet`` fixture for a handful of CPU steps: the loop must
complete with no shape errors, write a checkpoint, fire eval on its cadence, and
resume cleanly from a checkpoint to continue to a larger ``max_steps``.

The ``pad_to_multiple_of`` smoke test lives in
``tests/training/test_pad_to_multiple_smoke.py`` so it runs in the fast
(non-slow) suite.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.training import TrainerCallback

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.training.trainer import Trainer

pytestmark = pytest.mark.slow


class _RecordingCallback(TrainerCallback):
    """Capture every ``on_eval_end`` payload so the test can inspect eval firing."""

    def __init__(self) -> None:
        self.eval_calls: list[tuple[dict[str, float], int]] = []

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        self.eval_calls.append((dict(metrics), step))


def _cfg(
    training_parquet: Path,
    output_dir: Path,
    *,
    max_steps: int = 4,
    resume_from: str | None = None,
) -> OplmConfig:
    """Tiny end-to-end config: CPU, no wandb, sequence eval every 2 steps."""
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        ),
        train=TrainConfig(
            max_steps=max_steps,
            batch_size=4,
            warmup_steps=0,
            wandb_enabled=False,
            mixed_precision="no",
            save_every=2,
            save_total_limit=2,
            output_dir=str(output_dir),
            resume_from=resume_from,
        ),
        data=DataConfig(
            train=str(training_parquet),
            eval={"hd": {"path": str(training_parquet), "type": "sequence", "every": {"steps": 2}}},
            num_workers=0,
            pin_memory=False,
        ),
    )


def _latest_checkpoint(output_dir: Path) -> Path:
    """Return the highest-step ``checkpoint-*`` directory under ``output_dir``."""
    checkpoints = [d for d in output_dir.iterdir() if d.name.startswith("checkpoint-")]
    return max(checkpoints, key=lambda d: int(d.name.split("-", 1)[1]))


def test_pilot_train_runs_end_to_end(training_parquet: Path, tmp_path: Path) -> None:
    """A short run completes, writes a checkpoint, and fires eval with finite metrics."""
    from oplm.training.trainer import Trainer

    callback = _RecordingCallback()
    cfg = _cfg(training_parquet, tmp_path)
    Trainer(cfg, callbacks=[callback]).train()

    # A checkpoint directory was written (save_every=2 over 4 steps).
    assert any(d.name.startswith("checkpoint-") for d in tmp_path.iterdir())

    # Eval fired at least once, and every recorded metric value is finite.
    assert callback.eval_calls
    all_values = [v for metrics, _ in callback.eval_calls for v in metrics.values()]
    assert all_values
    assert all(math.isfinite(v) for v in all_values)


def test_stability_diagnostics_emit_during_real_run(training_parquet: Path, tmp_path: Path) -> None:
    """With stability_diagnostics on, a live run emits finite ``diag/*`` metrics.

    Exercises the integration path the stub unit tests cannot: auto-attachment in
    ``Trainer.__init__``, the grad-norm capture in the loop, and the periodic
    eager probe forward over a real Accelerate-prepared dataloader batch.
    """
    from oplm.training.trainer import Trainer

    cfg = _cfg(training_parquet, tmp_path, max_steps=4)
    cfg.train.log_every = 2  # ensure a training-loss log fires within the run
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
    assert "diag/residual_rms/max" in diag
    assert "diag/logit_rms" in diag
    assert "diag/grad_norm" in diag
    assert "diag/attn_entropy/mean" in diag  # Tier B probe fired
    assert all(math.isfinite(v) for v in diag.values())


def test_resume_continues_to_larger_max_steps(training_parquet: Path, tmp_path: Path) -> None:
    """Resuming from the last checkpoint restores the step count and trains to the new target."""
    from oplm.training.trainer import Trainer

    # First run to step 4, producing checkpoints.
    Trainer(_cfg(training_parquet, tmp_path, max_steps=4)).train()
    last_ckpt = _latest_checkpoint(tmp_path)

    # Fresh trainer that resumes and runs to step 6.
    resume_cfg = _cfg(training_parquet, tmp_path, max_steps=6, resume_from=str(last_ckpt))
    resumed = Trainer(resume_cfg)
    assert resumed.global_step == int(last_ckpt.name.split("-", 1)[1])

    resumed.train()
    assert resumed.global_step == 6

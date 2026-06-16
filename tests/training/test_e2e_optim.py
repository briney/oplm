"""G6 + G7 — optimizers, LR schedules, learning, and grad clipping (docs/TESTING_E2E.md §5).

Covers four trainer surfaces that fp32/AdamW/constant-LR runs never exercise:

* the two-optimizer / two-scheduler Muon ``prepare`` path (CUDA-only);
* the LR trajectory of all four schedulers, asserted against ``get_schedule_fn``;
* that a real run on real data actually reduces eval loss; and
* that ``max_grad_norm`` is wired through ``accelerator.clip_grad_norm_`` (G7).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.training.optim import get_schedule_fn
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_SCHEDULERS = ["warmup_linear", "warmup_cosine", "wsd_linear", "wsd_cosine"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Muon requires CUDA")
@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
def test_muon_two_optimizer_path(training_parquet: Path, tmp_path: Path) -> None:
    """Muon drives the two-optimizer/two-scheduler path; both optimizers step, loss finite."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=4,
        optimizer="muon",
        log_every=1,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])

    # Muon contributes a second optimizer + scheduler beyond the auxiliary AdamW.
    assert len(trainer.optimizers) == 2
    assert len(trainer.schedulers) == 2

    trainer.train()

    # Both optimizers actually stepped (each accumulated per-parameter state).
    for optimizer in trainer.optimizers:
        assert len(optimizer.state) > 0

    losses = [m["train/loss"] for _, m in callback.train_logs]
    assert losses and all(math.isfinite(v) for v in losses)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Muon requires CUDA")
@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
def test_mup_muon_two_optimizer_path(training_parquet: Path, tmp_path: Path) -> None:
    """μP+Muon drives the two-optimizer path; the aux AdamW carries a μP-scaled (lr/m) group."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=4,
        optimizer="muon",
        lr=1e-3,
        log_every=1,
        hidden_size=128,
        num_attention_heads=2,  # head_dim = 64
    )
    cfg.model.mup_enable = True
    cfg.model.mup_base_width = 64  # m = 2 at hidden 128
    cfg.train.muon_adjust_lr_fn = "original"  # the μP+Muon guard requires this
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])

    # Muon + auxiliary AdamW: two optimizers / two schedulers.
    assert len(trainer.optimizers) == 2
    assert len(trainer.schedulers) == 2
    # The auxiliary AdamW carries the new μP group: lm_head.dense at lr / m.
    aux_adamw = trainer.optimizers[1]
    assert any(g["lr"] == pytest.approx(cfg.train.lr / 2.0) for g in aux_adamw.param_groups)

    trainer.train()

    for optimizer in trainer.optimizers:
        assert len(optimizer.state) > 0  # both optimizers stepped
    losses = [m["train/loss"] for _, m in callback.train_logs]
    assert losses and all(math.isfinite(v) for v in losses)


@pytest.mark.parametrize("scheduler", _SCHEDULERS)
def test_lr_trajectory_matches_schedule(
    scheduler: str, training_parquet: Path, tmp_path: Path
) -> None:
    """The logged ``train/lr`` series matches ``get_schedule_fn`` step for step."""
    from oplm.training.trainer import Trainer

    lr = 1e-3
    min_lr = 1e-4
    warmup_steps = 3
    stable_steps = 2
    total_steps = 10

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=total_steps,
        batch_size=4,
        log_every=1,
        lr=lr,
        min_lr=min_lr,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    schedule_fn = get_schedule_fn(
        scheduler,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_ratio=min_lr / lr,
        stable_steps=stable_steps,
    )
    logged = {step: m["train/lr"] for step, m in callback.train_logs}
    assert sorted(logged) == list(range(1, total_steps + 1))
    for step, value in logged.items():
        assert value == pytest.approx(lr * schedule_fn(step), rel=1e-6, abs=1e-12)

    # Warmup ramps up; the run ends near min_lr (the schedule's decayed floor).
    assert logged[1] < logged[2] < logged[3]
    assert logged[total_steps] == pytest.approx(min_lr, rel=1e-6)
    if scheduler.startswith("wsd_"):
        # The stable plateau holds peak LR for `stable_steps` after warmup.
        assert logged[warmup_steps + 1] == pytest.approx(lr)
        assert logged[warmup_steps + stable_steps] == pytest.approx(lr)


def test_model_learns_over_run(training_parquet: Path, tmp_path: Path) -> None:
    """A ~40-step run reduces eval loss from its at_start value to its at_end value."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=40,
        batch_size=8,
        lr=1e-3,
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

    eval_losses = [m["eval/hd/loss"] for _, m in callback.evals]
    assert len(eval_losses) >= 2  # at_start + at_end
    assert all(math.isfinite(v) for v in eval_losses)
    assert eval_losses[-1] < eval_losses[0], "eval loss did not decrease over the run"


def test_grad_clipping_is_wired(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_grad_norm`` routes through ``accelerator.clip_grad_norm_`` once per opt step."""
    from accelerate import Accelerator

    from oplm.training.trainer import Trainer

    clip_calls: list[float] = []
    original = Accelerator.clip_grad_norm_

    def spy(self: Accelerator, parameters: object, max_norm: float, *args: object, **kw: object):  # noqa: ANN202
        clip_calls.append(max_norm)
        return original(self, parameters, max_norm, *args, **kw)

    monkeypatch.setattr(Accelerator, "clip_grad_norm_", spy)

    # Clipping enabled: one clip call per optimizer step, with the configured norm.
    cfg = tiny_train_cfg(
        tmp_path / "clip", training_parquet, max_steps=4, batch_size=4, max_grad_norm=1.0
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()
    assert clip_calls == [1.0, 1.0, 1.0, 1.0]
    assert all(math.isfinite(m["train/loss"]) for _, m in callback.train_logs)

    # Clipping disabled (max_grad_norm=0): clip_grad_norm_ is never invoked.
    clip_calls.clear()
    cfg_off = tiny_train_cfg(
        tmp_path / "noclip", training_parquet, max_steps=4, batch_size=4, max_grad_norm=0.0
    )
    Trainer(cfg_off).train()
    assert clip_calls == []

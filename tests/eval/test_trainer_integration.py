"""End-to-end trainer↔eval integration (gated on the trainer refactor).

This is the only test that runs the live :class:`~oplm.training.trainer.Trainer`,
which builds the model via ``OplmForMaskedLM(cfg.model)``. That call is broken
until the separate trainer-refactor effort converges the trainer onto the HF
``oplm.model.OplmConfig`` (this plan deliberately ships no ``ModelConfig →
OplmConfig`` converter — see TODOS.md Phase 0). The Phase 6 eval-integration code
(``_build_eval_context``, rank-reduced tokens, ``run_due``) is exercised in
isolation by ``test_evaluator.py`` and ``test_token_accounting.py``, so skipping
here defers only the full end-to-end run, not Phase 6 coverage.

Flip the skip off once the trainer constructs ``OplmForMaskedLM`` from the HF
``OplmConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from oplm.config import DataConfig, ModelConfig, OplmConfig, TrainConfig
from oplm.training import TrainerCallback

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.training.trainer import Trainer

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skip(
        reason="enable once the trainer constructs OplmForMaskedLM from the HF OplmConfig "
        "(TODOS.md Phase 0 / trainer refactor)"
    ),
]


class _RecordingCallback(TrainerCallback):
    """Capture every ``on_eval_end`` payload so the test can inspect eval firing."""

    def __init__(self) -> None:
        self.eval_calls: list[tuple[dict[str, float], int]] = []

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        self.eval_calls.append((dict(metrics), step))


def _cfg(training_parquet: Path, output_dir: Path, every: dict[str, int]) -> OplmConfig:
    """Tiny end-to-end config: 4 steps, sequence eval on the given cadence."""
    return OplmConfig(
        model=ModelConfig(hidden_dim=32, num_heads=4, num_kv_heads=4, num_layers=2, max_seq_len=64),
        train=TrainConfig(
            max_steps=4,
            batch_size=4,
            warmup_steps=0,
            wandb_enabled=False,
            mixed_precision="no",
            output_dir=str(output_dir),
        ),
        data=DataConfig(
            train=str(training_parquet),
            eval={"hd": {"path": str(training_parquet), "type": "sequence", "every": every}},
            num_workers=0,
            pin_memory=False,
        ),
    )


@pytest.mark.parametrize("every", [{"steps": 2}, {"tokens": 1}])
def test_trainer_fires_eval_on_cadence(
    training_parquet: Path, tmp_path: Path, every: dict[str, int]
) -> None:
    """A short run completes and the sequence eval fires on its configured cadence."""
    from oplm.training.trainer import Trainer

    callback = _RecordingCallback()
    cfg = _cfg(training_parquet, tmp_path, every)
    trainer: Any = Trainer(cfg, callbacks=[callback])
    trainer.train()

    fired_keys = [key for metrics, _ in callback.eval_calls for key in metrics]
    assert any(key.startswith("eval/hd/") for key in fired_keys)

"""End-to-end trainer↔eval integration.

This is the only test that runs the live :class:`~oplm.training.trainer.Trainer`,
which builds the model via ``OplmForMaskedLM(cfg.model)`` from the HF
``oplm.model.OplmConfig``. The Phase 6 eval-integration code
(``_build_eval_context``, rank-reduced tokens, ``run_due``) is also exercised in
isolation by ``test_evaluator.py`` and ``test_token_accounting.py``; this test
adds the full end-to-end run on top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _cfg(training_parquet: Path, output_dir: Path, every: dict[str, int]) -> OplmConfig:
    """Tiny end-to-end config: 4 steps, sequence eval on the given cadence."""
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        ),
        train=TrainConfig(
            max_steps=4,
            batch_size=4,
            warmup_steps=0,
            wandb_enabled=False,
            fsdp_sharding_strategy="none",  # single-rank, unsharded (CPU or 1 GPU)
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

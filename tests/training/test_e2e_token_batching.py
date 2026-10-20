"""Token mode end to end on CPU: derived accumulation, batch fill, empty-epoch error."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_token_mode_derives_accumulation_and_logs_fill(
    training_parquet: Path, tmp_path: Path
) -> None:
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        tokens_per_step=512,
        batch_tokens=256,
        pack_sequences=True,
    )
    callback = FullRecordingCallback()
    result_json = tmp_path / "result.json"
    trainer = Trainer(cfg, callbacks=[callback, SweepMetricsCallback(result_json)])
    trainer.train()
    assert trainer.cfg.train.gradient_accumulation_steps == 2  # 512 / (256 x 1 process)
    assert trainer.accelerator.gradient_accumulation_steps == 2
    logs = [metrics for _, metrics in callback.train_logs]
    assert len(logs) == 4
    for metrics in logs:
        assert 0.5 < metrics["train/batch_fill"] <= 1.0
        assert math.isfinite(metrics["train/loss"])
    assert 0 < trainer.tokens_seen <= 4 * 512
    assert logs[-1]["train/samples"] == trainer._samples_seen > 0
    # Sweep provenance records the token budget, not batch_size x accumulation.
    payload = json.loads(result_json.read_text())
    assert (payload["global_batch"], payload["global_batch_unit"]) == (512, "tokens")


def test_token_mode_turns_off_save_alignment(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    token = tiny_train_cfg(
        tmp_path / "t",
        training_parquet,
        num_workers=2,
        tokens_per_step=256,
        batch_tokens=256,
        pack_sequences=True,
    )
    assert Trainer(token, callbacks=[])._alignment_workers() == 1
    plain = tiny_train_cfg(tmp_path / "p", training_parquet, num_workers=2)
    assert Trainer(plain, callbacks=[])._alignment_workers() == 2


def test_token_mode_empty_epoch_raises_clearly(tiny_training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    # 16 sequences can never fill 64 rows of 64 tokens; drop_last drops the only partial batch.
    cfg = tiny_train_cfg(
        tmp_path,
        tiny_training_parquet,
        max_steps=2,
        tokens_per_step=4096,
        batch_tokens=4096,
        pack_sequences=True,
    )
    with pytest.raises(RuntimeError, match="produced no batch"):
        Trainer(cfg, callbacks=[]).train()

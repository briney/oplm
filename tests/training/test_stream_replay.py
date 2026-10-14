"""Stream replay: exact batches at any step (across an epoch boundary), source tags, CLI."""

from __future__ import annotations

import csv
import math
import os
from typing import TYPE_CHECKING

import pytest
import torch
from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import serialize_config
from oplm.data import build_train_dataloader
from oplm.model import OplmForMaskedLM
from oplm.training.stream_replay import StreamReplay, steps_per_epoch
from tests.data.conftest import write_sequence_shards
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.config import OplmConfig

WORLD = 2


def _tokens(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The unmasked tokens: labels hold the original token wherever a position was masked."""
    return torch.where(batch["labels"] != -100, batch["labels"], batch["input_ids"])


def _cfg(tmp_path: Path) -> OplmConfig:
    # Source a: 26 rows of length 8; source b: 22 rows of length 16 (tags are checkable by length).
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write_sequence_shards(tmp_path / "a", [(f"{i}", "ACDEFGHI") for i in range(26)], n_shards=3)
    b = write_sequence_shards(
        tmp_path / "b", [(f"{i}", "KLMNPQRSTVWYACDE") for i in range(22)], n_shards=2
    )
    cfg = tiny_train_cfg(tmp_path / "run", a, batch_size=2)
    cfg.data.train = {
        "a": {"path": str(a), "fraction": 0.7},
        "b": {"path": str(b), "fraction": 0.3},
    }
    cfg.data.num_workers = 2
    return cfg


def test_steps_per_epoch_counts_each_workers_partial_batch() -> None:
    # Rank 0 of 2 with 2 workers over 10 rows: workers serve rows {0,4,8} and {1,5,9}.
    assert steps_per_epoch(10, rank=0, world_size=2, num_workers=2, batch_size=2) == 4


def test_replay_matches_the_training_stream_across_an_epoch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(WORLD))
    loader = build_train_dataloader(cfg)
    baseline = []
    for epoch in range(2):  # what the trainer does: set_epoch, then a fresh iterator
        loader.dataset.set_epoch(epoch)
        baseline.extend(_tokens(b) for b in loader)
    per_epoch = steps_per_epoch(48, rank=0, world_size=WORLD, num_workers=2, batch_size=2)
    assert len(baseline) == 2 * per_epoch
    monkeypatch.delenv("RANK")  # replay sets rank and world size itself
    monkeypatch.delenv("WORLD_SIZE")
    replay = StreamReplay(cfg, rank=0, world_size=WORLD)
    assert replay.steps_per_epoch == per_epoch
    for step in (0, 3, per_epoch - 1, per_epoch, per_epoch + 2):  # one dataset, re-armed
        (batch,) = list(replay.batches(step, 1))
        assert replay.position(step) == divmod(step, per_epoch)
        assert torch.equal(_tokens(batch), baseline[step]), step
        lengths = batch["attention_mask"].sum(1)
        assert torch.equal(lengths > 12, batch["source"] == 1)  # b rows are the long ones
    assert "RANK" not in os.environ  # restored after iteration


def test_stream_replay_cli_scores_a_frozen_checkpoint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ckpt = tmp_path / "checkpoint-6"
    torch.manual_seed(0)
    OplmForMaskedLM(cfg.model).save_pretrained(ckpt / "hf")
    (ckpt / "config.yaml").write_text(serialize_config(cfg))
    out = tmp_path / "replay.csv"
    args = ["stream-replay", str(ckpt), "--world-size", str(WORLD), "--step", "0", "--step", "5"]
    args += ["--batches", "2", "--device", "cpu", "--out", str(out)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.open()))
    assert [r["step"] for r in rows] == ["0", "5"]
    for r in rows:
        assert int(r["rows"]) == 4 and math.isfinite(float(r["loss"]))
        assert float(r["frac_a"]) + float(r["frac_b"]) == pytest.approx(1.0)

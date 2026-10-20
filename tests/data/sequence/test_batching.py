"""TokenBatcher: fixed (rows, width) batches, contiguous runs of the stream, exact resume."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from torch.utils.data import DataLoader

from oplm.data import build_train_dataloader
from oplm.data.sequence.batching import TokenBatcher
from oplm.data.sequence.collate import MLMCollator, token_length
from oplm.data.sequence.dataset import ShardedProteinDataset
from oplm.data.tokenizer import get_tokenizer
from tests.data.conftest import write_sequence_shards
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

WIDTH, ROWS = 16, 2


@pytest.fixture(autouse=True)
def _single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def _dataset(tmp_path: Path, lengths: list[int], *, shuffle: bool = False) -> ShardedProteinDataset:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(exist_ok=True)
    residues = "ACDEFGHIKLMNPQRSTVWY" * 3
    records = [(f"{i}", residues[:n]) for i, n in enumerate(lengths)]
    path = write_sequence_shards(shard_dir, records, n_shards=1)
    return ShardedProteinDataset(path, shuffle_shards=shuffle, shuffle_rows=shuffle, seed=0)


def _batcher(dataset, *, drop_last: bool = True, rows: int = ROWS) -> TokenBatcher:
    collator = MLMCollator(
        get_tokenizer(),
        max_length=WIDTH,
        pack_sequences=True,
        pack_rows=rows,
        keep_sequence_ids=True,
    )
    return TokenBatcher(dataset, collator, rows=rows, width=WIDTH, drop_last=drop_last)


def test_batches_close_at_the_first_sequence_that_fits_nowhere(tmp_path: Path) -> None:
    # Token lengths 12, 12, 7, 14, 4, 3: [12 | 12] closes at 7; [7 + 4 + 3 | 14] is full; ...
    batches = list(_batcher(_dataset(tmp_path, [10, 10, 5, 12, 2, 1]), drop_last=False))
    assert [b["num_sequences"] for b in batches] == [2, 4]
    assert [b["sequence_ids"] for b in batches] == [["0", "1"], ["2", "3", "4", "5"]]
    assert [tuple(b["stream_position"]) for b in batches] == [(0, 2), (0, 6)]
    for b in batches:
        assert b["input_ids"].shape == b["segment_ids"].shape == (ROWS, WIDTH)


def test_every_batch_is_exactly_rows_by_width_and_a_run_of_the_stream(tmp_path: Path) -> None:
    lengths = [(i * 7) % 15 + 1 for i in range(60)]
    dataset = _dataset(tmp_path, lengths)
    stream = [str(r["sequence_id"]) for r in dataset]
    batches = list(_batcher(dataset))
    served = [sid for b in batches for sid in b["sequence_ids"]]
    assert served == stream[: len(served)]  # contiguous, in order, nothing skipped
    assert len(stream) - len(served) < 2 * ROWS * WIDTH  # only the dropped tail is missing
    for b in batches:
        assert b["input_ids"].shape == (ROWS, WIDTH)
        assert int(b["attention_mask"].sum()) == sum(
            token_length("x" * n, WIDTH) for n in (lengths[int(s)] for s in b["sequence_ids"])
        )


def test_final_partial_batch_follows_drop_last(tmp_path: Path) -> None:
    lengths = [10, 10, 5]  # [12 | 12] then a lone 7
    assert [b["num_sequences"] for b in _batcher(_dataset(tmp_path, lengths))] == [2]
    kept = list(_batcher(_dataset(tmp_path, lengths), drop_last=False))
    assert [b["num_sequences"] for b in kept] == [2, 1]


def test_long_sequence_takes_a_whole_row(tmp_path: Path) -> None:
    batches = list(_batcher(_dataset(tmp_path, [40, 40, 3]), drop_last=False, rows=1))
    assert [b["num_sequences"] for b in batches] == [1, 1, 1]
    assert int(batches[0]["attention_mask"].sum()) == WIDTH


def test_resume_rows_continue_each_worker_exactly(tmp_path: Path) -> None:
    lengths = [(i * 5) % 13 + 1 for i in range(200)]
    dataset = _dataset(tmp_path, lengths, shuffle=True)

    def run(batcher: TokenBatcher) -> list[tuple[int, int, list[str]]]:
        loader = DataLoader(batcher, batch_size=None, num_workers=2)
        return [
            (int(b["stream_position"][0]), int(b["stream_position"][1]), list(b["sequence_ids"]))
            for b in loader
        ]

    baseline = run(_batcher(dataset))
    cut = 7  # batches consumed before the "checkpoint"
    rows = [0, 0]
    for worker, emitted, _ in baseline[:cut]:
        rows[worker] = emitted
    resumed_batcher = _batcher(dataset)
    resumed_batcher.set_resume_rows(rows)
    resumed = run(resumed_batcher)
    for worker in range(2):
        before = [ids for w, _, ids in baseline[:cut] if w == worker]
        after = [ids for w, _, ids in baseline[cut:] if w == worker]
        assert sum(len(ids) for ids in before) == rows[worker]
        assert [ids for w, _, ids in resumed if w == worker] == after


def test_build_train_dataloader_in_token_mode(training_parquet: Path, tmp_path: Path) -> None:
    cfg = tiny_train_cfg(
        tmp_path, training_parquet, tokens_per_step=256, batch_tokens=256, pack_sequences=True
    )
    loader = build_train_dataloader(cfg)
    assert isinstance(loader.dataset, TokenBatcher) and loader.batch_size is None
    assert loader.dataset.total_length > 0
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 64)  # 256 / max_position_embeddings 64

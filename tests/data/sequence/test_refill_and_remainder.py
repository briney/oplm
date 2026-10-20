"""Reshuffled refills of an oversampled source, and equal per-rank epochs (drop_remainder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from torch.utils.data import DataLoader

from oplm.data import build_train_dataloader
from oplm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from tests.data.conftest import write_sequence_shards
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path


def _shards(root: Path, name: str, n: int, n_shards: int) -> Path:
    (root / name).mkdir()
    records = [(f"{i}", "ACDEFGHIKLMNPQRSTVWY"[: 5 + i % 9]) for i in range(n)]
    return write_sequence_shards(root / name, records, n_shards=n_shards, id_prefix=name)


def _sources(root: Path) -> list[ShardedProteinDataset]:
    small = ShardedProteinDataset(_shards(root, "a", 10, 2), seed=0)  # refills ~5 times
    big = ShardedProteinDataset(_shards(root, "b", 100, 4), seed=0)
    return [small, big]


def _mix(sources: list[ShardedProteinDataset], *, reshuffle: bool) -> InterleavedDataset:
    return InterleavedDataset(sources, [0.5, 0.5], seed=0, reshuffle_on_refill=reshuffle)


def _ids(rows: object) -> list[str]:
    return [str(r["sequence_id"]) for r in rows]  # ty: ignore[not-iterable, not-subscriptable]


@pytest.fixture(autouse=True)
def _single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


@pytest.mark.parametrize("reshuffle", [True, False])
def test_refills_reshuffle_only_when_enabled(tmp_path: Path, reshuffle: bool) -> None:
    small_ids = [i for i in _ids(_mix(_sources(tmp_path), reshuffle=reshuffle)) if i[0] == "a"]
    passes = [small_ids[i : i + 10] for i in range(0, len(small_ids) - 9, 10)]
    assert len(passes) >= 3
    assert all(sorted(p) == sorted(passes[0]) for p in passes)  # every pass is the whole source
    assert (len({tuple(p) for p in passes}) > 1) is reshuffle


def test_resume_across_reshuffled_refills_is_the_baseline_suffix(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    baseline = _ids(_mix(sources, reshuffle=True))
    for skip in (3, 10, 27, 61, len(baseline) - 1):  # lands in the first and later passes
        resumed = _mix(sources, reshuffle=True)
        resumed.set_resume_skip(batches_in_epoch=skip, per_rank_batch=1, num_workers=1)
        assert _ids(resumed) == baseline[skip:], skip


def test_resume_rows_across_reshuffled_refills_is_the_baseline_suffix(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    baseline = _ids(_mix(sources, reshuffle=True))
    for skip in (3, 10, 27, 61, len(baseline) - 1):  # lands in the first and later passes
        resumed = _mix(sources, reshuffle=True)
        resumed.set_resume_rows([skip])
        assert _ids(resumed) == baseline[skip:], skip


def test_drop_remainder_gives_every_rank_the_same_full_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 103 rows over 2 ranks x 2 workers: plain striping hands three streams 26 rows and one
    # 25; with drop_remainder every stream serves 103 // 4 = 25, i.e. 6 full batches of 4.
    ds = ShardedProteinDataset(_shards(tmp_path, "c", 103, 3), seed=0, drop_remainder=True)
    seen: list[str] = []
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        loader = DataLoader(ds, batch_size=4, num_workers=2, drop_last=True, collate_fn=list)
        batches = list(loader)
        assert [len(b) for b in batches] == [4] * 12
        seen += [i for b in batches for i in _ids(b)]
    assert len(seen) == len(set(seen)) == 96


def test_interleaved_drop_remainder_equalizes_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mix = InterleavedDataset(_sources(tmp_path), [0.5, 0.5], seed=0, drop_remainder=True)
    # 110 samples over 4 streams: 27 draws each, rather than 28 for two streams and 27 for two
    counts = []
    for rank in range(4):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "4")
        counts.append(len(list(mix)))
    assert counts == [27] * 4


def test_train_loader_applies_the_data_config(tmp_path: Path) -> None:
    cfg = tiny_train_cfg(tmp_path / "run", _shards(tmp_path, "d", 20, 2), batch_size=4)
    cfg.data.train = {
        "a": {"path": str(_shards(tmp_path, "e", 8, 1)), "fraction": 0.5},
        "b": {"path": str(_shards(tmp_path, "f", 30, 2)), "fraction": 0.5},
    }
    loader = build_train_dataloader(cfg)  # defaults: both fixes on
    assert loader.drop_last and loader.dataset._reshuffle_on_refill
    assert loader.dataset._drop_remainder and loader.dataset._datasets[0]._drop_remainder
    cfg.data.drop_last = cfg.data.reshuffle_on_refill = False  # the earlier behavior
    loader = build_train_dataloader(cfg)
    assert not loader.drop_last and not loader.dataset._reshuffle_on_refill
    assert not loader.dataset._drop_remainder

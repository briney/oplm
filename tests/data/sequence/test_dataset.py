"""Tests for the iterable sequence datasets (Phase 3).

Covers the reproducibility contract (same ``(seed, epoch)`` ⇒ identical order),
explicit rank/worker striping (disjoint, gap-free coverage), interleaved
sampling ratios + source refill, and the optional ``masking_weights`` column.

The ``(rank, worker)`` context is normally resolved from
``torch.distributed``/env/worker-info; tests monkeypatch
:func:`oplm.data.sequence.dataset._resolve_distributed_context` to drive specific
contexts deterministically in a single process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

import oplm.data.sequence.dataset as dataset_mod
from oplm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _single_process_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a single-process ``(rank=0, ws=1, worker=0, nw=1)`` context."""
    monkeypatch.setattr(dataset_mod, "_resolve_distributed_context", lambda: (0, 1, 0, 1))


def _set_context(monkeypatch: pytest.MonkeyPatch, rank: int, ws: int, worker: int, nw: int) -> None:
    """Pin the resolved sharding context for the next dataset iteration."""
    monkeypatch.setattr(
        dataset_mod,
        "_resolve_distributed_context",
        lambda: (rank, ws, worker, nw),
    )


def _all_sequence_ids(shard_dir: Path) -> list[str]:
    """Ground-truth set of every ``sequence_id`` in a shard directory."""
    ids: list[str] = []
    for shard in sorted(shard_dir.iterdir()):
        ids.extend(pq.read_table(shard, columns=["sequence_id"]).column("sequence_id").to_pylist())
    return ids


def _ids(dataset: ShardedProteinDataset) -> list[str]:
    """Materialize the ``sequence_id`` stream from a dataset iteration."""
    return [str(row["sequence_id"]) for row in dataset]


# --------------------------------------------------------------------------- #
# Construction & metadata
# --------------------------------------------------------------------------- #


def test_total_length_matches_parquet(sequence_shards: Path) -> None:
    """``len`` / ``total_length`` equal the parquet row count (read from metadata)."""
    ds = ShardedProteinDataset(sequence_shards)
    expected = len(_all_sequence_ids(sequence_shards))
    assert len(ds) == expected
    assert ds.total_length == expected


def test_missing_path_raises(tmp_path: Path) -> None:
    """A path that is neither a parquet file nor a shard directory errors clearly."""
    with pytest.raises(FileNotFoundError):
        ShardedProteinDataset(tmp_path / "does_not_exist.parquet")


def test_empty_directory_raises(tmp_path: Path) -> None:
    """A directory with no parquet shards errors."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ShardedProteinDataset(empty)


def test_single_file_is_one_shard(sequence_shards: Path) -> None:
    """A single parquet file resolves to a one-shard dataset."""
    shard = sorted(sequence_shards.iterdir())[0]
    ds = ShardedProteinDataset(shard)
    n = pq.ParquetFile(shard).metadata.num_rows
    assert len(ds) == n
    assert len(_ids(ds)) == n


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_epoch_identical_order(sequence_shards: Path) -> None:
    """Two iterations at the same ``(seed, epoch)`` yield identical row order."""
    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    first = _ids(ds)
    ds.set_epoch(0)
    second = _ids(ds)
    assert first == second


def test_different_epochs_differ(sequence_shards: Path) -> None:
    """Different epochs reorder the same rows (same multiset, different order)."""
    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    epoch0 = _ids(ds)
    ds.set_epoch(1)
    epoch1 = _ids(ds)
    assert sorted(epoch0) == sorted(epoch1)  # same rows
    assert epoch0 != epoch1  # different order


def test_no_shuffle_is_natural_order(sequence_shards: Path) -> None:
    """With shuffling off, rows come out in shard-then-row order."""
    ds = ShardedProteinDataset(sequence_shards, shuffle_shards=False, shuffle_rows=False)
    assert _ids(ds) == _all_sequence_ids(sequence_shards)


# --------------------------------------------------------------------------- #
# Striping coverage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("world_size", "num_workers"), [(1, 1), (1, 2), (2, 2)])
def test_striping_disjoint_and_complete(
    sequence_shards: Path, monkeypatch: pytest.MonkeyPatch, world_size: int, num_workers: int
) -> None:
    """Union of all ``(rank, worker)`` shards equals the dataset, with no duplicates."""
    full = _all_sequence_ids(sequence_shards)

    collected: list[str] = []
    for rank in range(world_size):
        for worker in range(num_workers):
            _set_context(monkeypatch, rank, world_size, worker, num_workers)
            ds = ShardedProteinDataset(sequence_shards, seed=0)
            ds.set_epoch(3)
            collected.extend(_ids(ds))

    assert len(collected) == len(set(collected))  # no duplicates
    assert sorted(collected) == sorted(full)  # complete coverage


# --------------------------------------------------------------------------- #
# masking_weights column
# --------------------------------------------------------------------------- #


def test_masking_weights_present(sequence_shards_weighted: Path) -> None:
    """When requested and present, weights surface aligned to the raw sequence."""
    ds = ShardedProteinDataset(sequence_shards_weighted, load_masking_weights=True)
    rows = list(ds)
    assert rows
    for row in rows:
        weights = row["masking_weights"]
        assert isinstance(weights, list)
        assert len(weights) == len(str(row["sequence"]))


def test_masking_weights_absent_yields_none(sequence_shards: Path) -> None:
    """Requested but column-absent shards surface ``masking_weights=None``."""
    ds = ShardedProteinDataset(sequence_shards, load_masking_weights=True)
    rows = list(ds)
    assert rows
    for row in rows:
        assert "masking_weights" in row
        assert row["masking_weights"] is None


def test_masking_weights_not_read_when_disabled(sequence_shards_weighted: Path) -> None:
    """With ``load_masking_weights=False`` the key is absent even if the column exists."""
    ds = ShardedProteinDataset(sequence_shards_weighted, load_masking_weights=False)
    for row in ds:
        assert "masking_weights" not in row


# --------------------------------------------------------------------------- #
# InterleavedDataset
# --------------------------------------------------------------------------- #


def test_interleaving_validation() -> None:
    """Constructor guards on empty/mismatched/negative/zero fractions."""
    dummy = _ListDataset([{"sequence_id": "a", "sequence": "AA"}])
    with pytest.raises(ValueError, match="at least one dataset"):
        InterleavedDataset([], [])
    with pytest.raises(ValueError, match="same length"):
        InterleavedDataset([dummy], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        InterleavedDataset([dummy, dummy], [-1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        InterleavedDataset([dummy, dummy], [0.0, 0.0])


def test_interleaving_ratio_matches_fractions(
    tmp_path: Path, make_sequence_shards: Callable[..., Path], real_records: list[tuple[str, str]]
) -> None:
    """Empirical source-sampling ratio over many draws matches the fractions."""
    dir0 = (tmp_path / "ds0").resolve()
    dir1 = (tmp_path / "ds1").resolve()
    dir0.mkdir()
    dir1.mkdir()
    make_sequence_shards(dir0, real_records[:8], n_shards=2, id_prefix="ds0_")
    make_sequence_shards(dir1, real_records[8:16], n_shards=2, id_prefix="ds1_")

    inter = InterleavedDataset(
        [ShardedProteinDataset(dir0, seed=0), ShardedProteinDataset(dir1, seed=0)],
        [0.75, 0.25],
        num_samples=4000,
        seed=0,
    )
    inter.set_epoch(0)

    from_ds0 = sum(1 for row in inter if str(row["sequence_id"]).startswith("ds0_"))
    total = inter.total_length
    assert total == 4000
    assert abs(from_ds0 / total - 0.75) < 0.04


def test_interleaving_refills_exhausted_source(
    tmp_path: Path, make_sequence_shards: Callable[..., Path], real_records: list[tuple[str, str]]
) -> None:
    """A small source is re-iterated so the mix keeps producing for the full epoch."""
    small = (tmp_path / "small").resolve()
    small.mkdir()
    make_sequence_shards(small, real_records[:3], n_shards=1)

    inter = InterleavedDataset(
        [ShardedProteinDataset(small, seed=0)], [1.0], num_samples=20, seed=0
    )
    inter.set_epoch(0)
    rows = list(inter)
    assert len(rows) == 20  # refilled well past the 3 underlying rows


def test_interleaving_set_epoch_propagates(
    tmp_path: Path, make_sequence_shards: Callable[..., Path], real_records: list[tuple[str, str]]
) -> None:
    """``set_epoch`` reaches sub-datasets (their order changes with the epoch)."""
    shard_dir = (tmp_path / "src").resolve()
    shard_dir.mkdir()
    make_sequence_shards(shard_dir, real_records[:12], n_shards=2)
    source = ShardedProteinDataset(shard_dir, seed=0)

    inter = InterleavedDataset([source], [1.0], num_samples=12, seed=0)
    inter.set_epoch(0)
    assert source._epoch == 0
    inter.set_epoch(5)
    assert source._epoch == 5


class _ListDataset:
    """Minimal in-memory iterable used only for constructor-validation tests."""

    def __init__(self, items: list[dict[str, str]]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

"""Tests for reshard layers in :class:`ShardedProteinDataset` streams (#63).

A reshard event records how far each old ``(rank, worker)`` stream got. Replayed as a layer, it
removes those rows from the epoch's order, and the live stride stripes what is left, so a run
resumed at a different stream count serves every row exactly once. With no events the stream
must be byte-identical to the pre-layer stream.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

import oplm.data.sequence.dataset as dataset_mod
from oplm.data import Reshard
from oplm.data.sequence.dataset import (
    _FULL,
    ShardedProteinDataset,
    _level_sizes,
    _resolve_layers,
)
from tests.data.conftest import _load_real_records, write_sequence_shards

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from oplm.data.sequence.dataset import Layer

_N_ROWS = 97  # odd, so none of the strides below divide it


@pytest.fixture(scope="module")
def records() -> list[tuple[str, str]]:
    return _load_real_records(_N_ROWS)


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory, records: list[tuple[str, str]]) -> Path:
    """97 real sequences in 4 shards (25, 25, 25, 22 rows)."""
    return write_sequence_shards(tmp_path_factory.mktemp("reshard"), records, n_shards=4)


def _stream(
    ds: ShardedProteinDataset,
    rank: int,
    world: int,
    worker: int = 0,
    workers: int = 1,
    layers: Sequence[Layer] | None = None,
    skip: int = 0,
) -> list[str]:
    """The ids one ``(rank, worker)`` stream yields; also checks it against ``stream_length``."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            dataset_mod, "_resolve_distributed_context", lambda: (rank, world, worker, workers)
        )
        ids = [str(row["sequence_id"]) for row in ds._iter_stream(skip, layers=layers)]
        assert len(ids) == max(ds.stream_length(layers) - skip, 0)
    return ids


def _streams(ds: ShardedProteinDataset, world: int) -> list[list[str]]:
    return [_stream(ds, rank, world) for rank in range(world)]


def _consume(streams: list[list[str]], rows: Sequence[int]) -> list[str]:
    """The rows each stream had served at the save: the ragged prefixes ``rows``."""
    return [row for stream, n in zip(streams, rows, strict=True) for row in stream[:n]]


# Captured from the pre-layer `_iter_stream(0)` (seed 7, epoch 2, drop_remainder=False) as indices
# into `records`: (stride, joint_index) -> stream. drop_remainder=True serves the first
# `97 // stride` of each.
_TODAYS_STREAMS = {
    (1, 0): [85, 90, 75, 86, 87, 79, 77, 81, 94, 84, 93, 80, 95, 88, 76, 82, 83, 91, 78, 89, 92,
             96, 59, 65, 56, 69, 51, 72, 62, 68, 55, 60, 53, 63, 50, 73, 61, 57, 54, 67, 70, 74,
             71, 64, 58, 66, 52, 8, 17, 6, 4, 5, 24, 10, 9, 0, 14, 19, 2, 1, 21, 11, 3, 20, 23,
             16, 13, 12, 18, 7, 22, 15, 26, 39, 29, 36, 25, 37, 45, 34, 30, 35, 44, 38, 28, 43,
             48, 49, 46, 41, 47, 31, 27, 33, 42, 32, 40],
    (3, 0): [85, 86, 77, 84, 95, 82, 78, 96, 56, 72, 55, 63, 61, 67, 71, 66, 17, 5, 9, 19, 21,
             20, 13, 7, 26, 36, 45, 35, 28, 49, 47, 33, 40],
    (3, 2): [75, 79, 94, 80, 76, 91, 92, 65, 51, 68, 53, 73, 54, 74, 58, 8, 4, 10, 14, 1, 3, 16,
             18, 15, 29, 37, 30, 38, 48, 41, 27, 32],
    (4, 0): [85, 87, 94, 95, 83, 92, 56, 62, 53, 61, 70, 58, 17, 24, 14, 21, 23, 18, 26, 25, 30,
             28, 46, 27, 40],
    (4, 3): [86, 81, 80, 82, 89, 65, 72, 60, 73, 67, 64, 8, 5, 0, 1, 20, 12, 15, 36, 34, 38, 49,
             31, 32],
}  # fmt: skip


@pytest.mark.parametrize("drop_remainder", [False, True])
def test_no_reshards_matches_todays_stream(
    shards: Path, records: list[tuple[str, str]], drop_remainder: bool
) -> None:
    index = {rid: i for i, (rid, _) in enumerate(records)}
    ds = ShardedProteinDataset(shards, seed=7, drop_remainder=drop_remainder)
    ds.set_epoch(2)
    ds.set_reshards([])
    for (stride, joint), expected in _TODAYS_STREAMS.items():
        # stride 4 as 2 ranks x 2 workers, so the joint index composes from both
        rank, worker, workers = (joint // 2, joint % 2, 2) if stride == 4 else (joint, 0, 1)
        ids = _stream(ds, rank, stride // workers, worker, workers)
        if drop_remainder:
            expected = expected[: _N_ROWS // stride]
        assert [index[i] for i in ids] == expected


@pytest.mark.parametrize(("n", "m"), [(4, 2), (2, 3), (3, 1), (1, 4)])
def test_reshard_is_exact_single_source(shards: Path, n: int, m: int) -> None:
    ds = ShardedProteinDataset(shards, seed=3)
    everything = _stream(ds, 0, 1)
    rows = [5, 0, 9, 2][:n]
    old = _consume(_streams(ds, n), rows)
    ds.set_reshards([Reshard(stride=n, epochs=[0] * n, rows=rows)])
    new = [row for stream in _streams(ds, m) for row in stream]
    assert len(new) == _N_ROWS - sum(rows)
    assert Counter(old + new) == Counter(everything)


def test_reshard_straddling_epochs(shards: Path) -> None:
    ds = ShardedProteinDataset(shards, seed=3)
    epoch0 = _streams(ds, 2)
    ds.set_epoch(1)
    epoch1 = _streams(ds, 2)
    reshard = Reshard(stride=2, epochs=[0, 1], rows=[3, 4])
    ds.set_reshards([reshard])

    # Stream 1 had moved on to epoch 1, so all of its epoch-0 rows count as consumed.
    ds.set_epoch(0)
    assert Counter(row for s in _streams(ds, 3) for row in s) == Counter(epoch0[0][3:])
    # Stream 0 had not reached epoch 1: only stream 1's first 4 epoch-1 rows are gone.
    ds.set_epoch(1)
    served = Counter(row for s in _streams(ds, 3) for row in s)
    assert served == Counter(epoch1[0] + epoch1[1][4:])
    # The event is over by epoch 2.
    ds.set_epoch(2)
    plain = ShardedProteinDataset(shards, seed=3)
    plain.set_epoch(2)
    assert _streams(ds, 3) == _streams(plain, 3)


def test_nested_reshard_is_exact(shards: Path) -> None:
    ds = ShardedProteinDataset(shards, seed=5)
    everything = _stream(ds, 0, 1)
    first = Reshard(stride=4, epochs=[0] * 4, rows=[5, 0, 9, 2])
    phase1 = _consume(_streams(ds, 4), first.rows)
    ds.set_reshards([first])
    second = Reshard(stride=3, epochs=[0] * 3, rows=[4, 7, 0])
    phase2 = _consume(_streams(ds, 3), second.rows)
    ds.set_reshards([first, second])
    phase3 = [row for stream in _streams(ds, 2) for row in stream]
    assert Counter(phase1 + phase2 + phase3) == Counter(everything)


def test_reshard_with_more_streams_than_rows_leaves_some_empty(shards: Path) -> None:
    ds = ShardedProteinDataset(shards, seed=3)
    everything = _stream(ds, 0, 1)
    old_streams = _streams(ds, 2)
    rows = [len(old_streams[0]) - 1, len(old_streams[1]) - 2]  # 3 rows left
    old = _consume(old_streams, rows)
    ds.set_reshards([Reshard(stride=2, epochs=[0, 0], rows=rows)])
    new = _streams(ds, 8)
    assert [len(s) for s in new] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert Counter(old + [row for s in new for row in s]) == Counter(everything)
    # The next epoch is untouched by the event: every new stream is full again.
    ds.set_epoch(1)
    assert Counter(row for s in _streams(ds, 8) for row in s) == Counter(everything)


def test_reshard_drop_remainder_never_repeats(shards: Path) -> None:
    ds = ShardedProteinDataset(shards, seed=5, drop_remainder=True)
    first = Reshard(stride=4, epochs=[0] * 4, rows=[5, 0, 9, 2])
    phase1 = _consume(_streams(ds, 4), first.rows)
    ds.set_reshards([first])
    second = Reshard(stride=3, epochs=[0] * 3, rows=[4, 7, 0])
    phase2 = _consume(_streams(ds, 3), second.rows)
    ds.set_reshards([first, second])
    phase3 = [row for stream in _streams(ds, 2) for row in stream]

    served = phase1 + phase2 + phase3
    assert len(set(served)) == len(served)
    # Each level drops only its own tail: fewer rows than that level's stride.
    layers = _resolve_layers([first, second], 0)
    sizes = _level_sizes(_N_ROWS, layers, drop_remainder=True)
    assert sizes == [97, 80, 67]
    consumed = [len(phase1), len(phase2), len(phase3)]
    strides = [4, 3, 2]
    for level in range(3):
        next_size = sizes[level + 1] if level < 2 else 0
        missing = sizes[level] - consumed[level] - next_size
        assert 0 <= missing < strides[level]


def test_reshards_survive_clear_resume_skip(shards: Path) -> None:
    ds = ShardedProteinDataset(shards, seed=3)
    old = _consume(_streams(ds, 2), [6, 1])
    reshard = Reshard(stride=2, epochs=[0, 0], rows=[6, 1])
    ds.set_reshards([reshard])
    ds.set_resume_rows([2])
    ds.clear_resume_skip()
    assert ds._reshards == [reshard]
    assert set(_stream(ds, 0, 1)).isdisjoint(old)
    assert len(_stream(ds, 0, 1)) == _N_ROWS - 7


def test_resolve_layers() -> None:
    reshard = Reshard(stride=3, epochs=[0, 0, 1], rows=[4, 2, 6])
    assert _resolve_layers([reshard], 0) == [(3, [4, 2, _FULL])]
    assert _resolve_layers([reshard], 1) == [(3, [0, 0, 6])]
    assert _resolve_layers([reshard], 2) == []
    # An event that consumed nothing is an identity layer and is dropped.
    assert _resolve_layers([Reshard(stride=2, epochs=[0, 0], rows=[0, 0])], 0) == []


def test_resolve_layers_per_source() -> None:
    reshard = Reshard(
        stride=3,
        epochs=[0, 0, 1],
        rows=[0, 0, 0],
        sources=[[[1, 4], [0, 2]], None, [[0, 9], [0, 0]]],
    )

    def resolve(epoch: int, source: int, pass_index: int) -> list[Layer]:
        return _resolve_layers([reshard], epoch, source=source, pass_index=pass_index)

    # Stream 0 is in pass 1 of source 0: pass 0 is all consumed, pass 1 up to 4, pass 2 none.
    # Stream 1 never produced a batch ([0, 0]); stream 2 is in epoch 1 (all of epoch 0 gone).
    assert resolve(0, 0, 0) == [(3, [_FULL, 0, _FULL])]
    assert resolve(0, 0, 1) == [(3, [4, 0, _FULL])]
    assert resolve(0, 0, 2) == [(3, [0, 0, _FULL])]
    assert resolve(0, 1, 0) == [(3, [2, 0, _FULL])]
    assert resolve(1, 0, 0) == [(3, [0, 0, 9])]
    assert resolve(1, 1, 0) == []  # every count is 0: identity, dropped

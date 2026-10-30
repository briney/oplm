"""Tests for phase-2 member sampling in :class:`ShardedProteinDataset`.

With ``sample_members`` each served cluster row yields one member, chosen by a pure hash of
``(pass seed, shard index, row index)``, so cursors, reshards and topologies are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import oplm.data.sequence.dataset as dataset_mod
from oplm.data.sequence.dataset import (
    MEMBER_IDS_COLUMN,
    MEMBER_SEQUENCES_COLUMN,
    ShardedProteinDataset,
    _member_index,
)

if TYPE_CHECKING:
    from pathlib import Path

# Two shards of clusters with 1, 3 and 5 members; the representative is member 0.
_SIZES = [[1, 3, 5], [5, 3, 1]]


def _write(tmp_path: Path) -> Path:
    d = tmp_path / "clusters"
    d.mkdir()
    for s, sizes in enumerate(_SIZES):
        reps, seqs, mids, mseqs = [], [], [], []
        for c, n in enumerate(sizes):
            ids = [f"s{s}c{c}m{j}" for j in range(n)]
            ms = ["ACDEFGHIKL"[: 3 + j] + "A" * c for j in range(n)]
            reps.append(ids[0])
            seqs.append(ms[0])
            mids.append(ids)
            mseqs.append(ms)
        table = pa.table(
            {
                "sequence_id": reps,
                "sequence": seqs,
                MEMBER_IDS_COLUMN: mids,
                MEMBER_SEQUENCES_COLUMN: mseqs,
            }
        )
        pq.write_table(table, d / f"shard-{s}.parquet")
    return d


def _rows(ds: ShardedProteinDataset, rank: int = 0, world: int = 1) -> list[dict[str, object]]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset_mod, "_resolve_distributed_context", lambda: (rank, world, 0, 1))
        return list(ds._iter_stream(0))


def test_flag_off_is_unchanged(tmp_path: Path) -> None:
    d = _write(tmp_path)
    rows = _rows(ShardedProteinDataset(d, seed=3))
    assert len(rows) == 6
    assert all(set(r) == {"sequence_id", "sequence"} for r in rows)
    assert all(str(r["sequence_id"]).endswith("m0") for r in rows)


def test_member_rows_come_from_their_cluster(tmp_path: Path) -> None:
    d = _write(tmp_path)
    for row in _rows(ShardedProteinDataset(d, seed=3, sample_members=True)):
        cluster = str(row["sequence_id"]).rsplit("m", 1)[0]
        n = _SIZES[int(cluster[1])][int(cluster[3])]
        j = int(str(row["sequence_id"]).rsplit("m", 1)[1])
        assert j < n
        assert row["sequence"] == "ACDEFGHIKL"[: 3 + j] + "A" * int(cluster[3])


def test_singleton_yields_representative(tmp_path: Path) -> None:
    d = _write(tmp_path)
    rows = _rows(ShardedProteinDataset(d, sample_members=True))
    singles = [r for r in rows if str(r["sequence_id"]).startswith(("s0c0", "s1c2"))]
    assert len(singles) == 2
    assert all(str(r["sequence_id"]).endswith("m0") for r in singles)


def _served(ds: ShardedProteinDataset, world: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for rank in range(world):
        for r in _rows(ds, rank, world):
            sid = str(r["sequence_id"])
            out[sid.rsplit("m", 1)[0]] = sid
    return out


def test_same_member_across_topologies(tmp_path: Path) -> None:
    d = _write(tmp_path)
    ds = ShardedProteinDataset(d, seed=5, sample_members=True)
    one = _served(ds, 1)
    assert len(one) == 6
    assert _served(ds, 2) == one
    assert _served(ds, 3) == one


def test_members_change_across_epochs(tmp_path: Path) -> None:
    d = _write(tmp_path)
    ds = ShardedProteinDataset(d, sample_members=True)
    seen = set()
    for epoch in range(10):
        ds.set_epoch(epoch)
        seen |= {
            str(r["sequence_id"]) for r in _rows(ds) if str(r["sequence_id"]).startswith("s0c2")
        }
    assert len(seen) > 1


def test_member_choice_is_uniform() -> None:
    counts = [0, 0, 0]
    for shard in range(300):
        for row in range(100):
            counts[_member_index(12345, shard, row, 3)] += 1
    assert all(abs(c - 10_000) < 300 for c in counts)


def test_missing_member_columns_raise(tmp_path: Path) -> None:
    pq.write_table(pa.table({"sequence_id": ["a"], "sequence": ["AC"]}), tmp_path / "x.parquet")
    with pytest.raises(ValueError, match=MEMBER_IDS_COLUMN):
        ShardedProteinDataset(tmp_path, sample_members=True)


def test_masking_weights_with_members_raise(tmp_path: Path) -> None:
    d = _write(tmp_path)
    with pytest.raises(ValueError, match="masking"):
        ShardedProteinDataset(d, sample_members=True, load_masking_weights=True)


@pytest.mark.parametrize(
    ("ids", "seqs"), [(["a", "b"], ["AC", "AD", "AE"]), ([], [])], ids=["mismatch", "empty"]
)
def test_mismatched_member_lists_raise(tmp_path: Path, ids: list[str], seqs: list[str]) -> None:
    table = pa.table(
        {
            "sequence_id": ["r0", "r1"],
            "sequence": ["AC", "AC"],
            MEMBER_IDS_COLUMN: pa.array([["r0"], ids], type=pa.list_(pa.string())),
            MEMBER_SEQUENCES_COLUMN: pa.array([["AC"], seqs], type=pa.list_(pa.string())),
        }
    )
    path = tmp_path / "bad.parquet"
    pq.write_table(table, path)
    ds = ShardedProteinDataset(path, shuffle_rows=False, sample_members=True)
    with pytest.raises(ValueError, match=r"bad\.parquet.*row 1"):
        _rows(ds)

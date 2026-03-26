"""Tests for ShardedProteinDataset and InterleavedDataset."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from oplm.data.dataset import InterleavedDataset, ShardedProteinDataset

# Real protein sequences (truncated for test brevity)
SEQUENCES = [
    ("sp|P0A8V2|RNB_ECOLI", "MEKPFIRTLAESHFAQYVAH"),
    ("sp|Q8ZKF6|NUOB_SALTI", "MVTDIRYEPELSPAG"),
    ("sp|P68871|HBB_HUMAN", "MVHLTPEEKSAVTALWGKVNV"),
    ("sp|P69905|HBA_HUMAN", "MVLSPADKTNVKAAWGKVGA"),
    ("sp|P01308|INS_HUMAN", "MALWMRLLPLLALLALWGPD"),
    ("sp|P10636|TAU_HUMAN", "MAEPRQEFEVMEDHAGTYGL"),
    ("sp|P04637|P53_HUMAN", "MEEPQSDPSVEPPLSQETFS"),
    ("sp|P38398|BRCA1_HUMAN", "MDLSALRVEEVQNVINAMQK"),
    ("sp|P01375|TNFA_HUMAN", "MSTESMIRDVELAEEALPKK"),
    ("sp|P21359|NF1_HUMAN", "MAAHRPVEWVQAVVSRFDEQ"),
]


def _write_parquet(path: Path, sequences: list[tuple[str, str]]) -> None:
    """Write a parquet file with sequence_id and sequence columns."""
    table = pa.table(
        {
            "sequence_id": [s[0] for s in sequences],
            "sequence": [s[1] for s in sequences],
        }
    )
    pq.write_table(table, path)


@pytest.fixture()
def single_parquet(tmp_path: Path) -> Path:
    """Create a single parquet file with 10 sequences."""
    path = tmp_path / "dataset.parquet"
    _write_parquet(path, SEQUENCES)
    return path


@pytest.fixture()
def sharded_dir(tmp_path: Path) -> Path:
    """Create a directory with 3 shards of parquet files."""
    shard_dir = tmp_path / "sharded"
    shard_dir.mkdir()
    _write_parquet(shard_dir / "shard_000.parquet", SEQUENCES[:4])
    _write_parquet(shard_dir / "shard_001.parquet", SEQUENCES[4:7])
    _write_parquet(shard_dir / "shard_002.parquet", SEQUENCES[7:])
    return shard_dir


class TestShardedProteinDataset:
    def test_single_file_iteration(self, single_parquet: Path) -> None:
        ds = ShardedProteinDataset(single_parquet, shuffle_shards=False, shuffle_rows=False)
        samples = list(ds)
        assert len(samples) == 10
        assert samples[0]["sequence_id"] == SEQUENCES[0][0]
        assert samples[0]["sequence"] == SEQUENCES[0][1]

    def test_sharded_dir_iteration(self, sharded_dir: Path) -> None:
        ds = ShardedProteinDataset(sharded_dir, shuffle_shards=False, shuffle_rows=False)
        samples = list(ds)
        assert len(samples) == 10

    def test_output_keys(self, single_parquet: Path) -> None:
        ds = ShardedProteinDataset(single_parquet, shuffle_shards=False, shuffle_rows=False)
        sample = next(iter(ds))
        assert set(sample.keys()) == {"sequence_id", "sequence"}
        assert isinstance(sample["sequence_id"], str)
        assert isinstance(sample["sequence"], str)

    def test_len(self, sharded_dir: Path) -> None:
        ds = ShardedProteinDataset(sharded_dir)
        assert len(ds) == 10

    def test_shuffle_changes_order(self, sharded_dir: Path) -> None:
        ds = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=42)
        epoch0 = [s["sequence_id"] for s in ds]
        epoch1 = [s["sequence_id"] for s in ds]
        # Different epochs should give different orderings
        assert epoch0 != epoch1

    def test_deterministic_with_same_seed(self, sharded_dir: Path) -> None:
        ds1 = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=42)
        ds2 = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=42)
        samples1 = [s["sequence_id"] for s in ds1]
        samples2 = [s["sequence_id"] for s in ds2]
        assert samples1 == samples2

    def test_different_seeds_different_order(self, sharded_dir: Path) -> None:
        ds1 = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=1)
        ds2 = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=2)
        samples1 = [s["sequence_id"] for s in ds1]
        samples2 = [s["sequence_id"] for s in ds2]
        assert samples1 != samples2

    def test_set_epoch(self, sharded_dir: Path) -> None:
        ds = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=42)
        ds.set_epoch(5)
        epoch5 = [s["sequence_id"] for s in ds]
        ds.set_epoch(5)
        epoch5_again = [s["sequence_id"] for s in ds]
        assert epoch5 == epoch5_again

    def test_no_shards_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError, match="No parquet shards"):
            ShardedProteinDataset(empty_dir)

    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Expected a parquet file"):
            ShardedProteinDataset(tmp_path / "nonexistent.txt")

    def test_all_sequences_present(self, sharded_dir: Path) -> None:
        ds = ShardedProteinDataset(sharded_dir, shuffle_shards=True, shuffle_rows=True, seed=0)
        ids = {s["sequence_id"] for s in ds}
        expected = {s[0] for s in SEQUENCES}
        assert ids == expected


class TestInterleavedDataset:
    def test_basic_interleaving(self, tmp_path: Path) -> None:
        # Create two small datasets
        path_a = tmp_path / "ds_a.parquet"
        path_b = tmp_path / "ds_b.parquet"
        _write_parquet(path_a, SEQUENCES[:5])
        _write_parquet(path_b, SEQUENCES[5:])

        ds_a = ShardedProteinDataset(path_a, shuffle_shards=False, shuffle_rows=False)
        ds_b = ShardedProteinDataset(path_b, shuffle_shards=False, shuffle_rows=False)

        interleaved = InterleavedDataset([ds_a, ds_b], [0.5, 0.5], seed=42)
        samples = list(interleaved)
        assert len(samples) == 10  # 5 + 5

    def test_fraction_approximate(self, tmp_path: Path) -> None:
        # Create two datasets, sample heavily from one
        path_a = tmp_path / "ds_a.parquet"
        path_b = tmp_path / "ds_b.parquet"
        _write_parquet(path_a, SEQUENCES[:5])
        _write_parquet(path_b, SEQUENCES[5:])

        ds_a = ShardedProteinDataset(path_a, shuffle_shards=False, shuffle_rows=False)
        ds_b = ShardedProteinDataset(path_b, shuffle_shards=False, shuffle_rows=False)

        interleaved = InterleavedDataset(
            [ds_a, ds_b], [0.8, 0.2], num_samples=100, seed=42
        )
        samples = list(interleaved)
        assert len(samples) == 100

        # Check approximate fractions (with tolerance for randomness)
        a_ids = {s[0] for s in SEQUENCES[:5]}
        from_a = sum(1 for s in samples if s["sequence_id"] in a_ids)
        # With 80/20 split over 100 samples, expect ~80 from A
        assert 60 < from_a < 95

    def test_empty_datasets_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            InterleavedDataset([], [])

    def test_mismatched_lengths_raises(self, single_parquet: Path) -> None:
        ds = ShardedProteinDataset(single_parquet, shuffle_shards=False, shuffle_rows=False)
        with pytest.raises(ValueError, match="same length"):
            InterleavedDataset([ds], [0.5, 0.5])

    def test_negative_fraction_raises(self, single_parquet: Path) -> None:
        ds = ShardedProteinDataset(single_parquet, shuffle_shards=False, shuffle_rows=False)
        with pytest.raises(ValueError, match="non-negative"):
            InterleavedDataset([ds], [-0.5])

    def test_zero_fraction_raises(self, single_parquet: Path) -> None:
        ds = ShardedProteinDataset(single_parquet, shuffle_shards=False, shuffle_rows=False)
        with pytest.raises(ValueError, match="positive"):
            InterleavedDataset([ds], [0.0])

    def test_set_epoch_propagates(self, tmp_path: Path) -> None:
        path_a = tmp_path / "ds_a.parquet"
        path_b = tmp_path / "ds_b.parquet"
        _write_parquet(path_a, SEQUENCES[:5])
        _write_parquet(path_b, SEQUENCES[5:])

        ds_a = ShardedProteinDataset(path_a, shuffle_shards=True, shuffle_rows=True, seed=42)
        ds_b = ShardedProteinDataset(path_b, shuffle_shards=True, shuffle_rows=True, seed=42)

        interleaved = InterleavedDataset([ds_a, ds_b], [0.5, 0.5], seed=42)
        interleaved.set_epoch(3)
        # Should not raise, and sub-datasets should have epoch set
        samples = list(interleaved)
        assert len(samples) > 0

    def test_reinitializes_exhausted(self, tmp_path: Path) -> None:
        # One dataset is much smaller — it should get re-initialized
        path_a = tmp_path / "ds_small.parquet"
        path_b = tmp_path / "ds_big.parquet"
        _write_parquet(path_a, SEQUENCES[:2])
        _write_parquet(path_b, SEQUENCES[2:])

        ds_a = ShardedProteinDataset(path_a, shuffle_shards=False, shuffle_rows=False)
        ds_b = ShardedProteinDataset(path_b, shuffle_shards=False, shuffle_rows=False)

        interleaved = InterleavedDataset(
            [ds_a, ds_b], [0.5, 0.5], num_samples=20, seed=42
        )
        samples = list(interleaved)
        assert len(samples) == 20

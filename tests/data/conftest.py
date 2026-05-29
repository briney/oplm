"""Shared fixtures for the ``oplm.data`` test suite.

Provides real sequence-parquet fixtures used by the dataset/collate/loader tests:
a tiny multi-shard parquet of *real* protein sequences (Phase 3), with an
optional ``masking_weights`` column (Phase 4/10). Sequences are sliced from the
existing ``tests/fixtures/training/test_sequences.parquet`` so fixtures stay
real-data-backed.

Real structure/variant/downstream fixtures live under ``tests/data/fixtures/``
(see ``tests/data/fixtures/README.md``); tests ``pytest.skip`` when absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Real protein sequences live here (100k rows, columns sequence_id + sequence).
_SOURCE_PARQUET = (
    Path(__file__).resolve().parents[1] / "fixtures" / "training" / "test_sequences.parquet"
)

# Type of one (sequence_id, sequence) record.
Record = tuple[str, str]


def _load_real_records(n: int) -> list[Record]:
    """Read the first ``n`` real ``(sequence_id, sequence)`` records from disk.

    Reads a single parquet batch so only ``n`` rows are materialized.
    """
    pf = pq.ParquetFile(_SOURCE_PARQUET)
    batch = next(pf.iter_batches(batch_size=n, columns=["sequence_id", "sequence"]))
    ids = batch.column("sequence_id").to_pylist()
    seqs = batch.column("sequence").to_pylist()
    return list(zip(ids, seqs, strict=True))


def _make_weights(sequence: str) -> list[float]:
    """Deterministic, varied per-residue weights of length ``len(sequence)``.

    Weights are a derived feature column (not biological data); a position-based
    pattern keeps them non-uniform so weighting tests are meaningful.
    """
    return [1.0 + float(i % 3) for i in range(len(sequence))]


def _zero_even_weights(sequence: str) -> list[float]:
    """Per-residue weights that are exactly ``0.0`` at every even residue index.

    Even residue indices (0, 2, 4, …) → weight ``0.0``; odd indices → positive
    weight. Aligned to ``input_ids`` (residue ``i`` lands at token position
    ``i + 1``), this makes zero-weight residues the *odd* token positions, so the
    Gumbel-top-k masker — which forces ``key = -inf`` wherever the weight is
    ``0`` — can only ever mask *even* token positions. That gives the end-to-end
    weighted-masking test a hard, draw-independent invariant to assert.
    """
    return [0.0 if i % 2 == 0 else 1.0 + float(i % 3) for i in range(len(sequence))]


def write_sequence_shards(
    directory: Path,
    records: Sequence[Record],
    *,
    n_shards: int = 2,
    id_prefix: str = "",
    with_weights: bool = False,
    weights_fn: Callable[[str], list[float]] | None = None,
) -> Path:
    """Write ``records`` as ``n_shards`` parquet shards under ``directory``.

    Args:
        directory: Existing directory to write shards into.
        records: ``(sequence_id, sequence)`` pairs to distribute across shards.
        n_shards: Number of shard files (records are split round-robin-free, in
            contiguous chunks).
        id_prefix: Prepended to every ``sequence_id`` (used to make interleaved
            sources distinguishable).
        with_weights: Also write a ``masking_weights`` (``list<float64>``) column.
        weights_fn: Per-residue weight generator used when ``with_weights`` is set;
            maps a sequence to a list of weights of the same length. Defaults to
            :func:`_make_weights` (all-positive, non-uniform). Pass
            :func:`_zero_even_weights` for a fixture with hard zero-weight positions.

    Returns:
        ``directory`` (for convenience).
    """
    directory = Path(directory)
    weights_fn = weights_fn or _make_weights
    n_shards = max(1, min(n_shards, len(records)))
    # Contiguous, near-equal chunks so every shard is non-empty.
    chunk = -(-len(records) // n_shards)  # ceil division
    for shard_idx in range(n_shards):
        chunk_records = records[shard_idx * chunk : (shard_idx + 1) * chunk]
        if not chunk_records:
            continue
        ids = [f"{id_prefix}{rid}" for rid, _ in chunk_records]
        seqs = [seq for _, seq in chunk_records]
        columns: dict[str, pa.Array] = {
            "sequence_id": pa.array(ids, type=pa.large_string()),
            "sequence": pa.array(seqs, type=pa.large_string()),
        }
        if with_weights:
            columns["masking_weights"] = pa.array(
                [weights_fn(seq) for seq in seqs], type=pa.list_(pa.float64())
            )
        pq.write_table(pa.table(columns), directory / f"shard_{shard_idx:03d}.parquet")
    return directory


@pytest.fixture(scope="session")
def real_records() -> list[Record]:
    """A pool of 40 real ``(sequence_id, sequence)`` records for building shards."""
    return _load_real_records(40)


@pytest.fixture(scope="session")
def make_sequence_shards() -> Callable[..., Path]:
    """Return the :func:`write_sequence_shards` factory (for ad-hoc fixtures)."""
    return write_sequence_shards


@pytest.fixture(scope="session")
def sequence_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-shard directory of 14 real sequences (no ``masking_weights`` column)."""
    directory = tmp_path_factory.mktemp("sequence_shards")
    return write_sequence_shards(directory, _load_real_records(14), n_shards=2)


@pytest.fixture(scope="session")
def sequence_shards_weighted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-shard directory of 14 real sequences *with* a ``masking_weights`` column."""
    directory = tmp_path_factory.mktemp("sequence_shards_weighted")
    return write_sequence_shards(directory, _load_real_records(14), n_shards=2, with_weights=True)


@pytest.fixture(scope="session")
def sequence_shards_zero_weighted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-shard directory of 14 real sequences with zero-weight even residues.

    Uses :func:`_zero_even_weights`, so every even residue index carries weight
    ``0.0``. The end-to-end weighted-masking test relies on the resulting hard
    invariant: masking can only ever land on *even* token positions.
    """
    directory = tmp_path_factory.mktemp("sequence_shards_zero_weighted")
    return write_sequence_shards(
        directory,
        _load_real_records(14),
        n_shards=2,
        with_weights=True,
        weights_fn=_zero_even_weights,
    )

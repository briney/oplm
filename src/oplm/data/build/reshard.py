"""Re-cut a directory of parquet shards into fixed-size shards, keeping row order."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from oplm.data.sequence.dataset import ShardedProteinDataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

_COMPRESSION = "zstd"


def _update(h: hashlib._Hash, batch: pa.RecordBatch) -> None:
    for sid, seq in zip(
        batch.column("sequence_id").to_pylist(), batch.column("sequence").to_pylist(), strict=True
    ):
        h.update(f"{sid}\t{seq}\n".encode())


def _batches(paths: Iterable[Path]) -> Iterable[pa.RecordBatch]:
    for p in paths:
        yield from pq.ParquetFile(p).iter_batches()


def stream_digest(paths: Sequence[Path]) -> tuple[int, str]:
    """Row count and sha256 over ``(sequence_id, sequence)`` of ``paths`` in order."""
    h, rows = hashlib.sha256(), 0
    for b in _batches(paths):
        _update(h, b)
        rows += b.num_rows
    return rows, h.hexdigest()


def reshard(src: Path, out: Path, *, rows_per_shard: int) -> dict:
    """Re-cut ``src`` shards into ``out`` shards of ``rows_per_shard`` rows.

    Global row order (sorted source filename order) is preserved; every column is kept.
    """
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory {out} is not empty")
    shards = ShardedProteinDataset._discover_shards(src)
    schema = pq.ParquetFile(shards[0]).schema_arrow

    h_in = hashlib.sha256()
    n_out = rows = 0
    writer: pq.ParquetWriter | None = None
    tmp = final = out
    in_shard = 0

    def close() -> None:
        nonlocal writer
        if writer is not None:
            writer.close()
            tmp.rename(final)
            writer = None

    for batch in _batches(shards):
        _update(h_in, batch)
        rows += batch.num_rows
        start = 0
        while start < batch.num_rows:
            if writer is None:
                final = out / f"shard-{n_out:05d}.parquet"
                tmp = out / f"{final.name}.tmp"  # not a parquet suffix: never looks like a shard
                writer = pq.ParquetWriter(tmp, schema, compression=_COMPRESSION)
                n_out += 1
                in_shard = 0
            take = min(rows_per_shard - in_shard, batch.num_rows - start)
            writer.write_batch(batch.slice(start, take))
            in_shard += take
            start += take
            if in_shard == rows_per_shard:
                close()
    close()

    outs = ShardedProteinDataset._discover_shards(out)
    rows_out, sha_out = stream_digest(outs)
    summary = {
        "source_shards": [p.name for p in shards],
        "rows": rows,
        "shards_out": n_out,
        "rows_per_shard": rows_per_shard,
        "compression": _COMPRESSION,
        "sha256_in": h_in.hexdigest(),
        "sha256_out": sha_out,
    }
    if rows_out != rows or sha_out != summary["sha256_in"]:
        raise RuntimeError(f"reshard verification failed: {rows} rows in, {rows_out} out")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

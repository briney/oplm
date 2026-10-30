"""UniRef90 -> 70% identity linclust stage and a streaming parser for its cluster output."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oplm.data.build.mmseqs import run_mmseqs_cmd
from oplm.data.sequence.dataset import MEMBER_IDS_COLUMN, MEMBER_SEQUENCES_COLUMN

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

__all__ = ["Cluster", "cluster_uniref90", "iter_clusters", "iter_rows", "shuffle_clusters"]


class Cluster(NamedTuple):
    """One cluster; the representative's own entry is first in the member lists."""

    rep_id: str
    member_ids: list[str]
    member_sequences: list[str]


def cluster_uniref90(
    fasta: Path, workdir: Path, *, mmseqs: str, threads: int, extra_args: Sequence[str] = ()
) -> Path:
    """Run ``easy-linclust`` at 70% identity / 80% coverage; return ``*_all_seqs.fasta``.

    Raises:
        RuntimeError: ``mmseqs`` is missing or exits non-zero.
    """
    prefix = workdir / "uniref70"
    run_mmseqs_cmd(
        mmseqs,
        [
            "easy-linclust",
            str(fasta),
            str(prefix),
            str(workdir / "tmp"),
            *("--min-seq-id", "0.7", "-c", "0.8", "--cov-mode", "0", "--dbtype", "1"),
            "--threads",
            str(threads),
            *extra_args,
        ],
    )
    return workdir / "uniref70_all_seqs.fasta"


def iter_clusters(all_seqs_fasta: Path) -> Iterator[Cluster]:
    """Stream clusters from an ``easy-linclust`` all-seqs FASTA.

    A header directly followed by another header opens a cluster (it carries only the rep
    id); header+sequence pairs then list the members, rep first. Sequences may be wrapped.
    """
    rep: str | None = None
    ids: list[str] = []
    seqs: list[str] = []
    parts: list[str] = []
    pending_header: str | None = None  # header seen, no sequence line yet

    def flush_member() -> None:
        if ids and len(seqs) < len(ids):
            seqs.append("".join(parts))
        parts.clear()

    with all_seqs_fasta.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if pending_header is not None:
                    # two headers in a row: the earlier one opens a new cluster
                    if rep is not None:
                        yield Cluster(rep, ids, seqs)
                    rep, ids, seqs = pending_header, [], []
                else:
                    flush_member()
                pending_header = line[1:].split(None, 1)[0]
                continue
            if pending_header is not None:
                ids.append(pending_header)
                pending_header = None
            parts.append(line.strip())
    flush_member()
    if rep is not None:
        yield Cluster(rep, ids, seqs)


_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("sequence", pa.string()),
        (MEMBER_IDS_COLUMN, pa.list_(pa.string())),
        (MEMBER_SEQUENCES_COLUMN, pa.list_(pa.string())),
    ]
)
_FLUSH_ROWS = 4096  # per-bucket buffer; bounds pass-1 memory at n_buckets * this many clusters
_DRAW_CHUNK = 1 << 16


def _table(rows: list[Cluster]) -> pa.Table:
    return pa.table(
        {
            "sequence_id": [c.rep_id for c in rows],
            "sequence": [c.member_sequences[0] for c in rows],
            MEMBER_IDS_COLUMN: [c.member_ids for c in rows],
            MEMBER_SEQUENCES_COLUMN: [c.member_sequences for c in rows],
        },
        schema=_SCHEMA,
    )


def shuffle_clusters(
    clusters: Iterable[Cluster], out_dir: Path, *, seed: int, n_buckets: int = 512
) -> list[Path]:
    """Seeded out-of-core global shuffle: scatter clusters to random buckets, permute each.

    Writes ``out_dir/part-{b:05d}.parquet``; reading the returned parts in order is the
    shuffled order. Output depends only on the input order, ``seed`` and ``n_buckets``.
    Pass 2 is resumable per bucket (finished parts are kept; an interrupted bucket restarts
    from its pass-1 file). Pass 1 is skipped if every bucket already has a pass-1 or final file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    finals = [out_dir / f"part-{b:05d}.parquet" for b in range(n_buckets)]
    pass1 = [f.with_suffix(".pass1") for f in finals]

    if not all(f.exists() or p.exists() for f, p in zip(finals, pass1, strict=True)):
        rng = np.random.default_rng(seed)
        tmps = [p.with_suffix(".pass1.tmp") for p in pass1]
        writers = [pq.ParquetWriter(t, _SCHEMA) for t in tmps]
        bufs: list[list[Cluster]] = [[] for _ in range(n_buckets)]

        def flush(b: int) -> None:
            if bufs[b]:
                writers[b].write_table(_table(bufs[b]))
                bufs[b] = []

        draws = iter(())
        for c in clusters:
            b = next(draws, None)
            if b is None:
                draws = iter(rng.integers(0, n_buckets, _DRAW_CHUNK).tolist())
                b = next(draws)
            bufs[b].append(c)
            if len(bufs[b]) >= _FLUSH_ROWS:
                flush(b)
        for b in range(n_buckets):
            flush(b)
            writers[b].close()
            tmps[b].rename(pass1[b])

    for b, (final, p1) in enumerate(zip(finals, pass1, strict=True)):
        if final.exists():
            p1.unlink(missing_ok=True)
            continue
        table = pq.read_table(p1, schema=_SCHEMA)
        perm = np.random.default_rng([seed, b]).permutation(table.num_rows)
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table.take(pa.array(perm, type=pa.int64())), tmp)
        tmp.rename(final)
        p1.unlink()
    return finals


def iter_rows(parts: Sequence[Path]) -> Iterator[dict]:
    """Stream uniref70 rows (dicts) from ``parts`` in order."""
    for part in parts:
        for batch in pq.ParquetFile(part).iter_batches(batch_size=1024):
            yield from batch.to_pylist()

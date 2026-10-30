"""uniref70 build: UniRef90 linclust, cluster parsing, global shuffle, deepclust70 leakage
screen, eval selection and shard writing."""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from oplm.data.build.mmseqs import mmseqs_version, run_mmseqs_cmd
from oplm.data.build.reshard import _COMPRESSION
from oplm.data.sequence.dataset import (
    MEMBER_IDS_COLUMN,
    MEMBER_SEQUENCES_COLUMN,
    ShardedProteinDataset,
)

pc: Any = _pc  # the pyarrow stubs omit the generated compute kernels

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

__all__ = [
    "build_uniref70",
    "SCREEN_ARGS",
    "Cluster",
    "LowSurvivalError",
    "cluster_uniref90",
    "iter_clusters",
    "iter_rows",
    "screen_against",
    "select_eval",
    "shuffle_clusters",
    "write_dataset",
]


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

    Raises:
        ValueError: a cluster's first member is not its representative.
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

    def cluster(rep: str) -> Cluster:
        if not ids or ids[0] != rep:
            raise ValueError(f"cluster {rep}: first member is {ids[:1]}, not the representative")
        return Cluster(rep, ids, seqs)

    with all_seqs_fasta.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if pending_header is not None:
                    # two headers in a row: the earlier one opens a new cluster
                    if rep is not None:
                        yield cluster(rep)
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
        yield cluster(rep)


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
    ``out_dir/params.json`` records ``seed`` / ``n_buckets``; pass 1 records the cluster and
    member counts it read in ``out_dir/counts.json``.

    Raises:
        ValueError: ``n_buckets < 1``, or ``out_dir`` holds parts of a shuffle with other
            (or unrecorded) parameters.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    out_dir.mkdir(parents=True, exist_ok=True)
    params, params_file = {"seed": seed, "n_buckets": n_buckets}, out_dir / "params.json"
    if any(out_dir.glob("part-*.parquet")) or any(out_dir.glob("part-*.pass1")):
        old = json.loads(params_file.read_text()) if params_file.exists() else None
        if old != params:
            raise ValueError(
                f"{out_dir} holds parts of another shuffle ({old} vs {params}); "
                f"delete it to reshuffle"
            )
    else:
        params_file.write_text(json.dumps(params))
    finals = [out_dir / f"part-{b:05d}.parquet" for b in range(n_buckets)]
    pass1 = [f.with_suffix(".pass1") for f in finals]

    if not all(f.exists() or p.exists() for f, p in zip(finals, pass1, strict=True)):
        rng = np.random.default_rng(seed)
        n_clusters = n_members = 0
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
            n_clusters += 1
            n_members += len(c.member_ids)
            if len(bufs[b]) >= _FLUSH_ROWS:
                flush(b)
        tmp = out_dir / "counts.json.tmp"  # before the renames: a crash there redoes pass 1
        tmp.write_text(json.dumps({"clusters": n_clusters, "members": n_members}))
        tmp.rename(out_dir / "counts.json")
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


# Clustering thresholds (symmetric at --cov-mode 0, so query/target roles can swap). -s 4 is
# below the 5.7 default: hits at >=70% identity share many exact k-mers, so the cheaper
# prefilter still finds them. Huge --max-seqs / -e so neither the per-query prefilter cap
# (the targets are the candidates) nor the e-value drops a hit: identity and coverage decide.
SCREEN_ARGS = (
    *("--min-seq-id", "0.7", "-c", "0.8", "--cov-mode", "0"),
    *("-s", "4", "--max-seqs", "100000", "-e", "1e6"),
)


def _ref_shards(reference: Path) -> list[tuple[Path, int, int]]:
    """``(shard, first_row, n_rows)`` in ``_discover_shards`` order (from parquet metadata)."""
    out, start = [], 0
    for p in ShardedProteinDataset._discover_shards(reference):
        n = pq.ParquetFile(p).metadata.num_rows
        out.append((p, start, n))
        start += n
    return out


def _write_chunk_fasta(
    shards: list[tuple[Path, int, int]], lo: int, hi: int, path: Path, *, members: bool = False
) -> None:
    """Stream global reference rows ``[lo, hi)`` to a FASTA, one record batch at a time.

    ``members``: write every member sequence of each row, headed by the row's rep id.
    """
    seq_col = MEMBER_SEQUENCES_COLUMN if members else "sequence"
    with path.open("w") as fh:
        for shard, start, n in shards:
            if start + n <= lo or start >= hi:
                continue
            pos = start
            for b in pq.ParquetFile(shard).iter_batches(
                batch_size=65536, columns=["sequence_id", seq_col]
            ):
                a, z = max(lo - pos, 0), min(hi - pos, b.num_rows)
                if a < z:
                    ids = b.column("sequence_id").slice(a, z - a).to_pylist()
                    seqs = b.column(seq_col).slice(a, z - a).to_pylist()
                    if members:
                        recs = ((i, s) for i, ms in zip(ids, seqs, strict=True) for s in ms)
                    else:
                        recs = zip(ids, seqs, strict=True)
                    fh.write("".join(f">{i}\n{s}\n" for i, s in recs))
                pos += b.num_rows
                if pos >= hi:
                    break


def _rm_prefix(workdir: Path, prefix: str) -> None:
    for p in workdir.glob(f"{prefix}*"):
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def screen_against(
    candidates: Sequence[dict],
    reference: Path,
    workdir: Path,
    *,
    mmseqs: str,
    threads: int,
    chunk_rows: int = 200_000_000,
    members: bool = False,
) -> set[str]:
    """Rep ids of candidates with any member hitting any ``reference`` sequence (``SCREEN_ARGS``).

    The candidate members are the indexed target; the reference (a parquet dir, e.g.
    deepclust70) is streamed as queries in chunks of ``chunk_rows`` rows. Each finished chunk
    leaves ``workdir/hits-{k:05d}.txt``; a rerun skips those chunks. Changing the candidates,
    the reference shards (names, row counts), ``chunk_rows``, ``SCREEN_ARGS`` or ``members``
    discards earlier hits files.

    ``members``: the reference is uniref70 cluster rows (e.g. the shuffle parts, which hold
    the candidates too); every member of each row is a query named by the row's rep id, and a
    candidate's hits from its own cluster are ignored (hits from other candidates count).

    Raises:
        ValueError: the reference has no rows.
        RuntimeError: ``mmseqs`` is missing or exits non-zero.
    """
    shards = _ref_shards(reference)
    total = shards[-1][1] + shards[-1][2]
    if total == 0:
        raise ValueError(f"reference {reference} has no rows")
    workdir.mkdir(parents=True, exist_ok=True)
    _rm_prefix(workdir, "candidates")  # the target DB is rebuilt every call (it is small)
    _rm_prefix(workdir, "chunk-")  # leftovers of a crashed chunk
    rep_of = {m: c["sequence_id"] for c in candidates for m in c[MEMBER_IDS_COLUMN]}
    target_fa = workdir / "candidates.fasta"
    with target_fa.open("w") as fh:
        for c in candidates:
            for i, s in zip(c[MEMBER_IDS_COLUMN], c[MEMBER_SEQUENCES_COLUMN], strict=True):
                fh.write(f">{i}\n{s}\n")
    h = hashlib.sha256()
    with target_fa.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)  # ty: ignore[invalid-argument-type]  # binary mode yields bytes
    h.update(f"{chunk_rows} {SCREEN_ARGS} {[(p.name, n) for p, _, n in shards]}".encode())
    if members:  # appended only here: keeps the deepclust70 key of earlier builds valid
        h.update(b" members")
    key, key_file = h.hexdigest(), workdir / "screen.key"
    if not key_file.exists() or key_file.read_text() != key:
        _rm_prefix(workdir, "hits-")
        key_file.write_text(key)

    target = workdir / "candidates"
    run_mmseqs_cmd(mmseqs, ["createdb", str(target_fa), str(target), "--dbtype", "1"])
    run_mmseqs_cmd(
        mmseqs,
        ["createindex", str(target), str(workdir / "candidates.tmp"), "--threads", str(threads)],
    )

    n_chunks = -(-total // chunk_rows)
    hits_files = [workdir / f"hits-{k:05d}.txt" for k in range(n_chunks)]
    for k, hits in enumerate(hits_files):
        if hits.exists():
            continue
        prefix = f"chunk-{k:05d}"
        fa, qdb, aln = (workdir / f"{prefix}{x}" for x in (".fasta", "", ".aln"))
        _write_chunk_fasta(shards, k * chunk_rows, (k + 1) * chunk_rows, fa, members=members)
        run_mmseqs_cmd(mmseqs, ["createdb", str(fa), str(qdb), "--dbtype", "1"])
        fa.unlink()  # disk: the FASTA and the DB do not need to coexist past createdb
        run_mmseqs_cmd(
            mmseqs,
            [
                "search",
                str(qdb),
                str(target),
                str(aln),
                str(workdir / f"{prefix}.tmp"),
                *SCREEN_ARGS,
                "--threads",
                str(threads),
            ],
        )
        tsv = workdir / f"{prefix}.tsv"
        run_mmseqs_cmd(
            mmseqs,
            [
                *("convertalis", str(qdb), str(target), str(aln), str(tsv)),
                *("--format-output", "query,target"),
            ],
        )
        with tsv.open() as fh:
            pairs = (line.rstrip("\n").split("\t") for line in fh if line.strip())
            ids = sorted({t for q, t in pairs if not members or q != rep_of[t]})
        tmp = hits.with_name(f"{hits.name}.tmp")
        tmp.write_text("".join(f"{i}\n" for i in ids))
        tmp.rename(hits)
        _rm_prefix(workdir, prefix)
    _rm_prefix(workdir, "candidates.tmp")
    return {rep_of[i] for hits in hits_files for i in hits.read_text().split()}


class LowSurvivalError(RuntimeError):
    """Too few eval candidates survive the leakage screen; the user decides how to proceed."""


def select_eval(
    rows: Iterable[dict],
    *,
    pool: int,
    n_eval: int,
    screen: Callable[[Sequence[dict]], set[str]],
    min_survival: float = 0.05,
) -> tuple[list[dict], dict]:
    """First ``n_eval`` of the first ``pool`` rows that ``screen`` does not flag, plus stats.

    Raises:
        LowSurvivalError: ``survivors / pool < min_survival``.
        ValueError: fewer than ``n_eval`` survivors (grow ``pool``).
    """
    candidates = list(itertools.islice(rows, pool))
    hits = screen(candidates)
    survivors = [r for r in candidates if r["sequence_id"] not in hits]
    n = len(candidates)
    stats = {
        "pool": n,
        "hits": n - len(survivors),
        "survivors": len(survivors),
        "survival_rate": len(survivors) / n if n else 0.0,
    }
    if stats["survival_rate"] < min_survival:
        raise LowSurvivalError(f"eval screen survival below {min_survival}: {stats}")
    if len(survivors) < n_eval:
        raise ValueError(f"{len(survivors)} survivors < n_eval={n_eval}; grow pool: {stats}")
    return survivors[:n_eval], stats


class _Tally:
    """Streaming row / residue counts and the exact members-per-cluster distribution."""

    def __init__(self) -> None:
        self.sizes: Counter[int] = Counter()
        self.rep_residues = self.member_residues = 0

    def add(self, b: pa.RecordBatch | pa.Table) -> None:
        sizes, counts = np.unique(
            pc.list_value_length(b[MEMBER_IDS_COLUMN]).to_numpy(), return_counts=True
        )
        self.sizes.update(dict(zip(sizes.tolist(), counts.tolist(), strict=True)))
        self.rep_residues += pc.sum(pc.utf8_length(b["sequence"])).as_py() or 0
        members = pc.list_flatten(b[MEMBER_SEQUENCES_COLUMN])
        self.member_residues += pc.sum(pc.utf8_length(members)).as_py() or 0

    def summary(self) -> dict:
        rows = sum(self.sizes.values())
        member_rows = sum(k * v for k, v in self.sizes.items())
        out: dict[str, Any] = {
            "rows": rows,
            "member_rows": member_rows,
            "rep_residues": self.rep_residues,
            "member_residues": self.member_residues,
        }
        if not rows:
            return out
        ordered = sorted(self.sizes.items())
        cum = list(itertools.accumulate(v for _, v in ordered))

        def nth(i: int) -> int:  # i-th smallest cluster size, 0-based
            return next(k for (k, _), c in zip(ordered, cum, strict=True) if c > i)

        hist: Counter[str] = Counter()
        for k, v in ordered:
            lo = 1 << (k.bit_length() - 1) if k else 0
            hist[str(lo) if lo < 2 else f"{lo}-{2 * lo - 1}"] += v
        out["members_per_cluster"] = {
            "min": ordered[0][0],
            "median": (nth((rows - 1) // 2) + nth(rows // 2)) / 2,
            "mean": member_rows / rows,
            "max": ordered[-1][0],
            "log2_histogram": dict(hist),
        }
        return out


def write_dataset(
    parts: Sequence[Path],
    eval_rows: list[dict],
    out: Path,
    *,
    rows_per_shard: int,
    test_rows: Sequence[dict] = (),
    row_group_size: int = 10_000,
) -> dict:
    """Write ``out/eval.parquet``, ``out/test.parquet`` (if ``test_rows``) and train shards
    ``out/train/shard-{i:05d}.parquet``.

    Train is ``parts`` in order with the eval and test rep ids removed, cut every
    ``rows_per_shard`` rows. Files are written as ``*.tmp`` and renamed when complete.

    Raises:
        FileExistsError: ``out/train`` is not empty.
        ValueError: ``rows_per_shard < 1``, or some eval/test rep id is not in ``parts``.
    """
    if rows_per_shard < 1:
        raise ValueError(f"rows_per_shard must be >= 1, got {rows_per_shard}")
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    if any(train_dir.iterdir()):
        raise FileExistsError(f"output directory {train_dir} is not empty")

    held: dict[str, _Tally] = {}
    held_ids = []
    for name, rows in (("eval", eval_rows), ("test", test_rows)):
        if name == "test" and not rows:
            continue
        table = pa.Table.from_pylist(list(rows), schema=_SCHEMA)
        tmp = out / f"{name}.parquet.tmp"
        pq.write_table(table, tmp, compression=_COMPRESSION)
        tmp.rename(out / f"{name}.parquet")
        held[name] = _Tally()
        held[name].add(table)
        held_ids.append(table["sequence_id"].combine_chunks())
    eval_ids = pa.chunked_array(held_ids, type=pa.string())

    train_tally = _Tally()
    n_shards = in_shard = in_group = skipped = 0
    writer: pq.ParquetWriter | None = None
    group: list[pa.RecordBatch] = []
    tmp = final = out

    def flush_group() -> None:
        nonlocal group, in_group
        if group:
            assert writer is not None
            writer.write_table(pa.Table.from_batches(group, schema=_SCHEMA))
            group, in_group = [], 0

    for part in parts:
        for batch in pq.ParquetFile(part).iter_batches(batch_size=row_group_size):
            keep = pc.invert(pc.is_in(batch["sequence_id"], value_set=eval_ids))
            skipped += batch.num_rows
            batch = batch.filter(keep)
            skipped -= batch.num_rows
            train_tally.add(batch)
            start = 0
            while start < batch.num_rows:
                if writer is None:
                    final = train_dir / f"shard-{n_shards:05d}.parquet"
                    tmp = train_dir / f"{final.name}.tmp"
                    writer = pq.ParquetWriter(tmp, _SCHEMA, compression=_COMPRESSION)
                    n_shards += 1
                    in_shard = 0
                take = min(rows_per_shard - in_shard, row_group_size - in_group)
                take = min(take, batch.num_rows - start)
                group.append(batch.slice(start, take))
                in_group += take
                in_shard += take
                start += take
                if in_group == row_group_size:
                    flush_group()
                if in_shard == rows_per_shard:
                    flush_group()
                    writer.close()
                    tmp.rename(final)
                    writer = None
    if writer is not None:
        flush_group()
        writer.close()
        tmp.rename(final)
    if skipped != len(eval_ids):
        raise ValueError(f"{len(eval_ids) - skipped} eval/test rep ids are not in the parts")

    return {
        **{name: tally.summary() for name, tally in held.items()},
        "train": train_tally.summary(),
        "train_shards": n_shards,
        "rows_per_shard": rows_per_shard,
        "row_group_size": row_group_size,
        "compression": _COMPRESSION,
    }


# ---- staged pipeline (`oplm data uniref70`) ----

_SCREEN_MIN_FREE = 200e9  # bytes: mmseqs index/tmp for the chunked screen


def _done(workdir: Path, stage: str, params: dict | None = None) -> dict | None:
    """Contents of ``workdir/<stage>.done``, or None. Refuses if recorded params differ."""
    p = workdir / f"{stage}.done"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if params is not None:
        old = d.get("params")
        if old != params:
            raise ValueError(
                f"{p} was written with different parameters ({old} vs {params}); "
                f"delete it (and later stages' .done files) to redo the stage"
            )
    return d


def _mark(workdir: Path, stage: str, data: dict) -> None:
    p = workdir / f"{stage}.done"
    tmp = workdir / f"{stage}.done.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(p)


def _need_space(stage: str, dirs: Sequence[Path], need: float) -> None:
    for d in dirs:
        free = shutil.disk_usage(d).free
        if free < need:
            raise RuntimeError(
                f"{stage}: {d} has {free / 1e9:.0f} GB free, needs ~{need / 1e9:.0f} GB"
            )


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            n += block.count(b"\n")
    return n


def _input_sha256(fasta: Path, workdir: Path) -> str:
    """sha256 of the UniRef90 FASTA, computed once and cached in the workdir."""
    from oplm.data.build.manifest import sha256_file

    cache = workdir / "input.sha256"
    if not cache.exists():
        tmp = workdir / "input.sha256.tmp"
        tmp.write_text(sha256_file(fasta))
        tmp.rename(cache)
    return cache.read_text().strip()


def build_uniref70(
    *,
    uniref90: Path,
    release: str,
    deepclust70: Path,
    workdir: Path,
    out: Path,
    seed: int = 0,
    threads: int = 32,
    mmseqs: str = "mmseqs",
    mmseqs_extra: Sequence[str] = (),
    pool: int = 400_000,
    n_eval: int = 10_000,
    n_test: int = 10_000,
    rows_per_shard: int = 100_000,
    n_buckets: int = 512,
    chunk_rows: int = 200_000_000,
    keep_workdir: bool = False,
) -> dict | None:
    """Run cluster -> shuffle -> screen -> write, resuming from ``workdir/<stage>.done``.

    Returns the summary dict, or None when ``deepclust70/summary.json`` (written by
    ``oplm data reshard`` after verification) does not exist yet (the run stops after
    ``shuffle``).

    Raises:
        ValueError: a finished stage was run with different parameters, or ``out`` would
            overwrite ``deepclust70``.
        RuntimeError: not enough free disk space for the next stage, or the parsed member
            count disagrees with ``uniref70_cluster.tsv``.
        FileExistsError: ``out/train`` is not empty and was not written by this build.
    """
    if out.resolve() == deepclust70.resolve().parent or (
        (out / "train").resolve() == deepclust70.resolve()
    ):
        raise ValueError(f"--out {out} would overwrite --deepclust70 {deepclust70}")
    workdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    spaces = (workdir, out)
    summary: dict = {"uniref90_release": release, "stages": {}}

    # cluster
    cl: dict[str, Any] | None = _done(workdir, "cluster")
    if cl is None:
        _need_space("cluster", spaces, 4 * uniref90.stat().st_size)
        t0 = time.monotonic()
        fasta = cluster_uniref90(
            uniref90, workdir, mmseqs=mmseqs, threads=threads, extra_args=mmseqs_extra
        )
        _input_sha256(uniref90, workdir)
        cl = {
            "all_seqs_fasta": str(fasta),
            "mmseqs_version": mmseqs_version(mmseqs),
            "threads": threads,
            "extra_args": list(mmseqs_extra),
            "seconds": int(time.monotonic() - t0),
        }
        _mark(workdir, "cluster", cl)
    summary["stages"]["cluster"] = cl

    # shuffle
    sh_params = {"seed": seed, "n_buckets": n_buckets}
    shuf_dir = workdir / "shuffled"
    sh: dict[str, Any] | None = _done(workdir, "shuffle", sh_params)
    if sh is None:
        _need_space("shuffle", spaces, 1.2 * uniref90.stat().st_size)
        _input_sha256(uniref90, workdir)  # cached so later stages never need the input
        t0 = time.monotonic()
        parts = shuffle_clusters(
            iter_clusters(Path(cl["all_seqs_fasta"])), shuf_dir, seed=seed, n_buckets=n_buckets
        )
        counts = json.loads((shuf_dir / "counts.json").read_text())
        tsv = workdir / "uniref70_cluster.tsv"  # easy-linclust: one line per member
        tsv_lines = _count_lines(tsv) if tsv.exists() else None
        if tsv_lines is not None and tsv_lines != counts["members"]:
            raise RuntimeError(
                f"parsed {counts['members']} members from {cl['all_seqs_fasta']} but {tsv} "
                f"has {tsv_lines} lines"
            )
        sh = {
            "params": sh_params,
            "parts": [p.name for p in parts],
            **counts,
            "cluster_tsv_lines": tsv_lines,
            "seconds": int(time.monotonic() - t0),
        }
        _mark(workdir, "shuffle", sh)
    if not keep_workdir:  # mmseqs outputs only; the shuffled parts and .done markers stay
        for p in workdir.glob("uniref70_*"):
            p.unlink()
        shutil.rmtree(workdir / "tmp", ignore_errors=True)
    summary["stages"]["shuffle"] = sh
    parts = [shuf_dir / n for n in sh["parts"]]

    deep_summary = deepclust70 / "summary.json"
    if not deep_summary.exists():
        return None
    deep = json.loads(deep_summary.read_text())

    # screen
    sc_params = {
        "pool": pool,
        "n_eval": n_eval,
        "n_test": n_test,
        "chunk_rows": chunk_rows,
        "screen_args": list(SCREEN_ARGS),
        "deepclust70": str(deepclust70),
        "screen_uniref70": True,
    }
    sc: dict[str, Any] | None = _done(workdir, "screen", sc_params)
    if sc is None:
        # the uniref70 query DB is ~2.5x the parts' (compressed) size
        _need_space("screen", spaces, _SCREEN_MIN_FREE + 2.5 * sum(p.stat().st_size for p in parts))
        t0 = time.monotonic()
        screen_hits: dict[str, int] = {}

        def screen(cand: Sequence[dict]) -> set[str]:
            def run(ref: Path, wd: str, members: bool) -> set[str]:
                return screen_against(
                    cand,
                    ref,
                    workdir / wd,
                    mmseqs=mmseqs,
                    threads=threads,
                    chunk_rows=chunk_rows,
                    members=members,
                )

            deep_hits = run(deepclust70, "screen", False)
            # linclust under-clusters: candidates can be >=70% identical to other clusters
            u70_hits = run(shuf_dir, "screen-uniref70", True)
            screen_hits.update(hits_deepclust70=len(deep_hits), hits_uniref70=len(u70_hits))
            return deep_hits | u70_hits

        chosen, stats = select_eval(
            iter_rows(parts), pool=pool, n_eval=n_eval + n_test, screen=screen
        )
        sc = {
            "params": sc_params,
            "eval_ids": [r["sequence_id"] for r in chosen[:n_eval]],
            "test_ids": [r["sequence_id"] for r in chosen[n_eval:]],
            "stats": {**stats, **screen_hits},
            "mmseqs_version": mmseqs_version(mmseqs),
            "seconds": int(time.monotonic() - t0),
        }
        _mark(workdir, "screen", sc)
    summary["stages"]["screen"] = sc["stats"]

    # write
    wr_params = {"rows_per_shard": rows_per_shard, "n_eval": n_eval, "n_test": n_test}
    wr: dict[str, Any] | None = _done(workdir, "write", wr_params)
    if wr is None:
        started, owner = workdir / "write.started", str(out.resolve())
        if started.exists() and started.read_text() == owner:
            # our own crashed earlier attempt left partial output: start clean
            shutil.rmtree(out / "train", ignore_errors=True)
            for stale in (
                *out.glob("eval.parquet*"),
                *out.glob("test.parquet*"),
                out / "summary.json",
            ):
                stale.unlink(missing_ok=True)
        elif (out / "train").exists() and any((out / "train").iterdir()):
            raise FileExistsError(f"{out / 'train'} is not empty and was not written by this build")
        _need_space("write", [out], 1.2 * sum(p.stat().st_size for p in parts))
        tmp = workdir / "write.started.tmp"
        tmp.write_text(owner)
        tmp.rename(started)
        by_id = {
            r["sequence_id"]: r
            for r in itertools.islice(iter_rows(parts), pool)
            if r["sequence_id"] in {*sc["eval_ids"], *sc["test_ids"]}
        }
        missing = [i for i in (*sc["eval_ids"], *sc["test_ids"]) if i not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} eval/test rep ids are not in the pool")
        stats = write_dataset(
            parts,
            [by_id[i] for i in sc["eval_ids"]],
            out,
            rows_per_shard=rows_per_shard,
            test_rows=[by_id[i] for i in sc["test_ids"]],
        )
        wr = {"params": wr_params, "stats": stats}
        _mark(workdir, "write", wr)
        started.unlink()
    summary["stages"]["write"] = wr["stats"]

    from oplm import __version__

    summary.update(
        input_sha256=_input_sha256(uniref90, workdir),
        uniref90_sequences_in=sh.get("members"),
        clusters_out=sh.get("clusters"),
        mmseqs_version=cl["mmseqs_version"],
        screen_mmseqs_version=sc.get("mmseqs_version"),
        deepclust70={
            "path": str(deepclust70),
            "rows": deep["rows"],
            "sha256_out": deep["sha256_out"],
        },
        oplm_version=__version__,
        parameters={
            "seed": seed,
            "threads": threads,
            "mmseqs": mmseqs,
            "mmseqs_extra": list(mmseqs_extra),
            "pool": pool,
            "n_eval": n_eval,
            "n_test": n_test,
            "rows_per_shard": rows_per_shard,
            "n_buckets": n_buckets,
            "chunk_rows": chunk_rows,
            "screen_args": list(SCREEN_ARGS),
            "deepclust70": str(deepclust70),
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

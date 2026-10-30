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
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from oplm.data.build.mmseqs import mmseqs_version, run_mmseqs_cmd
from oplm.data.build.reshard import _COMPRESSION
from oplm.data.sequence.dataset import (
    MEMBER_IDS_COLUMN,
    MEMBER_SEQUENCES_COLUMN,
    ShardedProteinDataset,
)

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


def _write_chunk_fasta(shards: list[tuple[Path, int, int]], lo: int, hi: int, path: Path) -> None:
    """Stream global reference rows ``[lo, hi)`` to a FASTA, one record batch at a time."""
    with path.open("w") as fh:
        for shard, start, n in shards:
            if start + n <= lo or start >= hi:
                continue
            pos = start
            for b in pq.ParquetFile(shard).iter_batches(
                batch_size=65536, columns=["sequence_id", "sequence"]
            ):
                a, z = max(lo - pos, 0), min(hi - pos, b.num_rows)
                if a < z:
                    ids = b.column("sequence_id").slice(a, z - a).to_pylist()
                    seqs = b.column("sequence").slice(a, z - a).to_pylist()
                    fh.write("".join(f">{i}\n{s}\n" for i, s in zip(ids, seqs, strict=True)))
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
) -> set[str]:
    """Rep ids of candidates with any member hitting any ``reference`` sequence (``SCREEN_ARGS``).

    The candidate members are the indexed target; the reference (a parquet dir, e.g.
    deepclust70) is streamed as queries in chunks of ``chunk_rows`` rows. Each finished chunk
    leaves ``workdir/hits-{k:05d}.txt``; a rerun skips those chunks. Changing the candidates,
    the reference shards (names, row counts), ``chunk_rows`` or ``SCREEN_ARGS`` discards
    earlier hits files.

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
            h.update(block)
    h.update(f"{chunk_rows} {SCREEN_ARGS} {[(p.name, n) for p, _, n in shards]}".encode())
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
        _write_chunk_fasta(shards, k * chunk_rows, (k + 1) * chunk_rows, fa)
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
            ["convertalis", str(qdb), str(target), str(aln), str(tsv), "--format-output", "target"],
        )
        with tsv.open() as fh:
            ids = sorted({line.strip() for line in fh} - {""})
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
        out = {
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
    row_group_size: int = 10_000,
) -> dict:
    """Write ``out/eval.parquet`` and ``out/train/shard-{i:05d}.parquet``.

    Train is ``parts`` in order with the eval rep ids removed, cut every ``rows_per_shard``
    rows. Files are written as ``*.tmp`` and renamed when complete.

    Raises:
        FileExistsError: ``out/train`` is not empty.
        ValueError: some eval rep id is not in ``parts``.
    """
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    if any(train_dir.iterdir()):
        raise FileExistsError(f"output directory {train_dir} is not empty")

    eval_table = pa.Table.from_pylist(eval_rows, schema=_SCHEMA)
    tmp = out / "eval.parquet.tmp"
    pq.write_table(eval_table, tmp, compression=_COMPRESSION)
    tmp.rename(out / "eval.parquet")
    eval_tally = _Tally()
    eval_tally.add(eval_table)
    eval_ids = eval_table["sequence_id"].combine_chunks()

    train_tally = _Tally()
    n_shards = in_shard = in_group = skipped = 0
    writer: pq.ParquetWriter | None = None
    group: list[pa.RecordBatch] = []
    tmp = final = out

    def flush_group() -> None:
        nonlocal group, in_group
        if group:
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
        raise ValueError(f"{len(eval_ids) - skipped} eval rep ids are not in the parts")

    return {
        "eval": eval_tally.summary(),
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
    pool: int = 200_000,
    n_eval: int = 10_000,
    rows_per_shard: int = 100_000,
    n_buckets: int = 512,
    chunk_rows: int = 500_000_000,
    keep_workdir: bool = False,
) -> dict | None:
    """Run cluster -> shuffle -> screen -> write, resuming from ``workdir/<stage>.done``.

    Returns the summary dict, or None when ``deepclust70`` does not exist yet (the run stops
    after ``shuffle``).

    Raises:
        ValueError: a finished stage was run with different parameters.
        RuntimeError: not enough free disk space for the next stage.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    spaces = (workdir, out)
    in_bytes = uniref90.stat().st_size
    summary: dict = {"uniref90_release": release, "stages": {}}

    # cluster
    cl = _done(workdir, "cluster")
    if cl is None:
        _need_space("cluster", spaces, 2.5 * in_bytes)
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
    sh = _done(workdir, "shuffle", sh_params)
    if sh is None:
        _need_space("shuffle", spaces, 1.2 * in_bytes)
        t0 = time.monotonic()
        parts = shuffle_clusters(
            iter_clusters(Path(cl["all_seqs_fasta"])), shuf_dir, seed=seed, n_buckets=n_buckets
        )
        sh = {
            "params": sh_params,
            "parts": [p.name for p in parts],
            "seconds": int(time.monotonic() - t0),
        }
        _mark(workdir, "shuffle", sh)
    if not keep_workdir:  # mmseqs outputs only; the shuffled parts and .done markers stay
        for p in workdir.glob("uniref70_*"):
            p.unlink()
        shutil.rmtree(workdir / "tmp", ignore_errors=True)
    summary["stages"]["shuffle"] = sh
    parts = [shuf_dir / n for n in sh["parts"]]

    if not deepclust70.exists():
        return None

    # screen
    sc_params = {
        "pool": pool,
        "n_eval": n_eval,
        "chunk_rows": chunk_rows,
        "screen_args": list(SCREEN_ARGS),
        "deepclust70": str(deepclust70),
    }
    sc = _done(workdir, "screen", sc_params)
    if sc is None:
        _need_space("screen", spaces, _SCREEN_MIN_FREE)
        t0 = time.monotonic()
        eval_rows, stats = select_eval(
            iter_rows(parts),
            pool=pool,
            n_eval=n_eval,
            screen=lambda cand: screen_against(
                cand,
                deepclust70,
                workdir / "screen",
                mmseqs=mmseqs,
                threads=threads,
                chunk_rows=chunk_rows,
            ),
        )
        sc = {
            "params": sc_params,
            "eval_ids": [r["sequence_id"] for r in eval_rows],
            "stats": stats,
            "seconds": int(time.monotonic() - t0),
        }
        _mark(workdir, "screen", sc)
    summary["stages"]["screen"] = sc["stats"]

    # write
    wr_params = {"rows_per_shard": rows_per_shard, "n_eval": n_eval}
    wr = _done(workdir, "write", wr_params)
    if wr is None:
        eval_ids = set(sc["eval_ids"])
        eval_rows = [
            r for r in itertools.islice(iter_rows(parts), pool) if r["sequence_id"] in eval_ids
        ]
        stats = write_dataset(parts, eval_rows, out, rows_per_shard=rows_per_shard)
        wr = {"params": wr_params, "stats": stats}
        _mark(workdir, "write", wr)
    summary["stages"]["write"] = wr["stats"]

    from oplm import __version__

    summary.update(
        input_sha256=_input_sha256(uniref90, workdir),
        mmseqs_version=cl["mmseqs_version"],
        oplm_version=__version__,
        parameters={
            "seed": seed,
            "threads": threads,
            "mmseqs": mmseqs,
            "mmseqs_extra": list(mmseqs_extra),
            "pool": pool,
            "n_eval": n_eval,
            "rows_per_shard": rows_per_shard,
            "n_buckets": n_buckets,
            "chunk_rows": chunk_rows,
            "screen_args": list(SCREEN_ARGS),
            "deepclust70": str(deepclust70),
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

"""``oplm data`` — dataset build commands."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 (typer resolves annotations at runtime)
from typing import Annotated

import typer

app = typer.Typer(help="Dataset build tools", no_args_is_help=True)


@app.command()
def reshard(
    src: Annotated[Path, typer.Argument(help="Directory of source parquet shards")],
    out: Annotated[Path, typer.Argument(help="Empty output directory")],
    rows_per_shard: Annotated[int, typer.Option(help="Rows per output shard")] = 1_000_000,
) -> None:
    """Re-cut shards to a fixed row count, keeping every row and the global order."""
    from oplm.data.build.reshard import reshard as run

    s = run(src, out, rows_per_shard=rows_per_shard)
    typer.echo(f"{s['rows']} rows -> {s['shards_out']} shards; sha256 {s['sha256_out']}")


@app.command()
def uniref70(
    uniref90: Annotated[Path, typer.Option(help="UniRef90 FASTA")],
    release: Annotated[str, typer.Option(help="UniRef90 release, e.g. 2026_03")],
    deepclust70: Annotated[Path, typer.Option(help="deepclust70 train/ shard directory")],
    workdir: Annotated[Path, typer.Option(help="Scratch/stage directory")],
    out: Annotated[Path, typer.Option(help="Output dataset directory")],
    seed: Annotated[int, typer.Option()] = 0,
    threads: Annotated[int, typer.Option()] = 32,
    mmseqs: Annotated[str, typer.Option(help="mmseqs executable")] = "mmseqs",
    mmseqs_extra: Annotated[
        list[str] | None,
        typer.Option(help="Extra mmseqs arg, repeatable (--mmseqs-extra=--split-memory-limit)"),
    ] = None,
    pool: Annotated[int, typer.Option(help="Eval candidate pool (rows)")] = 200_000,
    n_eval: Annotated[int, typer.Option()] = 10_000,
    rows_per_shard: Annotated[int, typer.Option()] = 100_000,
    n_buckets: Annotated[int, typer.Option()] = 512,
    chunk_rows: Annotated[
        int, typer.Option(help="deepclust70 rows per screen chunk")
    ] = 500_000_000,
    keep_workdir: Annotated[bool, typer.Option(help="Keep mmseqs outputs after shuffle")] = False,
) -> None:
    """Build the uniref70 dataset: cluster, shuffle, leakage screen, write."""
    from oplm.data.build.uniref70 import build_uniref70

    s = build_uniref70(
        uniref90=uniref90,
        release=release,
        deepclust70=deepclust70,
        workdir=workdir,
        out=out,
        seed=seed,
        threads=threads,
        mmseqs=mmseqs,
        mmseqs_extra=mmseqs_extra or [],
        pool=pool,
        n_eval=n_eval,
        rows_per_shard=rows_per_shard,
        n_buckets=n_buckets,
        chunk_rows=chunk_rows,
        keep_workdir=keep_workdir,
    )
    if s is None:
        typer.echo(f"shuffle done; screen waits for {deepclust70} (rerun once it exists)")
    else:
        typer.echo(f"wrote {out} ({s['stages']['write']['train']['rows']} train rows)")


@app.command()
def manifest(dataset: Annotated[Path, typer.Argument(help="Dataset directory")]) -> None:
    """Write DATASET/manifest.parquet (path, rows, bytes, sha256 per file)."""
    from oplm.data.build.manifest import write_manifest

    typer.echo(str(write_manifest(dataset)))

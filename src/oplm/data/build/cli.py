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

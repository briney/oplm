"""μP learning-rate sweep orchestrator.

Fans out one :mod:`scripts.mup_pilot_run` subprocess per ``(width, lr)`` grid
point, each pinned to a single GPU via ``CUDA_VISIBLE_DEVICES`` with a
GPU-sized concurrency pool. On completion it loads every run's ``metrics.json``
(:func:`~oplm.training.mup.summarize_sweep`), prints the argmin-loss LR per
width and the **transfer verdict** (:func:`~oplm.training.mup.best_lr_per_width`),
and writes a loss-vs-LR plot.

The μP payoff: the loss-vs-LR minimum should land at the **same** ``lr`` for
every proxy width (``MupTransferResult.transferred == True``), so that ``lr`` —
the value you put in ``train.lr`` — can be reused unchanged at larger presets.

Run (one GPU per concurrent point)::

    python -m scripts.mup_sweep --widths 256,512 --lrs 1e-3,3e-3,1e-2,3e-2 \\
        --gpus 4 --steps 400 --data corpus.parquet --out sweeps/run1

``--widths`` are hidden sizes; with ``--scaling width`` depth is fixed
(``--depth``), with ``--scaling preset_ray`` depth co-scales with width (which
validates the preset ray, the combined width+depth path). Every run shares
``seed``, ``batch_size``, ``warmup_steps``, and ``steps`` — only ``lr`` and the
width (plus depth in preset_ray mode) vary. Requires the ``train`` extra.
"""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product

# typer resolves these annotations at runtime (get_type_hints), so Path must be a
# real runtime import, not a TYPE_CHECKING-only one.
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from scripts._mup_common import Optimizer, Scaling, parse_floats, parse_widths

if TYPE_CHECKING:
    import pandas as pd

    from oplm.training.mup import MupTransferResult

app = typer.Typer(name="mup-sweep", help=__doc__, add_completion=False)
console = Console()

# Tail of a failed run's stderr surfaced in the summary (chars).
_STDERR_TAIL = 2000


@dataclass(frozen=True)
class _GridPoint:
    """One sweep cell: a width × lr to train and where its metrics land."""

    width: int
    lr: float
    run_dir: Path


@dataclass(frozen=True)
class _RunResult:
    """Outcome of one pilot subprocess."""

    point: _GridPoint
    returncode: int
    stderr_tail: str


def _pilot_argv(point: _GridPoint, *, steps: int, data: Path, common: list[str]) -> list[str]:
    """Build the ``python -m scripts.mup_pilot_run`` argv for one grid point."""
    return [
        sys.executable,
        "-m",
        "scripts.mup_pilot_run",
        "--width",
        str(point.width),
        "--lr",
        repr(point.lr),
        "--steps",
        str(steps),
        "--data",
        str(data),
        "--out",
        str(point.run_dir),
        *common,
    ]


def _run_point(
    point: _GridPoint,
    *,
    steps: int,
    data: Path,
    common: list[str],
    gpu_pool: queue.Queue[int],
) -> _RunResult:
    """Run one pilot subprocess pinned to a GPU borrowed from ``gpu_pool``."""
    gpu = gpu_pool.get()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        console.print(f"[dim]start[/dim] width={point.width} lr={point.lr:g} (GPU {gpu})")
        proc = subprocess.run(
            _pilot_argv(point, steps=steps, data=data, common=common),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        gpu_pool.put(gpu)

    if proc.returncode == 0:
        console.print(f"[green]done[/green]  width={point.width} lr={point.lr:g}")
    else:
        console.print(
            f"[red]FAIL[/red]  width={point.width} lr={point.lr:g} (rc={proc.returncode})"
        )
    return _RunResult(point, proc.returncode, (proc.stderr or "")[-_STDERR_TAIL:])


def _print_results(result: MupTransferResult, df: pd.DataFrame) -> None:
    """Print the per-(width, lr) losses, the best LR per width, and the verdict."""
    table = Table(title="Sweep losses", border_style="dim")
    table.add_column("width", justify="right")
    table.add_column("lr", justify="right")
    table.add_column("final_train_loss", justify="right")
    sorted_df = df.sort_values(["width", "lr"])
    losses = sorted_df["final_train_loss"].tolist()
    for width, lr, loss in zip(
        sorted_df["width"].tolist(), sorted_df["lr"].tolist(), losses, strict=True
    ):
        shown = "—" if loss is None or math.isnan(loss) else f"{float(loss):.4f}"
        table.add_row(str(width), f"{float(lr):g}", shown)
    console.print(table)

    for width, lr in sorted(result.best_lr.items()):
        console.print(f"  width [bold]{width}[/bold]: best lr = [cyan]{lr:g}[/cyan]")
    if result.transferred:
        console.print("[green]✓ transferred[/green]: every width's best LR agrees — reuse it.")
    else:
        console.print(
            "[yellow]not transferred[/yellow]: widths disagree on the best LR "
            "(too-coarse grid, too-few steps, or μP misconfigured)."
        )


def _plot_loss_vs_lr(df: pd.DataFrame, out_png: Path, result: MupTransferResult) -> None:
    """Write a loss-vs-LR plot, one line per width, argmin starred."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    # `best_lr` is int-keyed and covers every swept width, so iterating it both
    # picks the lines to draw and stars each width's argmin (and keeps types clean).
    for width in sorted(result.best_lr):
        valid = df[df["width"] == width].sort_values("lr").dropna(subset=["final_train_loss"])
        if valid.empty:
            continue
        ax.plot(valid["lr"], valid["final_train_loss"], marker="o", label=f"width {width}")
        best = result.best_lr[width]
        best_row = valid[valid["lr"] == best]
        if not best_row.empty:
            ax.scatter(
                [best],
                [best_row["final_train_loss"].iloc[0]],
                marker="*",
                s=220,
                color="black",
                zorder=5,
            )
    ax.set(xscale="log", xlabel="learning rate", ylabel="final EMA train loss")
    verdict = " — transferred ✓" if result.transferred else ""
    ax.set_title(f"μP LR sweep{verdict}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


@app.command()
def main(
    lrs: Annotated[str, typer.Option("--lrs", help="Comma-separated LR grid.")],
    data: Annotated[Path, typer.Option("--data", help="Training parquet file or shard dir.")],
    out: Annotated[Path, typer.Option("--out", help="Sweep output directory.")],
    widths: Annotated[
        str, typer.Option("--widths", help="Comma-separated hidden sizes (multiples of 64).")
    ] = "256,512",
    gpus: Annotated[int, typer.Option("--gpus", help="Concurrent runs = GPUs (ids 0..N-1).")] = 1,
    steps: Annotated[int, typer.Option("--steps", help="max_steps per run (shared).")] = 200,
    depth: Annotated[
        int, typer.Option("--depth", help="Layers (width mode; preset_ray co-scales).")
    ] = 4,
    scaling: Annotated[
        Scaling, typer.Option("--scaling", help="'width' (fixed depth) or 'preset_ray'.")
    ] = Scaling.width,
    optimizer: Annotated[
        Optimizer, typer.Option("--optimizer", help="Optimizer (μP+muon uses 'original').")
    ] = Optimizer.muon,
    mup: Annotated[bool, typer.Option("--mup/--no-mup", help="Enable μP for every run.")] = True,
    base_width: Annotated[int, typer.Option("--base-width", help="μP base width.")] = 512,
    output_mult: Annotated[float, typer.Option("--output-mult", help="μP readout mult.")] = 1.0,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Per-step micro-batch (per GPU).")
    ] = 32,
    grad_accum: Annotated[
        int,
        typer.Option(
            "--grad-accum",
            help="Gradient-accumulation steps; global batch = batch_size × grad_accum (per point).",
        ),
    ] = 1,
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", help="Scheduler warmup.")] = 0,
    weight_decay: Annotated[float, typer.Option("--weight-decay", help="Weight decay.")] = 0.0,
    max_length: Annotated[int, typer.Option("--max-length", help="max_position_embeddings.")] = 512,
    seed: Annotated[int, typer.Option("--seed", help="Shared random seed.")] = 0,
    num_workers: Annotated[int, typer.Option("--num-workers", help="DataLoader workers.")] = 4,
    mixed_precision: Annotated[
        str, typer.Option("--mixed-precision", help="'bf16'/'fp16'/'no'.")
    ] = "bf16",
) -> None:
    """Run a μP LR × width sweep, then report the best LR per width and the verdict."""
    from oplm.training.mup import best_lr_per_width, summarize_sweep

    width_list = parse_widths(widths)
    lr_list = parse_floats(lrs, name="--lrs")
    if gpus < 1:
        raise typer.BadParameter(f"--gpus must be >= 1; got {gpus}")
    out.mkdir(parents=True, exist_ok=True)

    # Options identical across every grid point (only width + lr vary).
    common = [
        "--depth", str(depth),
        "--scaling", scaling.value,
        "--optimizer", optimizer.value,
        "--mup" if mup else "--no-mup",
        "--base-width", str(base_width),
        "--output-mult", repr(output_mult),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--warmup-steps", str(warmup_steps),
        "--weight-decay", repr(weight_decay),
        "--max-length", str(max_length),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
        "--mixed-precision", mixed_precision,
    ]  # fmt: skip

    grid = [_GridPoint(w, lr, out / f"w{w}_lr{lr:g}") for w, lr in product(width_list, lr_list)]
    console.print(
        f"[bold]μP sweep[/bold] {len(grid)} runs "
        f"({len(width_list)} widths × {len(lr_list)} LRs) on {gpus} GPU(s) → {out}"
    )
    # Global batch is shared across every grid point (one GPU/point), so only width
    # + lr vary — μP transfers LR across width, not batch size (see docs/MUP.md).
    console.print(
        f"[dim]global batch = {batch_size} × {grad_accum} = {batch_size * grad_accum} "
        f"per optimizer step[/dim]"
    )

    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in range(gpus):
        gpu_pool.put(gpu)

    results: list[_RunResult] = []
    with ThreadPoolExecutor(max_workers=gpus) as pool:
        futures = [
            pool.submit(_run_point, point, steps=steps, data=data, common=common, gpu_pool=gpu_pool)
            for point in grid
        ]
        for future in as_completed(futures):
            results.append(future.result())

    failed = [r for r in results if r.returncode != 0]
    succeeded = [
        r for r in results if r.returncode == 0 and (r.point.run_dir / "metrics.json").exists()
    ]
    if failed:
        console.print(f"[red]{len(failed)} run(s) failed.[/red] Example stderr tail:")
        console.print(failed[0].stderr_tail or "(no stderr captured)")
    if not succeeded:
        raise typer.Exit(code=1)

    df = summarize_sweep([r.point.run_dir for r in succeeded])
    result = best_lr_per_width(df)
    _print_results(result, df)
    plot_path = out / "sweep_loss_vs_lr.png"
    _plot_loss_vs_lr(df, plot_path, result)
    console.print(f"[green]wrote[/green] {plot_path}")


if __name__ == "__main__":
    app()

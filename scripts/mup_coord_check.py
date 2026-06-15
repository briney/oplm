"""μP coordinate-check CLI — the correctness gate for the μP implementation.

A thin ``typer`` front end over :func:`oplm.training.mup.coord_check`. For a
sweep of hidden widths it builds an OPLM model per width, runs a few optimizer
steps on one fixed batch, and records every ``nn.Linear``/``nn.Embedding``
output RMS at each step. With μP on, per-module RMS should **not grow** with
width; the ``--no-mup`` control fans out. Run this before trusting any LR sweep.

Run::

    python -m scripts.mup_coord_check --widths 128,256,512,1024 --optimizer muon
    python -m scripts.mup_coord_check --no-mup --widths 128,256,512,1024   # control
    python -m scripts.mup_coord_check --scaling preset_ray --widths 256,512,1024

Outputs, into ``--out`` (a directory): a tidy CSV of ``(width, module, step,
rms)`` and a per-module RMS-vs-width plot (one panel per step, one curve per
module type, log-log).

Pass/fail oracle (the correctness gate — eyeball the plot / the printed growth
table against it; the authoritative assertion lives in the slow coord-check
test):

* The oracle is **one-sided**: with μP, **no module's RMS grows with width**.
  Read it as "does not grow", not "stays perfectly flat" — small,
  non-systematic width-to-width wobble is fine. The ``--no-mup`` control is the
  contrast: it fans out / grows with width.
* The **readout-logits module** (``lm_head.decoder``) is **allowed to shrink at
  init** — its logits are ``Θ(1/√m)`` by μP design. Assess it across widths at
  steps ``t ≥ 1`` only and exclude ``t=0``; this script's growth summary is
  taken at the final step (``t = steps ≥ 1``), so ``t=0`` is already excluded.
* Internal attention pre-softmax logits live inside SDPA and are **not** hooked
  as named-submodule outputs, so they need no separate exclusion.

Requires the ``train`` extra (``pip install -e '.[train]'``) for pandas and
matplotlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripts._mup_common import (
    HEAD_DIM,
    PRESET_ASPECT_RATIO,
    Optimizer,
    Scaling,
    num_layers_for,
    parse_widths,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import torch

    from oplm.model import OplmConfig

app = typer.Typer(name="mup-coord-check", help=__doc__, add_completion=False)
console = Console()

# Heuristic line for the printed growth table: μP should hold RMS roughly flat
# across width (growth ~1); the control fans out well past this.
_GROWTH_FLAG = 2.0

# A few real protein sequences used when no `--data` parquet is given. The
# coord-check measures activation scaling, so biological identity is irrelevant
# — only that they are valid canonical-amino-acid strings of varied length.
_DEFAULT_SEQUENCES = [
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "GSHMKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGG",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWK",
    "NLYIQWLKDGGPSSGRPPPS",
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",
    "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFMKTAYIAKQRQISFVKSHFSRQ",
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT",
]


def _build_cfg_fn(
    *,
    depth: int,
    mup: bool,
    scaling: Scaling,
    base_width: int,
    output_mult: float,
) -> Callable[[int], OplmConfig]:
    """Make a ``width -> OplmConfig`` builder for the requested scaling.

    ``head_dim`` is held at 64 (only the head count grows). ``"width"`` keeps
    depth at ``depth``; ``"preset_ray"`` co-scales depth with width at the preset
    aspect ratio (``--depth`` is then ignored). Shared with the sweep harness via
    :func:`scripts._mup_common.num_layers_for` so the gate and the runs match.
    """
    from oplm.model import OplmConfig

    def build(width: int) -> OplmConfig:
        return OplmConfig(
            hidden_size=width,
            num_hidden_layers=num_layers_for(width, depth, scaling),
            num_attention_heads=width // HEAD_DIM,
            head_dim=HEAD_DIM,
            mup_enable=mup,
            mup_base_width=base_width,
            mup_output_mult=output_mult,
        )

    return build


def _load_sequences(data: Path | None, n_seqs: int) -> list[str]:
    """Load up to ``n_seqs`` sequences from a parquet ``--data`` file, else built-ins."""
    if data is None:
        return _DEFAULT_SEQUENCES[:n_seqs]
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(data)
    batch = next(parquet_file.iter_batches(batch_size=n_seqs, columns=["sequence"]))
    return batch.column("sequence").to_pylist()


def _build_batch(sequences: list[str], *, max_length: int, seed: int) -> dict[str, torch.Tensor]:
    """Collate sequences into one fixed ``{input_ids, attention_mask, labels}`` batch.

    Masking is frozen (``deterministic=True``) so the batch — and thus the loss
    the coord-check optimizes — is identical run to run. ``coord_check`` reuses
    this one batch at every width and step.
    """
    from oplm.data import MLMCollator, get_tokenizer

    collator = MLMCollator(get_tokenizer(), max_length=max_length, deterministic=True, seed=seed)
    return collator(sequences)


def _module_type(name: str) -> str:
    """Collapse per-layer module names (``layers.3.…`` -> ``layers.*.…``) into one curve."""
    return re.sub(r"layers\.\d+", "layers.*", name)


@dataclass(frozen=True)
class _GrowthRow:
    """One module type's RMS at the smallest vs largest width and their ratio."""

    module: str
    rms_lo: float
    rms_hi: float
    growth: float


def _growth_summary(df: pd.DataFrame) -> tuple[list[_GrowthRow], int, int]:
    """Per-module RMS growth from the smallest to the largest width at the final step.

    The final step is ``t = steps ≥ 1``, so the readout's init-time ``Θ(1/√m)``
    shrink (``t=0``) is already excluded, matching the oracle.

    Returns:
        Rows sorted by descending growth, plus the ``(lo, hi)`` widths compared.
    """
    final = df[df["step"] == df["step"].max()].copy()
    final["module_type"] = final["module"].map(_module_type)
    agg = final.groupby(["module_type", "width"], as_index=False)["rms"].mean()
    widths = sorted(agg["width"].unique())
    lo, hi = int(widths[0]), int(widths[-1])
    rows: list[_GrowthRow] = []
    for module_type, group in agg.groupby("module_type"):
        by_width = group.set_index("width")["rms"]
        rms_lo, rms_hi = float(by_width.loc[lo]), float(by_width.loc[hi])
        growth = rms_hi / rms_lo if rms_lo > 0 else float("inf")
        rows.append(_GrowthRow(str(module_type), rms_lo, rms_hi, growth))
    rows.sort(key=lambda r: r.growth, reverse=True)
    return rows, lo, hi


def _plot(df: pd.DataFrame, out_png: Path, *, scaling: Scaling, mup: bool) -> None:
    """Write a per-module RMS-vs-width plot: one panel per step, log-log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df["module_type"] = plot_df["module"].map(_module_type)
    agg = plot_df.groupby(["module_type", "step", "width"], as_index=False)["rms"].mean()
    steps = sorted(agg["step"].unique())
    module_types = sorted(agg["module_type"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {m: cmap(i % 20) for i, m in enumerate(module_types)}

    fig, axes = plt.subplots(
        1, len(steps), figsize=(4.0 * len(steps), 4.0), sharex=True, squeeze=False
    )
    for ax, step in zip(axes[0], steps, strict=True):
        step_df = agg[agg["step"] == step]
        for module_type in module_types:
            line = step_df[step_df["module_type"] == module_type].sort_values("width")
            if not line.empty:
                ax.plot(line["width"], line["rms"], marker="o", color=colors[module_type])
        ax.set(xscale="log", yscale="log", xlabel="width (hidden_size)", title=f"t={step}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("activation RMS")

    handles = [plt.Line2D([], [], color=colors[m], marker="o", label=m) for m in module_types]
    fig.legend(handles=handles, loc="center right", fontsize="small", frameon=False)
    tag = "μP on" if mup else "μP OFF (control)"
    fig.suptitle(f"Coord check — {scaling.value} scaling — {tag}")
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _print_growth_table(rows: list[_GrowthRow], lo: int, hi: int, *, mup: bool) -> None:
    """Render the per-module growth summary as a rich table with the oracle verdict."""
    table = Table(title=f"RMS growth, width {lo} → {hi} (final step)", border_style="dim")
    table.add_column("module", style="bold")
    table.add_column(f"RMS@{lo}", justify="right")
    table.add_column(f"RMS@{hi}", justify="right")
    table.add_column("growth", justify="right")
    for row in rows:
        flagged = row.growth > _GROWTH_FLAG
        table.add_row(
            row.module,
            f"{row.rms_lo:.3g}",
            f"{row.rms_hi:.3g}",
            f"[{'red' if flagged else 'green'}]{row.growth:.2f}×[/]",
            style="red" if flagged else None,
        )
    console.print(table)

    worst = rows[0]  # sorted growth-descending in `_growth_summary`
    if mup and worst.growth > _GROWTH_FLAG:
        console.print(
            f"[red]⚠ μP oracle FAIL[/red]: '{worst.module}' RMS grows "
            f"{worst.growth:.2f}× with width (should not grow)."
        )
    elif mup:
        console.print(
            f"[green]✓ μP oracle holds[/green]: no module's RMS grows with width "
            f"(worst {worst.module} {worst.growth:.2f}×)."
        )
    else:
        console.print(
            f"[yellow]control (μP off)[/yellow]: RMS fans out with width as expected "
            f"(worst {worst.module} {worst.growth:.2f}×)."
        )


@app.command()
def main(
    widths: Annotated[
        str, typer.Option("--widths", help="Comma-separated hidden sizes (multiples of 64).")
    ] = "128,256,512,1024",
    depth: Annotated[
        int, typer.Option("--depth", help="Layers (width mode only; preset_ray co-scales).")
    ] = 4,
    steps: Annotated[int, typer.Option("--steps", help="Optimizer steps per width.")] = 3,
    optimizer: Annotated[
        Optimizer, typer.Option("--optimizer", help="Throwaway sweep optimizer.")
    ] = Optimizer.muon,
    data: Annotated[
        Path | None,
        typer.Option("--data", help="Parquet with a 'sequence' column; built-ins if omitted."),
    ] = None,
    mup: Annotated[
        bool, typer.Option("--mup/--no-mup", help="Enable μP (the gate) or run the control.")
    ] = True,
    scaling: Annotated[
        Scaling, typer.Option("--scaling", help="'width' (μP gate) or 'preset_ray'.")
    ] = Scaling.width,
    out: Annotated[Path, typer.Option("--out", help="Output directory for the CSV + plot.")] = Path(
        "mup_coord_check"
    ),
    base_width: Annotated[
        int, typer.Option("--base-width", help="μP base width (m=1 reference).")
    ] = 512,
    output_mult: Annotated[
        float, typer.Option("--output-mult", help="μP readout output multiplier.")
    ] = 1.0,
    n_seqs: Annotated[int, typer.Option("--n-seqs", help="Sequences in the fixed batch.")] = 8,
    max_length: Annotated[int, typer.Option("--max-length", help="Max tokenized length.")] = 128,
    lr: Annotated[float, typer.Option("--lr", help="Base LR for the sweep optimizer.")] = 1e-2,
    seed: Annotated[int, typer.Option("--seed", help="Seed (init + deterministic masking).")] = 0,
    device: Annotated[
        str | None, typer.Option("--device", help="torch device; auto-detects CUDA if omitted.")
    ] = None,
) -> None:
    """Run a μP coordinate check and write a CSV + per-module RMS-vs-width plot."""
    import torch

    from oplm.training.mup import coord_check

    width_list = parse_widths(widths)
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    console.print(
        f"[bold]Coord check[/bold] · widths={width_list} · scaling={scaling.value} · "
        f"optimizer={optimizer.value} · steps={steps} · "
        f"μP={'on' if mup else 'OFF (control)'} · device={run_device}"
    )
    if scaling is Scaling.preset_ray:
        console.print(
            f"[dim]preset_ray: depth co-scales with width at {PRESET_ASPECT_RATIO}:1 "
            f"(--depth ignored).[/dim]"
        )

    sequences = _load_sequences(data, n_seqs)
    batch = _build_batch(sequences, max_length=max_length, seed=seed)
    build_cfg_fn = _build_cfg_fn(
        depth=depth, mup=mup, scaling=scaling, base_width=base_width, output_mult=output_mult
    )

    with console.status("running coord check…"):
        df = coord_check(
            build_cfg_fn,
            width_list,
            batch,
            steps=steps,
            optimizer=optimizer.value,
            seed=seed,
            lr=lr,
            scaling=scaling.value,
            device=run_device,
        )

    out.mkdir(parents=True, exist_ok=True)
    stem = f"coord_check_{scaling.value}_{optimizer.value}_{'mup' if mup else 'nomup'}"
    csv_path, png_path = out / f"{stem}.csv", out / f"{stem}.png"
    df.to_csv(csv_path, index=False)
    _plot(df, png_path, scaling=scaling, mup=mup)
    console.print(f"[green]wrote[/green] {csv_path}  ({len(df)} rows)")
    console.print(f"[green]wrote[/green] {png_path}")

    if len(width_list) >= 2:
        rows, lo, hi = _growth_summary(df)
        _print_growth_table(rows, lo, hi, mup=mup)
    else:
        console.print("[dim]single width — no cross-width growth summary.[/dim]")

    console.print(
        Panel(
            "Oracle (one-sided): with μP, no module's RMS should [bold]grow[/bold] with width; "
            "the --no-mup control fans out. The readout (lm_head.decoder) may [bold]shrink[/bold] "
            "at init (Θ(1/√m)) — assessed at t≥1 (this table uses the final step). Attention "
            "pre-softmax logits live inside SDPA and are not hooked.",
            title="μP coord-check oracle",
            border_style="dim",
        )
    )


if __name__ == "__main__":
    app()

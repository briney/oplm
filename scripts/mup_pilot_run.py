"""Single μP pilot training run — the unit a sweep launches.

Builds one μP-enabled :class:`~oplm.config.OplmConfig` (model + train + data)
from CLI options and runs ``Trainer(cfg, callbacks=[SweepMetricsCallback(...)])``
in-process, so the callback can capture the loss the orchestrator selects on.
Writes ``<out>/metrics.json`` (final EMA train loss + any eval losses + lr +
width + steps) via :class:`~oplm.training.mup.SweepMetricsCallback`.

The sweep (:mod:`scripts.mup_sweep`) fans one of these out per ``(width, lr)``
grid point as ``python -m scripts.mup_pilot_run`` pinned to a single GPU via
``CUDA_VISIBLE_DEVICES`` — accelerate picks that up automatically. Can also be
run standalone::

    python -m scripts.mup_pilot_run --width 512 --lr 1e-2 --steps 200 \\
        --data corpus.parquet --out runs/w512_lr1e-2

μP geometry mirrors the coordinate check (``head_dim=64`` fixed; ``preset_ray``
co-scales depth with width) via :mod:`scripts._mup_common`, so the LR validated
by the coord-check gate is tuned on the same model. Requires the ``train`` extra.
"""

from __future__ import annotations

# typer resolves these annotations at runtime (get_type_hints), so Path must be a
# real runtime import, not a TYPE_CHECKING-only one.
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from scripts._mup_common import HEAD_DIM, Optimizer, Scaling, num_layers_for

if TYPE_CHECKING:
    from oplm.model import OplmConfig as OplmModelConfig

app = typer.Typer(name="mup-pilot-run", help=__doc__, add_completion=False)
console = Console()


def _build_model_config(
    width: int,
    *,
    depth: int,
    scaling: Scaling,
    mup: bool,
    base_width: int,
    output_mult: float,
    max_length: int,
) -> OplmModelConfig:
    """Build the μP-enabled HF model config for one width (shared sweep geometry)."""
    from oplm.model import OplmConfig as OplmModelConfig

    return OplmModelConfig(
        hidden_size=width,
        num_hidden_layers=num_layers_for(width, depth, scaling),
        num_attention_heads=width // HEAD_DIM,
        head_dim=HEAD_DIM,
        max_position_embeddings=max_length,
        mup_enable=mup,
        mup_base_width=base_width,
        mup_output_mult=output_mult,
    )


@app.command()
def main(
    width: Annotated[int, typer.Option("--width", help="Hidden size (multiple of 64).")],
    lr: Annotated[float, typer.Option("--lr", help="Base learning rate (μP transfers this).")],
    data: Annotated[Path, typer.Option("--data", help="Training parquet file or shard directory.")],
    out: Annotated[
        Path, typer.Option("--out", help="Run directory (metrics.json + any checkpoints).")
    ],
    steps: Annotated[int, typer.Option("--steps", help="max_steps for this run.")] = 100,
    depth: Annotated[
        int, typer.Option("--depth", help="Layers (width mode; preset_ray co-scales).")
    ] = 4,
    scaling: Annotated[
        Scaling, typer.Option("--scaling", help="'width' (fixed depth) or 'preset_ray'.")
    ] = Scaling.width,
    optimizer: Annotated[
        Optimizer, typer.Option("--optimizer", help="Optimizer (μP+muon uses 'original').")
    ] = Optimizer.muon,
    mup: Annotated[bool, typer.Option("--mup/--no-mup", help="Enable μP.")] = True,
    base_width: Annotated[
        int, typer.Option("--base-width", help="μP base width (m=1 reference).")
    ] = 512,
    output_mult: Annotated[
        float, typer.Option("--output-mult", help="μP readout output multiplier.")
    ] = 1.0,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Per-step batch size.")] = 32,
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", help="Scheduler warmup.")] = 0,
    weight_decay: Annotated[float, typer.Option("--weight-decay", help="AdamW/Muon WD.")] = 0.0,
    muon_adjust_lr_fn: Annotated[
        str, typer.Option("--muon-adjust-lr-fn", help="Muon LR adjust ('original' for μP).")
    ] = "original",
    max_length: Annotated[
        int, typer.Option("--max-length", help="max_position_embeddings + token truncation.")
    ] = 512,
    log_every: Annotated[int, typer.Option("--log-every", help="Steps between train logs.")] = 10,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    num_workers: Annotated[int, typer.Option("--num-workers", help="DataLoader workers.")] = 4,
    mixed_precision: Annotated[
        str, typer.Option("--mixed-precision", help="'bf16'/'fp16'/'no'.")
    ] = "bf16",
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Keep a final checkpoint.")
    ] = False,
) -> None:
    """Run one μP pilot training run and write ``<out>/metrics.json``."""
    from oplm.config import DataConfig, OplmConfig, TrainConfig
    from oplm.train import _bootstrap_training_environment
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    if width % HEAD_DIM != 0:
        raise typer.BadParameter(f"--width must be a multiple of head_dim={HEAD_DIM}; got {width}")

    _bootstrap_training_environment()
    out.mkdir(parents=True, exist_ok=True)

    cfg = OplmConfig(
        model=_build_model_config(
            width,
            depth=depth,
            scaling=scaling,
            mup=mup,
            base_width=base_width,
            output_mult=output_mult,
            max_length=max_length,
        ),
        train=TrainConfig(
            optimizer=optimizer.value,
            lr=lr,
            weight_decay=weight_decay,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            max_steps=steps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            seed=seed,
            log_every=max(1, min(log_every, steps)),
            wandb_enabled=False,
            output_dir=str(out),
            mixed_precision=mixed_precision,
            save_final=save,
            save_every=steps + 1,  # no mid-run checkpoints during a sweep
        ),
        data=DataConfig(
            train=str(data),
            num_workers=num_workers,
            pin_memory=num_workers > 0,
        ),
    )

    layers = cfg.model.num_hidden_layers
    console.print(
        f"[bold]pilot[/bold] width={width} ({layers}L) lr={lr:g} optimizer={optimizer.value} "
        f"steps={steps} μP={'on' if mup else 'OFF'}"
    )
    metrics_path = out / "metrics.json"
    Trainer(cfg, callbacks=[SweepMetricsCallback(metrics_path)]).train()
    console.print(f"[green]wrote[/green] {metrics_path}")


if __name__ == "__main__":
    app()

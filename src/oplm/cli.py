"""CLI for OPLM: train, encode, info subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from oplm.config import AVAILABLE_PRESETS, load_config
from oplm.inference import load_model_for_inference, resolve_inference_config

app = typer.Typer(name="oplm", help="Open Protein Language Model")
console = Console()

_PRESET_HELP = f"Model size preset ({', '.join(AVAILABLE_PRESETS)})"

# -- Shared type aliases for CLI parameters -------------------------------------------

ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Path to YAML config")]
PresetOpt = Annotated[str | None, typer.Option("--preset", "-p", help=_PRESET_HELP)]
NameOpt = Annotated[
    str | None,
    typer.Option(
        "--name",
        "-n",
        help="W&B run name. Ignored if train.wandb_run_name is set in the YAML or an override.",
    ),
]
OverridesOpt = Annotated[
    list[str] | None,
    typer.Option("--override", help="Config override (key=value). Repeat as needed."),
]
OverridesArg = Annotated[
    list[str] | None,
    typer.Argument(
        metavar="[KEY=VALUE]...",
        help="Config overrides, e.g. train.max_steps=500000 model.nope_dim=512",
    ),
]


def _validate_overrides(overrides: list[str] | None) -> None:
    """Reject positional overrides that aren't ``KEY=VALUE`` before config parsing."""
    for tok in overrides or []:
        if "=" not in tok:
            raise typer.BadParameter(
                f"override must be KEY=VALUE (e.g. train.max_steps=500000), got: {tok!r}"
            )


def _build_argv(
    config: str | None,
    preset: str | None,
    overrides: list[str] | None,
    name: str | None = None,
) -> list[str]:
    """Build an argv list for load_config from CLI options."""
    argv: list[str] = []
    if preset:
        argv.extend(["--preset", preset])
    if config:
        argv.extend(["--config", config])
    if name:
        argv.extend(["--name", name])
    if overrides:
        argv.extend(overrides)
    return argv


@app.command()
def train(
    overrides: OverridesArg = None,
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    name: NameOpt = None,
) -> None:
    """Launch training.

    For distributed training: torchrun --nproc_per_node=N -m oplm.train --config <path>
    """
    _validate_overrides(overrides)
    cfg = load_config(_build_argv(config, preset, overrides, name))
    console.print(f"[bold]Model:[/bold] {cfg.model.num_hidden_layers}L / {cfg.model.hidden_size}D")
    console.print(f"[bold]Output:[/bold] {cfg.train.output_dir}")

    from oplm.train import main as train_main

    train_main(cfg)


@app.command()
def encode(
    sequences: Annotated[list[str], typer.Argument(help="Protein sequences to encode")],
    model_path: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to model weights file or checkpoint directory"),
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output file path")
    ] = "embeddings.pt",
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    overrides: OverridesOpt = None,
) -> None:
    """Encode protein sequences to per-residue embeddings."""
    import torch

    from oplm.data import get_tokenizer
    from oplm.model import LogitsConfig

    cfg = resolve_inference_config(
        model_path,
        config_path=config,
        preset=preset,
        overrides=overrides,
    )
    model = load_model_for_inference(model_path, cfg)
    if getattr(model, "tokenizer", None) is None:
        # `tokenizer` is intentionally un-annotated on the model (see modeling_oplm.py).
        model.tokenizer = get_tokenizer()

    with torch.no_grad():
        embeddings = model.logits(list(sequences), LogitsConfig(return_embeddings=True)).embeddings
    assert embeddings is not None  # return_embeddings=True always populates this

    out_path = Path(output)
    torch.save(embeddings, out_path)
    console.print(f"[green]Saved embeddings[/green] {tuple(embeddings.shape)} → {out_path}")


@app.command()
def info(
    overrides: OverridesArg = None,
    config: ConfigOpt = None,
    preset: PresetOpt = None,
) -> None:
    """Print model config and parameter count."""
    import torch

    from oplm.model import OplmForMaskedLM

    _validate_overrides(overrides)
    cfg = load_config(_build_argv(config, preset, overrides))

    # Build model on meta device to avoid memory allocation
    with torch.device("meta"):
        model = OplmForMaskedLM(cfg.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Format parameter counts
    def _fmt(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    console.print()
    console.rule("[bold]OPLM Model Info[/bold]")
    console.print()

    # Architecture table
    table = Table(title="Architecture", show_header=False, border_style="dim")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Parameters", f"{_fmt(total_params)} ({total_params:,})")
    table.add_row("Trainable", f"{_fmt(trainable_params)} ({trainable_params:,})")
    table.add_row("Hidden size", str(cfg.model.hidden_size))
    table.add_row("Layers", str(cfg.model.num_hidden_layers))
    table.add_row("Attention heads", str(cfg.model.num_attention_heads))
    table.add_row("Head dim", str(cfg.model.head_dim))
    table.add_row("Intermediate size", str(cfg.model.intermediate_size))
    table.add_row("FFN activation", cfg.model.ffn_activation)
    table.add_row("Vocab size", str(cfg.model.vocab_size))
    table.add_row("Max positions", str(cfg.model.max_position_embeddings))
    console.print(table)

    # Features table
    features = Table(title="Features", show_header=False, border_style="dim")
    features.add_column("Feature", style="bold")
    features.add_column("Status")

    def _status(enabled: bool) -> str:
        return "[green]on[/green]" if enabled else "[dim]off[/dim]"

    features.add_row("Norm type", cfg.model.norm_type)
    features.add_row("Norm strategy", cfg.model.norm_strategy)
    features.add_row("Q/K norm", _status(cfg.model.qk_norm))
    features.add_row("Post-embed norm", _status(cfg.model.post_embed_norm))
    features.add_row("Residual scaling", cfg.model.residual_scaling)
    features.add_row("MLM head act", cfg.model.mlm_head_activation)
    features.add_row("Canon", _status(cfg.model.canon_enabled))
    canon_str = (
        ", ".join(cfg.model.canon_positions) if cfg.model.canon_positions else "[dim]none[/dim]"
    )
    features.add_row("Canon positions", canon_str)
    features.add_row("Gradient ckpt", _status(cfg.model.gradient_checkpointing))
    features.add_row("Tied embeddings", _status(cfg.model.tie_word_embeddings))
    console.print(features)
    console.print()

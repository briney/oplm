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
OverridesOpt = Annotated[
    list[str] | None,
    typer.Option("--override", help="Config override (key=value). Repeat as needed."),
]


def _build_argv(
    config: str | None,
    preset: str | None,
    overrides: list[str] | None,
) -> list[str]:
    """Build an argv list for load_config from CLI options."""
    argv: list[str] = []
    if preset:
        argv.extend(["--preset", preset])
    if config:
        argv.extend(["--config", config])
    if overrides:
        argv.extend(overrides)
    return argv


@app.command()
def train(
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    overrides: OverridesOpt = None,
) -> None:
    """Launch training.

    For distributed training: accelerate launch -m oplm.train --config <path>
    """
    cfg = load_config(_build_argv(config, preset, overrides))
    console.print(f"[bold]Model:[/bold] {cfg.model.num_layers}L / {cfg.model.hidden_dim}D")
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
    """Encode protein sequences to embeddings."""
    import torch

    from oplm.data.tokenizer import ProteinTokenizer

    cfg = resolve_inference_config(
        model_path,
        config_path=config,
        preset=preset,
        overrides=overrides,
    )
    model = load_model_for_inference(model_path, cfg)

    # Tokenize and encode
    tokenizer = ProteinTokenizer()
    batch = tokenizer.batch_encode(sequences, max_length=cfg.model.max_seq_len)

    with torch.no_grad():
        hidden = model.encoder(batch["input_ids"], attention_mask=batch["attention_mask"])[0]

    out_path = Path(output)
    torch.save(hidden, out_path)
    console.print(f"[green]Saved embeddings[/green] {tuple(hidden.shape)} → {out_path}")


@app.command()
def info(
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    overrides: OverridesOpt = None,
) -> None:
    """Print model config and parameter count."""
    import torch

    from oplm.model import OplmForMLM

    cfg = load_config(_build_argv(config, preset, overrides))

    # Build model on meta device to avoid memory allocation
    with torch.device("meta"):
        model = OplmForMLM(cfg.model)

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
    table.add_row("Hidden dim", str(cfg.model.hidden_dim))
    table.add_row("Layers", str(cfg.model.num_layers))
    table.add_row("Attention heads", f"{cfg.model.num_heads} (KV: {cfg.model.num_kv_heads})")
    table.add_row("Head dim", str(cfg.model.head_dim))
    table.add_row("FFN dim", str(cfg.model.ffn_dim))
    table.add_row("FFN activation", cfg.model.ffn_activation)
    table.add_row("Vocab size", str(cfg.model.vocab_size))
    table.add_row("Max seq len", str(cfg.model.max_seq_len))
    console.print(table)

    # Features table
    features = Table(title="Features", show_header=False, border_style="dim")
    features.add_column("Feature", style="bold")
    features.add_column("Status")

    def _status(enabled: bool) -> str:
        return "[green]on[/green]" if enabled else "[dim]off[/dim]"

    features.add_row("Shared K/V", _status(cfg.model.shared_kv))
    features.add_row("Q/K norm", _status(cfg.model.qk_norm))
    features.add_row("Output gate", _status(cfg.model.output_gate))
    features.add_row("Post-SDPA norm", _status(cfg.model.post_sdpa_norm))
    features.add_row("Partial RoPE", _status(cfg.model.partial_rope))
    features.add_row("Value residual", _status(cfg.model.value_residual))
    ve_str = str(cfg.model.num_value_embeds) if cfg.model.num_value_embeds else "[dim]off[/dim]"
    features.add_row("Value embeds", ve_str)
    cv_str = cfg.model.conv_positions if cfg.model.conv_positions else "[dim]none[/dim]"
    features.add_row("Conv positions", cv_str)
    if cfg.model.conv_kernel_schedule == "static":
        conv_kernel_str = str(cfg.model.conv_kernel_size)
    else:
        conv_kernel_str = (
            f"{cfg.model.conv_kernel_size} +{cfg.model.conv_kernel_increment} "
            f"every {cfg.model.conv_kernel_block_size} layer(s)"
        )
        if cfg.model.conv_kernel_max_size is not None:
            conv_kernel_str += f", max {cfg.model.conv_kernel_max_size}"
    features.add_row("Conv kernels", conv_kernel_str)
    features.add_row("Attn residual", _status(cfg.model.attn_residual))
    features.add_row("Gradient ckpt", _status(cfg.model.gradient_checkpointing))
    features.add_row("Tied embeddings", _status(cfg.model.tie_embeddings))
    console.print(features)
    console.print()

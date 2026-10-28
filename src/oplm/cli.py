"""CLI for OPLM: train, encode, info subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from oplm.config import AVAILABLE_PRESETS, load_config
from oplm.fold.cli import app as fold_app
from oplm.inference import load_model_for_inference, resolve_inference_config
from oplm.slurm.cli import app as slurm_app
from oplm.sweep.cli import app as sweep_app

app = typer.Typer(name="oplm", help="Open Protein Language Model")
app.add_typer(sweep_app, name="sweep", help="μP learning-rate sweep phases")
app.add_typer(slurm_app, name="slurm", help="Generate and submit Slurm jobs")
app.add_typer(fold_app, name="fold", help="Structure prediction head")
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
        help=(
            "Run name. Sets the W&B run name and the output directory (./<name>). "
            "Each is ignored if train.wandb_run_name / train.output_dir is set in "
            "the YAML or an override."
        ),
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

    For distributed training: accelerate launch -m oplm.train --config <path>
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


def _safetensors_tensors(files: list[Path]):  # noqa: ANN202 — generator of (name, tensor)
    from safetensors import safe_open

    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 — safe_open handles are not mappings
                yield key, handle.get_tensor(key)


@app.command(name="weight-rms")
def weight_rms(
    checkpoints: Annotated[
        list[Path], typer.Argument(help="Checkpoint directories (or their hf/ exports)")
    ],
) -> None:
    """Per-group weight RMS of each checkpoint's hf/ export (offline diag/weight_rms/*)."""
    from oplm.training.diagnostics import GROUPS, weight_rms_by_group

    columns: dict[str, dict[str, float]] = {}
    for checkpoint in checkpoints:
        hf_dir = checkpoint / "hf" if (checkpoint / "hf").is_dir() else checkpoint
        files = sorted(hf_dir.glob("*.safetensors"))
        if not files:
            raise typer.BadParameter(f"no *.safetensors under {hf_dir}")
        columns[checkpoint.name] = weight_rms_by_group(_safetensors_tensors(files))

    table = Table(title="weight RMS by parameter group")
    table.add_column("group")
    for label in columns:
        table.add_column(label, justify="right")
    for group in GROUPS:
        if any(group in column for column in columns.values()):
            table.add_row(
                group, *(f"{c[group]:.4g}" if group in c else "-" for c in columns.values())
            )
    console.print(table)


@app.command(name="stream-replay")
def stream_replay(
    checkpoint: Annotated[
        Path, typer.Argument(help="Checkpoint directory (holds config.yaml and hf/)")
    ],
    world_size: Annotated[
        int, typer.Option("--world-size", help="The run's world size (global batch / batch_size)")
    ],
    steps: Annotated[
        list[int], typer.Option("--step", help="Global step to replay from (repeatable)")
    ],
    batches: Annotated[int, typer.Option(help="Batches per step window")] = 50,
    rank: Annotated[int, typer.Option(help="Rank whose stream to rebuild")] = 0,
    score: Annotated[
        bool, typer.Option(help="Score the checkpoint's frozen model (else data stats only)")
    ] = True,
    device: Annotated[str, typer.Option(help="cuda, cpu or auto")] = "auto",
    compiled: Annotated[
        bool, typer.Option(help="Also score each batch through the trainer's compiled step")
    ] = False,
    ddp: Annotated[
        bool, typer.Option(help="With --compiled: compile a single-process DDP wrapper")
    ] = False,
    partial_after: Annotated[
        int | None,
        typer.Option(help="With --compiled: after this many windows, run the epoch-end batches"),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Write the table as CSV")] = None,
    sets: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="Override the checkpoint's config, KEY=VALUE (repeatable). Runs trained "
            "before data.reshuffle_on_refill / data.drop_last existed need both =false.",
        ),
    ] = None,
) -> None:
    """Rebuild the exact batches a run trained on at given steps; score the frozen model.

    Per window: rows, mean sequence length, source mix, and (with --score) masked-token
    loss overall and per data.train source. Comparing windows before and after a loss jump
    separates harder data from a worse model. --compiled adds the loss of the same batches
    through ``torch.compile`` in train mode with a backward pass (the trainer's step, weights
    untouched) and its difference from eager; --partial-after feeds the short batches that end
    an epoch through that compiled step mid-run.

    Background, the late-loss investigation:
    https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/LATE_LOSS_INVESTIGATION.md
    """
    import csv

    import torch

    from oplm.data.config import parse_train_configs
    from oplm.training.stream_replay import (
        StreamReplay,
        compile_like_trainer,
        compiled_step_loss,
        score_batch,
        trim_batch,
    )

    if not (checkpoint / "config.yaml").is_file():
        raise typer.BadParameter(f"{checkpoint} has no config.yaml")
    if (compiled or ddp or partial_after is not None) and not (score and compiled):
        raise typer.BadParameter(
            "--ddp and --partial-after need --compiled, and --compiled --score"
        )
    cfg = load_config(["--config", str(checkpoint / "config.yaml"), *(sets or [])])
    names = [e.name for e in parse_train_configs(cfg.data.train)] or ["train"]
    model = None
    if score:
        from oplm.model import OplmForMaskedLM

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OplmForMaskedLM.from_pretrained(checkpoint / "hf").to(device).eval()

    step_module = compile_like_trainer(model, cfg, ddp=ddp) if compiled and model else None

    torch.manual_seed(0)  # the training collator masks with each worker's global RNG
    replay = StreamReplay(cfg, rank=rank, world_size=world_size, spawn_workers=ddp)
    console.print(f"{replay.steps_per_epoch} steps per epoch for rank {rank} of {world_size}")
    rows_out: list[dict[str, object]] = []
    for window, step in enumerate(steps, 1):
        epoch, in_epoch = replay.position(step)
        totals = torch.zeros(len(names), 4, dtype=torch.float64)
        compiled_sum = compiled_masked = 0.0
        batch: dict[str, torch.Tensor] = {}
        for batch in replay.batches(step, batches):
            totals += score_batch(batch, len(names), model)
            if step_module is not None and model is not None:
                c_sum, c_masked = compiled_step_loss(step_module, model, batch)
                compiled_sum, compiled_masked = compiled_sum + c_sum, compiled_masked + c_masked
        if step_module is not None and model is not None and window == partial_after and batch:
            for size in replay.partial_sizes:  # the epoch's closing steps, then back to full
                short = trim_batch(batch, size)
                c_sum, c_masked = compiled_step_loss(step_module, model, short)
                e_loss = score_batch(short, len(names), model)[:, 0].sum() / max(c_masked, 1)
                console.print(
                    f"epoch-end batch of {size} rows: compiled {c_sum / max(c_masked, 1):.4f}, "
                    f"eager {float(e_loss):.4f}"
                )
        loss_sum, masked, n_rows, tokens = totals.sum(0).tolist()
        row: dict[str, object] = {
            "step": step,
            "epoch": epoch,
            "batches_in_epoch": in_epoch,
            "rows": int(n_rows),
            "mean_seq_len": tokens / max(n_rows, 1),
            "loss": loss_sum / masked if model is not None and masked else float("nan"),
        }
        for name, (s_loss, s_masked, s_rows, s_tokens) in zip(names, totals.tolist(), strict=True):
            row[f"frac_{name}"] = s_rows / max(n_rows, 1)
            row[f"seq_len_{name}"] = s_tokens / s_rows if s_rows else float("nan")
            scored = model is not None and s_masked > 0
            row[f"loss_{name}"] = s_loss / s_masked if scored else float("nan")
        if step_module is not None:
            loss_compiled = compiled_sum / compiled_masked if compiled_masked else float("nan")
            row["loss_compiled"] = loss_compiled
            row["compiled_minus_eager"] = loss_compiled - loss_sum / max(masked, 1)
        rows_out.append(row)

    table = Table(title=f"stream replay: {checkpoint.name}, rank {rank} of {world_size}")
    for key in rows_out[0]:
        table.add_column(key, justify="right")
    for row in rows_out:
        table.add_row(*(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()))
    console.print(table)
    if out is not None:
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        console.print(f"wrote {out}")
    if ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


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
    features.add_row("Canon residual", _status(cfg.model.canon_residual))
    canon_str = (
        ", ".join(cfg.model.canon_positions) if cfg.model.canon_positions else "[dim]none[/dim]"
    )
    features.add_row("Canon positions", canon_str)
    features.add_row("Gradient ckpt", _status(cfg.model.gradient_checkpointing))
    features.add_row("Tied embeddings", _status(cfg.model.tie_word_embeddings))
    console.print(features)
    console.print()

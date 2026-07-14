"""Generate deterministic μP sweep phase artifacts from a production YAML."""

from __future__ import annotations

import shlex
from pathlib import Path  # noqa: TC003  # Typer resolves annotations at runtime.
from typing import Annotated

import typer

from oplm.config import get_preset_config, load_config, serialize_config
from scripts._mup_common import (
    Params,
    PhaseManifest,
    RunSpec,
    accelerate_argv,
    gradient_accumulation_steps,
    load_phase,
    parse_candidates,
    parse_floats,
    relative_path,
    result_metric,
    write_phase,
)

app = typer.Typer(name="mup-sweep", help=__doc__, add_completion=False)

SMOKE_LRS = (0.0025, 0.01, 0.04)
COARSE_LRS = (0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04)
OUTPUT_MULTS = (0.5, 1.0, 2.0)


def _cell(
    *,
    preset: str,
    lr: float,
    output_mult: float,
    depth_exponent: float,
    seed: int,
    global_examples: int,
    max_steps: int,
    warmup_steps: int,
    batch_mult: float | None = None,
) -> Params:
    params: Params = {
        "preset": preset,
        "lr": lr,
        "output_mult": output_mult,
        "depth_exponent": depth_exponent,
        "seed": seed,
        "global_examples": global_examples,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    }
    if batch_mult is not None:
        params["batch_mult"] = batch_mult
    return params


def _write_run_config(
    base_config: Path,
    run_dir: Path,
    params: Params,
    *,
    num_processes: int,
) -> Path:
    max_steps = int(params["max_steps"])
    warmup_steps = int(params["warmup_steps"])
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError(
            "warmup_steps must satisfy 0 <= warmup_steps < max_steps, "
            f"got {warmup_steps} and {max_steps}"
        )

    preset = str(params["preset"])
    preset_model = get_preset_config(preset).model
    base = load_config(["--config", str(base_config)])
    if base.train.optimizer != "muon":
        raise ValueError(f"μP sweep requires train.optimizer=muon, got {base.train.optimizer}")
    if base.train.muon_adjust_lr_fn != "original":
        raise ValueError(
            "μP sweep requires train.muon_adjust_lr_fn=original, "
            f"got {base.train.muon_adjust_lr_fn}"
        )
    if base.train.weight_decay != 0.01:
        raise ValueError(
            f"μP sweep requires train.weight_decay=0.01, got {base.train.weight_decay}"
        )
    grad_accum = gradient_accumulation_steps(
        int(params["global_examples"]),
        per_device_batch=base.train.batch_size,
        world_size=num_processes,
    )
    run_name = run_dir.name
    overrides = [
        f"model.hidden_size={preset_model.hidden_size}",
        f"model.num_hidden_layers={preset_model.num_hidden_layers}",
        f"model.num_attention_heads={preset_model.num_attention_heads}",
        "model.head_dim=64",
        "model.mup_enable=true",
        "model.mup_base_width=768",
        f"model.mup_output_mult={params['output_mult']}",
        f"train.lr={params['lr']}",
        f"train.mup_depth_lr_exponent={params['depth_exponent']}",
        "train.mup_depth_reference_layers=24",
        f"train.seed={params['seed']}",
        f"train.gradient_accumulation_steps={grad_accum}",
        f"train.max_steps={params['max_steps']}",
        "train.max_epochs=null",
        f"train.warmup_steps={params['warmup_steps']}",
        f"train.stable_steps={max_steps - warmup_steps}",
        "train.scheduler=wsd_linear",
        f"train.output_dir={run_dir / 'output'}",
        f"train.wandb_run_name={run_name}",
    ]
    cfg = load_config(["--config", str(base_config), *overrides])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.yaml"
    path.write_text(serialize_config(cfg))
    return path


def _resolve_metric(base_config: Path, metric: str | None) -> str:
    if metric is not None:
        return metric
    cfg = load_config(["--config", str(base_config)])
    eval_names = [name for name, value in (cfg.data.eval or {}).items() if value is not None]
    if len(eval_names) != 1:
        raise ValueError("--metric is required unless the base config has exactly one eval task")
    return f"eval/{eval_names[0]}/loss"


def _input_candidates(source: Path, raw: str | None, names: tuple[str, ...]) -> list[Params]:
    if raw is not None:
        return parse_candidates(raw, names)
    selected = load_phase(source).selected
    if not selected:
        raise ValueError(
            f"source phase {source} has no selected candidates; analyze it or pass --candidates"
        )
    return [dict(candidate) for candidate in selected]


def _run_id(params: Params) -> str:
    return (
        f"{params['preset']}-lr{float(params['lr']):g}"
        f"-om{float(params['output_mult']):g}"
        f"-a{float(params['depth_exponent']):g}-s{int(params['seed'])}"
    )


def _generate_phase(
    *,
    name: str,
    base_config: Path,
    out: Path,
    metric: str | None,
    source: Path | None,
    cells: list[Params],
    num_processes: int,
    accelerate_config: Path | None,
) -> tuple[Path, list[list[str]]]:
    base_config = base_config.resolve()
    out = out.resolve()
    source = source.resolve() if source is not None else None
    out.mkdir(parents=True, exist_ok=True)
    runs: list[RunSpec] = []
    commands: list[list[str]] = []
    for params in cells:
        run_id = _run_id(params)
        run_dir = out / "runs" / run_id
        config = _write_run_config(base_config, run_dir, params, num_processes=num_processes)
        result = run_dir / "result.json"
        runs.append(
            RunSpec(
                run_id,
                str(relative_path(config, out)),
                str(relative_path(result, out)),
                params,
            )
        )
        commands.append(
            accelerate_argv(
                config=config,
                result=result,
                num_processes=num_processes,
                accelerate_config=accelerate_config,
            )
        )
    phase_path = out / "phase.json"
    write_phase(
        phase_path,
        PhaseManifest(
            version=1,
            phase=name,
            metric=_resolve_metric(base_config, metric),
            source=str(relative_path(source, out)) if source is not None else None,
            runs=runs,
            ranking=[],
            selected=[],
        ),
    )
    (out / "commands.txt").write_text(
        "\n".join(shlex.join(command) for command in commands) + "\n"
    )
    return phase_path, commands


def _smoke_cells(
    lrs: list[float], global_examples: int, seed: int, steps: int, warmup: int
) -> list[Params]:
    return [
        _cell(
            preset="170M",
            lr=lr,
            output_mult=1.0,
            depth_exponent=0.0,
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
        )
        for lr in lrs
    ]


def _coarse_cells(
    lrs: list[float], global_examples: int, seed: int, steps: int, warmup: int
) -> list[Params]:
    return _smoke_cells(lrs, global_examples, seed, steps, warmup)


def _refine_cells(
    candidates: list[Params],
    output_mults: list[float],
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    return [
        _cell(
            preset="170M",
            lr=float(candidate["lr"]),
            output_mult=output_mult,
            depth_exponent=0.0,
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
        )
        for candidate in candidates
        for output_mult in output_mults
    ]


def _smoke_gated_lrs(source: Path, lrs: list[float]) -> list[float]:
    phase = load_phase(source)
    runs_by_lr = {float(run.params["lr"]): run for run in phase.runs}
    phase_dir = source.resolve().parent
    for lr in SMOKE_LRS[:2]:
        run = runs_by_lr.get(lr)
        if run is None or result_metric(phase_dir, run, phase.metric) is None:
            raise ValueError(f"smoke lr={lr:g} lacks a finite {phase.metric}")
    high_run = runs_by_lr.get(SMOKE_LRS[-1])
    if high_run is None or result_metric(phase_dir, high_run, phase.metric) is None:
        return [lr for lr in lrs if lr != SMOKE_LRS[-1]]
    return lrs


@app.command()
def smoke(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    lrs: Annotated[str, typer.Option("--lrs")] = "0.0025,0.01,0.04",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 2048,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 1000,
    warmup: Annotated[int, typer.Option("--warmup")] = 100,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the three-cell production smoke phase."""
    try:
        _generate_phase(
            name="smoke",
            base_config=config,
            out=out,
            metric=metric,
            source=None,
            cells=_smoke_cells(
                parse_floats(lrs, name="--lrs"), global_examples, seed, steps, warmup
            ),
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def coarse(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    source: Annotated[
        Path | None, typer.Option("--from", exists=True, dir_okay=False)
    ] = None,
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    lrs: Annotated[str, typer.Option("--lrs")] = "0.0025,0.004,0.0063,0.01,0.016,0.025,0.04",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 2048,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 10000,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the production coarse LR phase, optionally gated by smoke results."""
    try:
        lr_values = parse_floats(lrs, name="--lrs")
        if source is not None:
            lr_values = _smoke_gated_lrs(source, lr_values)
        _generate_phase(
            name="coarse",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=_coarse_cells(lr_values, global_examples, seed, steps, warmup),
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def refine(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    candidates: Annotated[str | None, typer.Option("--candidates")] = None,
    output_mults: Annotated[str, typer.Option("--output-mults")] = "0.5,1,2",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 2048,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 20000,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the production LR × output-multiplier refinement phase."""
    try:
        cells = _refine_cells(
            _input_candidates(source, candidates, ("lr", "output_mult")),
            parse_floats(output_mults, name="--output-mults"),
            global_examples,
            seed,
            steps,
            warmup,
        )
        _generate_phase(
            name="refine",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()

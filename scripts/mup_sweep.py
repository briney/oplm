"""Generate deterministic μP sweep phase artifacts from a production YAML."""

from __future__ import annotations

import math
import shlex
import subprocess
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


def _parse_ints(raw: str, *, name: str) -> list[int]:
    try:
        values = [int(token) for token in raw.split(",") if token.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be comma-separated ints, got {raw!r}") from exc
    if not values:
        raise typer.BadParameter(f"{name} must list at least one value")
    return values


def _parse_strings(raw: str, *, name: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise typer.BadParameter(f"{name} must list at least one value")
    return values


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
    cfg = load_config(["--config", str(base_config)])
    eval_names = [name for name, value in (cfg.data.eval or {}).items() if value is not None]
    allowed_metrics = [f"eval/{name}/loss" for name in eval_names]
    if metric is not None:
        if metric not in allowed_metrics:
            if len(allowed_metrics) == 1:
                expected = allowed_metrics[0]
            elif allowed_metrics:
                expected = f"one of: {', '.join(allowed_metrics)}"
            else:
                expected = "a configured eval/<task>/loss"
            raise ValueError(f"--metric must be {expected}")
        return metric
    if len(eval_names) != 1:
        raise ValueError("--metric is required unless the base config has exactly one eval task")
    return allowed_metrics[0]


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
    accelerate_config = accelerate_config.resolve() if accelerate_config is not None else None
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
    (out / "commands.txt").write_text("\n".join(shlex.join(command) for command in commands) + "\n")
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


def _replicate_cells(
    source_runs: list[RunSpec], candidates: list[Params], seeds: list[int]
) -> list[Params]:
    cells: list[Params] = []
    if 42 not in seeds:
        raise ValueError("replicate seeds must include source seed 42")
    for candidate in candidates:
        source = next(
            run
            for run in source_runs
            if run.params.get("lr") == candidate["lr"]
            and run.params.get("output_mult") == candidate["output_mult"]
            and run.params.get("seed") == 42
        )
        for seed in seeds:
            if seed == 42:
                continue
            cells.append({**source.params, "seed": seed})
    return cells


def _transfer_cells(
    candidates: list[Params],
    *,
    presets: list[str],
    steps: list[int],
    exponents: list[float],
    global_examples: int,
    seed: int,
    warmup: int,
) -> list[Params]:
    if len(presets) != len(steps):
        raise ValueError("--presets and --steps must contain the same number of values")
    return [
        _cell(
            preset=preset,
            lr=float(candidate["lr"]),
            output_mult=float(candidate["output_mult"]),
            depth_exponent=exponent,
            seed=seed,
            global_examples=global_examples,
            max_steps=model_steps,
            warmup_steps=warmup,
        )
        for candidate in candidates
        for preset, model_steps in zip(presets, steps, strict=True)
        for exponent in exponents
    ]


def _bridge_cells(
    candidates: list[Params],
    *,
    multipliers: list[float],
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    candidate = candidates[0]
    return [
        _cell(
            preset="170M",
            lr=float(candidate["lr"]) * multiplier,
            output_mult=float(candidate["output_mult"]),
            depth_exponent=float(candidate["depth_exponent"]),
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
            batch_mult=multiplier,
        )
        for multiplier in multipliers
    ]


def _confirm_cells(
    candidates: list[Params],
    *,
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    return [
        _cell(
            preset="800M",
            lr=float(candidate["lr"]),
            output_mult=float(candidate["output_mult"]),
            depth_exponent=float(candidate["depth_exponent"]),
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
            batch_mult=float(candidate["batch_mult"]),
        )
        for candidate in candidates
    ]


def _scale_cells(
    candidates: list[Params],
    *,
    presets: list[str],
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    candidate = candidates[0]
    return [
        _cell(
            preset=preset,
            lr=float(candidate["lr"]),
            output_mult=float(candidate["output_mult"]),
            depth_exponent=float(candidate["depth_exponent"]),
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
            batch_mult=float(candidate["batch_mult"]),
        )
        for preset in presets
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


def _candidate(params: Params, keys: tuple[str, ...]) -> Params:
    return {key: params[key] for key in keys if key in params}


def _direct_ranking(phase_dir: Path, phase: PhaseManifest) -> list[dict[str, object]]:
    entries = [
        {
            "id": run.id,
            "params": run.params,
            "score": result_metric(phase_dir, run, phase.metric),
        }
        for run in phase.runs
    ]
    return sorted(
        entries,
        key=lambda entry: (
            entry["score"] is None,
            float(entry["score"]) if entry["score"] is not None else math.inf,
            float(entry["params"].get("lr", math.inf)),
        ),
    )


def _source_phase(phase_path: Path, phase: PhaseManifest) -> tuple[Path, PhaseManifest]:
    if phase.source is None:
        raise ValueError(f"{phase.phase} requires a source phase")
    source_path = (phase_path.parent / phase.source).resolve()
    return source_path, load_phase(source_path)


def _aggregate_key(params: Params) -> tuple[float, float, float, float | None]:
    batch_mult = params.get("batch_mult")
    return (
        float(params["lr"]),
        float(params["output_mult"]),
        float(params.get("depth_exponent", 0.0)),
        float(batch_mult) if batch_mult is not None else None,
    )


def _aggregate_params(key: tuple[float, float, float, float | None]) -> Params:
    lr, output_mult, depth_exponent, batch_mult = key
    params: Params = {
        "lr": lr,
        "output_mult": output_mult,
        "depth_exponent": depth_exponent,
    }
    if batch_mult is not None:
        params["batch_mult"] = batch_mult
    return params


def _sort_aggregate(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(entry: dict[str, object]) -> tuple[bool, float, float, float, float, float]:
        params = entry["params"]
        batch_mult = params.get("batch_mult")
        return (
            entry["score"] is None,
            float(entry["score"]) if entry["score"] is not None else math.inf,
            float(params["lr"]),
            float(params["output_mult"]),
            float(params["depth_exponent"]),
            float(batch_mult) if batch_mult is not None else 0.0,
        )

    return sorted(entries, key=key)


def _select_count(ranking: list[dict[str, object]], count: int) -> list[Params]:
    eligible = [entry for entry in ranking if entry["score"] is not None]
    if len(eligible) < count:
        return []
    return [dict(entry["params"]) for entry in eligible[:count]]


def _replicate_ranking(phase_path: Path, phase: PhaseManifest) -> list[dict[str, object]]:
    source_path, source = _source_phase(phase_path, phase)
    locations = [
        *((source_path.parent, run) for run in source.runs),
        *((phase_path.parent, run) for run in phase.runs),
    ]
    keys = {_aggregate_key(run.params) for run in phase.runs}
    entries: list[dict[str, object]] = []
    for key in keys:
        matching = [
            (phase_dir, run) for phase_dir, run in locations if _aggregate_key(run.params) == key
        ]
        seeds = {int(run.params["seed"]) for _, run in matching}
        values = [result_metric(phase_dir, run, phase.metric) for phase_dir, run in matching]
        score = (
            sum(float(value) for value in values) / len(values)
            if 42 in seeds
            and len(seeds) == len(matching)
            and len(seeds) >= 2
            and all(value is not None for value in values)
            else None
        )
        entries.append({"params": _aggregate_params(key), "score": score})
    return _sort_aggregate(entries)


def _transfer_ranking(phase_path: Path, phase: PhaseManifest) -> list[dict[str, object]]:
    keys = {_aggregate_key(run.params) for run in phase.runs}
    presets = {str(run.params["preset"]) for run in phase.runs}
    per_model_rank: dict[tuple[str, tuple[float, float, float, float | None]], int] = {}
    for preset in presets:
        rows = [
            (result_metric(phase_path.parent, run, phase.metric), run)
            for run in phase.runs
            if run.params["preset"] == preset
        ]
        valid = [(loss, run) for loss, run in rows if loss is not None]
        valid.sort(key=lambda row: (float(row[0]), float(row[1].params["lr"])))
        for rank, (_, run) in enumerate(valid, start=1):
            per_model_rank[(preset, _aggregate_key(run.params))] = rank

    entries: list[dict[str, object]] = []
    for key in keys:
        ranks = [per_model_rank.get((preset, key)) for preset in presets]
        score = (
            sum(int(rank) for rank in ranks) if all(rank is not None for rank in ranks) else None
        )
        params = _aggregate_params(key)
        params.pop("batch_mult", None)
        entries.append({"params": params, "score": score})
    return _sort_aggregate(entries)


def analyze_phase(path: Path) -> PhaseManifest:
    path = path.resolve()
    phase = load_phase(path)
    phase.selected = []

    if phase.phase == "replicate":
        phase.ranking = _replicate_ranking(path, phase)
        phase.selected = _select_count(phase.ranking, 2)
    elif phase.phase == "transfer":
        phase.ranking = _transfer_ranking(path, phase)
        phase.selected = _select_count(phase.ranking, 2)
    else:
        phase.ranking = _direct_ranking(path.parent, phase)
        eligible = [entry for entry in phase.ranking if entry["score"] is not None]
        if phase.phase == "smoke":
            scores = {float(entry["params"]["lr"]): entry["score"] for entry in phase.ranking}
            if scores.get(0.0025) is None or scores.get(0.01) is None:
                raise ValueError("smoke requires finite validation loss at LR 0.0025 and 0.01")
        elif phase.phase == "coarse":
            if eligible:
                by_lr = sorted(eligible, key=lambda entry: float(entry["params"]["lr"]))
                winner_lr = float(phase.ranking[0]["params"]["lr"])
                winner_index = next(
                    index
                    for index, entry in enumerate(by_lr)
                    if float(entry["params"]["lr"]) == winner_lr
                )
                if 0 < winner_index < len(by_lr) - 1:
                    phase.selected = [
                        _candidate(entry["params"], ("lr", "output_mult"))
                        for entry in by_lr[winner_index - 1 : winner_index + 2]
                    ]
        elif phase.phase == "refine":
            ranked = [
                {**entry, "params": _candidate(entry["params"], ("lr", "output_mult"))}
                for entry in phase.ranking
            ]
            phase.selected = _select_count(ranked, 2)
        elif phase.phase == "bridge":
            keys = ("lr", "output_mult", "depth_exponent", "batch_mult")
            ranked = [
                {**entry, "params": _candidate(entry["params"], keys)} for entry in phase.ranking
            ]
            phase.selected = _select_count(ranked, 2)
        elif phase.phase == "confirm":
            keys = ("lr", "output_mult", "depth_exponent", "batch_mult")
            ranked = [
                {**entry, "params": _candidate(entry["params"], keys)} for entry in phase.ranking
            ]
            phase.selected = _select_count(ranked, 1)
        elif phase.phase == "scale":
            pass
        else:
            raise ValueError(f"unknown μP phase {phase.phase!r}")

    write_phase(path, phase)
    return phase


@app.command()
def analyze(
    phase_json: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Rank completed cells and update one phase manifest in place."""
    analyze_phase(phase_json)


def _run_local(commands: list[list[str]]) -> None:
    for command in commands:
        subprocess.run(command, check=True)


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
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the three-cell production smoke phase."""
    try:
        phase_path, commands = _generate_phase(
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
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def coarse(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    source: Annotated[Path | None, typer.Option("--from", exists=True, dir_okay=False)] = None,
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    lrs: Annotated[str, typer.Option("--lrs")] = "0.0025,0.004,0.0063,0.01,0.016,0.025,0.04",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 2048,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 10000,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the production coarse LR phase, optionally gated by smoke results."""
    try:
        lr_values = parse_floats(lrs, name="--lrs")
        if source is not None:
            lr_values = _smoke_gated_lrs(source, lr_values)
        phase_path, commands = _generate_phase(
            name="coarse",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=_coarse_cells(lr_values, global_examples, seed, steps, warmup),
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
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
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
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
        phase_path, commands = _generate_phase(
            name="refine",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def replicate(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    candidates: Annotated[str | None, typer.Option("--candidates")] = None,
    seeds: Annotated[str, typer.Option("--seeds")] = "42,43,44",
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate missing replicate seeds while retaining the source seed-42 results."""
    try:
        source_phase = load_phase(source)
        cells = _replicate_cells(
            source_phase.runs,
            _input_candidates(source, candidates, ("lr", "output_mult")),
            _parse_ints(seeds, name="--seeds"),
        )
        phase_path, commands = _generate_phase(
            name="replicate",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def transfer(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    candidates: Annotated[str | None, typer.Option("--candidates")] = None,
    presets: Annotated[str, typer.Option("--presets")] = "400M,800M,1B",
    steps: Annotated[str, typer.Option("--steps")] = "10000,20000,10000",
    exponents: Annotated[str, typer.Option("--exponents")] = "0,0.5,0.75,1.0",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 2048,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the paired LR/output depth-transfer grid."""
    try:
        cells = _transfer_cells(
            _input_candidates(source, candidates, ("lr", "output_mult")),
            presets=_parse_strings(presets, name="--presets"),
            steps=_parse_ints(steps, name="--steps"),
            exponents=parse_floats(exponents, name="--exponents"),
            global_examples=global_examples,
            seed=seed,
            warmup=warmup,
        )
        phase_path, commands = _generate_phase(
            name="transfer",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def bridge(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    candidates: Annotated[str | None, typer.Option("--candidates")] = None,
    multipliers: Annotated[str, typer.Option("--multipliers")] = "0.7,1,1.4,2",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 8192,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 10000,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the production-batch bridge for the top transfer candidate."""
    try:
        cells = _bridge_cells(
            _input_candidates(
                source,
                candidates,
                ("lr", "output_mult", "depth_exponent"),
            ),
            multipliers=parse_floats(multipliers, name="--multipliers"),
            global_examples=global_examples,
            seed=seed,
            steps=steps,
            warmup=warmup,
        )
        phase_path, commands = _generate_phase(
            name="bridge",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def confirm(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    candidates: Annotated[str | None, typer.Option("--candidates")] = None,
    global_examples: Annotated[int, typer.Option("--global-examples")] = 8192,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 10000,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate the production-batch 800M confirmation phase."""
    try:
        cells = _confirm_cells(
            _input_candidates(
                source,
                candidates,
                ("lr", "output_mult", "depth_exponent", "batch_mult"),
            ),
            global_examples=global_examples,
            seed=seed,
            steps=steps,
            warmup=warmup,
        )
        phase_path, commands = _generate_phase(
            name="confirm",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def scale(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    candidates: Annotated[str | None, typer.Option("--candidates")] = None,
    presets: Annotated[str, typer.Option("--presets")] = "50M,170M,400M,800M,1B",
    global_examples: Annotated[int, typer.Option("--global-examples")] = 8192,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    steps: Annotated[int, typer.Option("--steps")] = 100000,
    warmup: Annotated[int, typer.Option("--warmup")] = 5000,
    num_processes: Annotated[int, typer.Option("--num-processes")] = 8,
    local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    accelerate_config: Annotated[
        Path | None, typer.Option("--accelerate-config", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate winner-only scaling runs across the production preset ray."""
    try:
        cells = _scale_cells(
            _input_candidates(
                source,
                candidates,
                ("lr", "output_mult", "depth_exponent", "batch_mult"),
            ),
            presets=_parse_strings(presets, name="--presets"),
            global_examples=global_examples,
            seed=seed,
            steps=steps,
            warmup=warmup,
        )
        phase_path, commands = _generate_phase(
            name="scale",
            base_config=config,
            out=out,
            metric=metric,
            source=source,
            cells=cells,
            num_processes=num_processes,
            accelerate_config=accelerate_config,
        )
        if local:
            _run_local(commands)
            analyze_phase(phase_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()

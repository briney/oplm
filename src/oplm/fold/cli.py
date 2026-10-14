"""``oplm fold`` command group: ``bench-kernels``, ``make-fixtures``, ``measure``, ``predict``
(design §5.2), ``preprocess``, and ``train``.

Keep torch out of module scope: ``oplm.cli`` imports this module eagerly.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import logging
import os
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from oplm.fold.modeling_fold import OplmForFolding
    from oplm.fold.trimul import Backend, Direction, TriangleMultiplication

app = typer.Typer(name="fold", help="Structure prediction head", add_completion=False)
console = Console()

_PATH_TO_BACKEND = {
    "reference": "reference",
    "fused_autograd": "fused",
    "fused_forward_reference_backward": "fused_forward_reference_backward",
}
_DTYPES = {"fp32": "float32", "bf16": "bfloat16"}


def _environment(device: torch.device) -> dict[str, Any]:
    import torch

    try:
        cueq_version: str | None = metadata.version("cuequivariance-torch")
    except metadata.PackageNotFoundError:
        cueq_version = None
    try:
        git_rev: str | None = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,  # this package's repo, not the caller's cwd
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_rev = None
    return {
        "host": platform.node(),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuequivariance_torch": cueq_version,
        "git": git_rev,
    }


def _time(fn: Callable[[], None], device: torch.device, iters: int, warmup: int) -> dict[str, Any]:
    import torch

    try:
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result: dict[str, Any] = {
            "status": "ok",
            "ms_per_iter": (time.perf_counter() - start) * 1000.0 / iters,
        }
        if device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        return result
    except Exception as exc:  # the report records per-path failures instead of aborting
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _phases(
    runner: Callable[..., torch.Tensor], z: torch.Tensor, mask: torch.Tensor
) -> dict[str, Callable[[], None]]:
    """The three timed workloads for one case, bound to this case's module and inputs."""
    import torch
    from torch.utils.checkpoint import checkpoint

    def forward() -> None:
        with torch.no_grad():
            runner(z, mask)

    def forward_backward() -> None:
        runner(z, mask).float().sum().backward()
        z.grad = None

    def checkpointed() -> None:
        checkpoint(runner, z, mask, use_reentrant=False).float().sum().backward()
        z.grad = None

    return {
        "forward": forward,
        "forward_backward": forward_backward,
        "checkpointed": checkpointed,
    }


def _error_vs_fp32(module: TriangleMultiplication, z: torch.Tensor, mask: torch.Tensor) -> float:
    """Max abs difference between ``module`` and an fp32 reference-path copy on the same input."""
    import torch

    with torch.no_grad():
        reference = copy.deepcopy(module).float()
        reference.backend = "reference"
        expected = reference(z.detach().float(), mask)
        return float((module(z, mask).float() - expected).abs().max())


def run_kernel_benchmark(
    *,
    widths: list[int],
    lengths: list[int],
    directions: list[str],
    paths: list[str],
    device: str,
    dtype: str,
    iters: int,
    warmup: int,
    chunk_size: int,
    compile_reference: bool = False,
    error_max_length: int = 512,
) -> dict[str, Any]:
    """Time forward, forward+backward and checkpointed forward+backward per case.

    A case is one (width, length, direction, path). A case whose requested path does not
    resolve on this machine (no CUDA, no cuEquivariance, unsupported width) is reported as
    ``"unavailable"`` with the path that would actually run. ``compile_reference`` wraps the
    ``reference`` path's module in ``torch.compile`` (design §5.2 path 3: the compiled
    reference); ``torch.compiler.reset()`` runs before each compile so every compiled case
    starts from a fresh cache instead of hitting dynamo's recompile limit and silently
    running eager. Numerical error against an fp32 reference is recorded for lengths up to
    ``error_max_length``; a failure there is recorded as ``{"status": "error", ...}`` instead
    of aborting the run.
    """
    import torch

    from oplm.fold.trimul import TriangleMultiplication, resolve_trimul_path

    dev = torch.device(device)
    torch_dtype = getattr(torch, _DTYPES[dtype])
    report: dict[str, Any] = {
        "environment": _environment(dev),
        "settings": {
            "dtype": dtype,
            "iters": iters,
            "warmup": warmup,
            "chunk_size": chunk_size,
            "compile_reference": compile_reference,
        },
        "cases": [],
    }
    for width, length, direction, path in itertools.product(widths, lengths, directions, paths):
        case: dict[str, Any] = {
            "width": width,
            "length": length,
            "direction": direction,
            "path": path,
        }
        torch.manual_seed(0)
        backend = cast("Backend", _PATH_TO_BACKEND[path])
        module = TriangleMultiplication(
            width, cast("Direction", direction), chunk_size=chunk_size, backend=backend
        ).to(dev, torch_dtype)
        z = torch.randn(1, length, length, width, device=dev, dtype=torch_dtype, requires_grad=True)
        mask = torch.ones(1, length, length, device=dev)
        actual = resolve_trimul_path(z, needs_grad=True, backend=module.backend)
        case["actual_path"] = actual
        if actual != path:
            case["status"] = "unavailable"
            report["cases"].append(case)
            continue
        case["status"] = "ok"
        case["compiled"] = compile_reference and path == "reference"
        runner: Callable[..., torch.Tensor] = module
        if case["compiled"]:
            # Each case adds shape/direction/grad-mode guards to the same code object; past
            # dynamo's recompile limit it silently falls back to eager. Start every compiled
            # case from an empty cache so ``compiled: True`` always means compiled timings.
            torch.compiler.reset()
            runner = torch.compile(module, dynamic=False)

        for name, fn in _phases(runner, z, mask).items():
            case[name] = _time(fn, dev, iters, warmup)
        if length <= error_max_length:
            try:
                case["max_abs_error_vs_fp32_reference"] = _error_vs_fp32(module, z, mask)
            except Exception as exc:  # keep the timings already measured for this case
                case["max_abs_error_vs_fp32_reference"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        report["cases"].append(case)
    return report


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x]


def _strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _ms(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "-"
    return f"{entry['ms_per_iter']:.1f}" if entry["status"] == "ok" else "err"


def _err(value: float | dict[str, Any] | None) -> str:
    if value is None:
        return "-"
    return "err" if isinstance(value, dict) else f"{value:.2e}"


@app.command("bench-kernels")
def bench_kernels(
    out: Annotated[Path, typer.Option(help="JSON report path")] = Path("bench-kernels.json"),
    widths: Annotated[str, typer.Option(help="Comma-separated pair widths")] = "128,256",
    lengths: Annotated[
        str, typer.Option(help="Comma-separated token lengths")
    ] = "384,768,1024,1536,2048",
    directions: Annotated[str, typer.Option()] = "outgoing,incoming",
    paths: Annotated[
        str,
        typer.Option(help="reference | fused_autograd | fused_forward_reference_backward"),
    ] = "reference,fused_autograd,fused_forward_reference_backward",
    device: Annotated[str, typer.Option()] = "cuda",
    dtype: Annotated[str, typer.Option(help="fp32 | bf16")] = "bf16",
    iters: Annotated[int, typer.Option()] = 5,
    warmup: Annotated[int, typer.Option()] = 2,
    chunk_size: Annotated[int, typer.Option(help="Reference contraction row chunk")] = 64,
    compile_reference: Annotated[
        bool, typer.Option("--compile-reference", help="torch.compile the reference path")
    ] = False,
) -> None:
    """Time trimul paths and record peak memory, dispatch, versions, and numerical error."""
    unknown = set(_strs(paths)) - set(_PATH_TO_BACKEND)
    if unknown:
        raise typer.BadParameter(
            f"unknown paths {sorted(unknown)}; choose from {sorted(_PATH_TO_BACKEND)}"
        )
    if dtype not in _DTYPES:
        raise typer.BadParameter(f"dtype must be one of {sorted(_DTYPES)}")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise typer.BadParameter(
                "--device cuda requested but CUDA is not available; use --device cpu"
            )
    report = run_kernel_benchmark(
        widths=_ints(widths),
        lengths=_ints(lengths),
        directions=_strs(directions),
        paths=_strs(paths),
        device=device,
        dtype=dtype,
        iters=iters,
        warmup=warmup,
        chunk_size=chunk_size,
        compile_reference=compile_reference,
    )
    out.write_text(json.dumps(report, indent=2))

    table = Table(
        title=f"trimul kernels ({report['environment']['device_name'] or device}, {dtype})"
    )
    for column in (
        "width",
        "length",
        "dir",
        "path",
        "status",
        "fwd ms",
        "fwd+bwd ms",
        "ckpt ms",
        "peak GiB",
        "max err",
    ):
        table.add_column(column)
    for case in report["cases"]:
        peak = case.get("forward_backward", {}).get("peak_allocated_gib")
        path_label = case["path"] + (" (compiled)" if case.get("compiled") else "")
        table.add_row(
            str(case["width"]),
            str(case["length"]),
            case["direction"],
            path_label,
            case["status"],
            _ms(case.get("forward")),
            _ms(case.get("forward_backward")),
            _ms(case.get("checkpointed")),
            f"{peak:.2f}" if peak is not None else "-",
            _err(case.get("max_abs_error_vs_fp32_reference")),
        )
    # Non-tty output (Slurm logs, CI) defaults to 80 columns, which ellipsizes the path and
    # status cells into ambiguity; widen it. A real terminal keeps its own width.
    table_console = console if console.is_terminal else Console(width=max(console.width, 160))
    table_console.print(table)
    console.print(f"Wrote {out}")


@app.command("make-fixtures")
def make_fixtures(
    out: Annotated[Path, typer.Option("--out", help="Fixture directory to create")],
    repo: Annotated[str, typer.Option(help="Upstream HF repo")] = "biohub/ESMFold2-Fast",
    revision: Annotated[
        str, typer.Option(help="Upstream HF revision (sha)")
    ] = "45fe8656f5b3ef493c17fcf9abe9a2968902e712",
    cases: Annotated[str, typer.Option(help="Comma-separated case names, or 'all'")] = "all",
    seed: Annotated[int, typer.Option(help="torch.manual_seed before every upstream forward")] = 0,
) -> None:
    """Record ESMFold2 golden fixtures (requires the pinned `esm` venv; CPU, fp32)."""
    from oplm.fold.fixtures import FIXTURE_CASES, generate_fixtures

    chosen = (
        FIXTURE_CASES
        if cases == "all"
        else tuple(c for c in FIXTURE_CASES if c.name in cases.split(","))
    )
    if not chosen:
        raise typer.BadParameter(f"no fixture case matches {cases!r}")
    path = generate_fixtures(out, repo=repo, revision=revision, cases=chosen, seed=seed)
    console.print(f"[green]fixtures written to {path}[/green]")


@app.command("measure")
def measure(
    config: Annotated[Path, typer.Option("--config", help="Run config YAML with a fold: block")],
    out: Annotated[Path, typer.Option("--out", help="JSON report path")],
    steps: Annotated[
        int,
        typer.Option(
            help="Optimizer steps to run; step_time_s is per optimizer step (all accumulation "
            "micro-steps)"
        ),
    ] = 20,
    overrides: Annotated[list[str] | None, typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Run a few real optimizer steps and record step time and peak memory (docs/FOLD.md §9).

    A failed run still writes the records gathered so far, plus an "error" field, then re-raises.
    """
    from oplm.config import load_config
    from oplm.fold.training import FoldTask, PeakMemoryCallback, require_fold
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config), *(overrides or [])])
    run = require_fold(cfg)
    cfg.train.max_steps = steps
    cfg.train.log_every = 1
    cfg.train.save_every = 10**9
    cfg.train.save_final = False
    cfg.train.save_every_minutes = None
    cfg.train.wandb_enabled = False
    cfg.data.eval = None  # a cost measurement times training steps only
    cfg.train.throughput_warmup_steps = min(3, steps - 1) if steps > 1 else 0
    callback = PeakMemoryCallback()
    task = FoldTask()
    trainer = Trainer(cfg, task=task, callbacks=[callback])
    error: str | None = None
    try:
        trainer.train()
    except BaseException as exc:  # write the partial report below, then re-raise
        error = repr(exc)
        raise
    finally:
        device = trainer.accelerator.device
        model = run.model or {}
        records = callback.records
        settled = [r for r in records if r["step"] > cfg.train.throughput_warmup_steps] or records
        times = [r["step_time_s"] for r in settled if "step_time_s" in r]
        cuda = device.type == "cuda"
        step_time_s = (sum(times) / len(times)) if times else None
        report: dict[str, Any] = {
            "environment": _environment(device),
            "settings": {
                "steps": steps,
                "pair_width": int(model.get("pair_width", 256)),
                "trunk_blocks": int(model.get("trunk_blocks", 24)),
                "lm": run.lm,
                "crop_size": run.crop_size,
                "batch_size": cfg.train.batch_size,
                "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
                "diffusion_samples": run.diffusion_samples,
                "diffusion_microbatch": run.diffusion_microbatch,
                "confidence_rollout_steps": run.confidence_rollout_steps,
                "mixed_precision": cfg.train.mixed_precision,
                "num_processes": trainer.accelerator.num_processes,
            },
            "steps": records,
            "peak_allocated_gib": (
                max((r["peak_allocated_gib"] for r in records), default=None) if cuda else None
            ),
            "peak_reserved_gib": (
                max((r["peak_reserved_gib"] for r in records), default=None) if cuda else None
            ),
            # Per optimizer step: includes all gradient_accumulation_steps micro-steps.
            "step_time_s": step_time_s,
            "micro_step_time_s": (
                None if step_time_s is None else step_time_s / cfg.train.gradient_accumulation_steps
            ),
        }
        if error is not None:
            report["error"] = error
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        console.print(
            f"wrote {out}  steps {steps}  step_time_s {report['step_time_s']} "
            f"(per optimizer step, all accumulation micro-steps)  "
            f"peak_allocated_gib {report['peak_allocated_gib']}"
        )


def _load_folder(model: Path, lm: str | None, device: str) -> tuple[OplmForFolding, torch.device]:
    """Load a fold head onto ``device`` ("auto" picks CUDA when present) with its frozen LM."""
    import torch

    from oplm.fold.modeling_fold import OplmForFolding

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    folder = OplmForFolding.from_pretrained(model, lm_name_or_path=lm).to(dev).eval()
    if folder.lm is None:
        raise typer.BadParameter(
            "no language model: pass --lm or save the checkpoint with lm_name_or_path"
        )
    folder.lm.to(dev)
    if dev.type == "cuda":
        folder.lm.to(torch.bfloat16)  # spec §5.4: the frozen LM runs in bf16 on CUDA
    return folder, dev


@app.command("predict")
def predict(
    chains: Annotated[
        list[str], typer.Argument(help="Chains as SEQ, ID:SEQ or ID:SEQ*N (homo-oligomer copies)")
    ],
    model: Annotated[Path, typer.Option("--model", help="Fold checkpoint directory")],
    out: Annotated[Path, typer.Option("--out", help="Output .cif path")],
    lm: Annotated[
        str | None,
        typer.Option("--lm", help="Frozen LM path or Hub id (overrides the checkpoint's)"),
    ] = None,
    samples: Annotated[
        int, typer.Option(help="Diffusion samples; the best by ipTM (complex) or pTM is written")
    ] = 1,
    loops: Annotated[
        int | None, typer.Option(help="Recurrence iterations (default: config)")
    ] = None,
    steps: Annotated[
        int | None, typer.Option(help="Sampler steps before the sigma cap (default: config)")
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Seed for the initial pair state and the sampler")
    ] = None,
    device: Annotated[str, typer.Option(help="cuda, cpu or auto")] = "auto",
) -> None:
    """Predict a protein (complex) structure and write it as mmCIF."""
    from oplm.fold.predict import fold, parse_chain_arg, write_mmcif

    specs = [parse_chain_arg(text, chr(ord("A") + i)) for i, text in enumerate(chains)]
    folder, _ = _load_folder(model, lm, device)
    result = fold(folder, specs, num_samples=samples, num_loops=loops, num_steps=steps, seed=seed)
    write_mmcif(result, out)
    console.print(
        f"wrote {out}  pTM {result.ptm:.3f}  ipTM {result.iptm:.3f}  "
        f"mean pLDDT {float(result.plddt.mean()) * 100:.1f}  sample {result.best_sample}/{samples}"
    )


_WEIGHT_DIRS = {"ema": "hf_ema", "raw": "hf"}


def _resolve_model_dir(model: Path, weights: str) -> Path:
    """An HF directory as given, or a checkpoint directory's ``hf_ema/`` / ``hf/``."""
    if (model / "config.json").is_file():
        return model
    if weights not in _WEIGHT_DIRS:
        raise typer.BadParameter(f"--weights must be ema or raw, got {weights!r}")
    path = model / _WEIGHT_DIRS[weights]
    if not (path / "config.json").is_file():
        raise typer.BadParameter(f"{model} has no {_WEIGHT_DIRS[weights]}/ model directory")
    return path


@app.command("eval")
def eval_(
    model: Annotated[
        Path, typer.Option("--model", help="A checkpoint directory (see --weights) or HF dir")
    ],
    data: Annotated[
        Path, typer.Option("--data", help="Preprocessed dataset (index.parquet, shards/, targets/)")
    ],
    target_sets: Annotated[
        list[str], typer.Option("--target-set", help="cameo, casp14, valid, ... (repeatable)")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory")],
    weights: Annotated[str, typer.Option(help="ema or raw, for a checkpoint directory")] = "ema",
    lm: Annotated[
        str | None,
        typer.Option("--lm", help="Frozen LM path or Hub id (overrides the checkpoint's)"),
    ] = None,
    loops: Annotated[
        int | None, typer.Option(help="Recurrence iterations (default: config)")
    ] = None,
    steps: Annotated[int | None, typer.Option(help="Sampler steps (default: config)")] = None,
    samples: Annotated[int, typer.Option(help="Diffusion samples; the best by pTM is scored")] = 1,
    seed: Annotated[int, typer.Option(help="Seed for every target")] = 0,
    max_length: Annotated[
        int | None, typer.Option(help="Skip longer chains (default: lm_max_chain_length)")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Score only the first N per set")] = None,
    device: Annotated[str, typer.Option(help="cuda, cpu or auto")] = "auto",
) -> None:
    """Score a checkpoint on held-out target sets: per-target CSV plus a JSON summary.

    Full-chain TM-score, CA and all-atom lDDT, mean pLDDT and pTM (docs/FOLD.md §10).
    """
    import datetime

    import torch

    import oplm
    from oplm.fold.eval.monomer import SamplingSettings, score_target, summarize, write_scores
    from oplm.fold.eval.targets import load_targets

    hf_dir = _resolve_model_dir(model, weights)
    folder, dev = _load_folder(hf_dir, lm, device)
    settings = SamplingSettings(num_loops=loops, num_steps=steps, num_samples=samples, seed=seed)
    cap = max_length or folder.config.lm_max_chain_length
    started = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    loaded = []  # every set is validated before any folding starts
    for name in target_sets:
        try:
            loaded.append((name, *load_targets(data, name, max_length=cap, limit=limit)))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    rows = []
    sets: dict[str, Any] = {}
    for name, targets, coverage in loaded:
        scores = []
        for i, target in enumerate(targets, 1):
            scores.append(score_target(folder, target, settings))
            console.print(
                f"[{name} {i}/{len(targets)}] {target.key} {scores[-1].status} "
                f"TM {scores[-1].tm_score:.3f}"
            )
        rows.extend(scores)
        sets[name] = {
            "coverage": {
                "listed": coverage.listed,
                "missing": coverage.missing,
                "skipped": coverage.skipped,
                "selected": coverage.selected,
            },
            "metrics": summarize(scores),
        }
    out.mkdir(parents=True, exist_ok=True)
    write_scores(out / "targets.csv", rows)
    summary = {
        "model": str(hf_dir),
        "weights": weights,
        "lm": lm or folder.config.lm_name_or_path,
        "data": str(data),
        "sampling": dataclasses.asdict(settings),
        "resolved_sampling": {
            "num_loops": loops or folder.config.inference_num_loops,
            "num_steps": steps or folder.config.inference_num_steps,
        },
        "device": str(dev),
        "versions": {"oplm": oplm.__version__, "torch": torch.__version__},
        "started": started,
        "seconds": time.perf_counter() - clock,
        "target_sets": sets,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, block in sets.items():
        m = block["metrics"]
        console.print(
            f"{name}: {int(m['targets'])} scored, {int(m['failed'])} failed, "
            f"{len(block['coverage']['skipped'])} skipped  TM {m['tm_score']:.3f}  "
            f"lDDT {m['lddt']:.3f}  pLDDT {m['plddt']:.3f}"
        )
    console.print(f"wrote {out / 'targets.csv'} and {out / 'summary.json'}")


@app.command("compare")
def compare(
    a: Annotated[Path, typer.Argument(help="Baseline eval directory or targets.csv")],
    b: Annotated[Path, typer.Argument(help="Candidate eval directory or targets.csv")],
    metric: Annotated[str, typer.Option(help="tm_score, lddt_ca, lddt, plddt or ptm")] = "tm_score",
    target_set: Annotated[
        str | None, typer.Option("--target-set", help="Only this set (default: all, pooled)")
    ] = None,
    samples: Annotated[int, typer.Option(help="Bootstrap resamples")] = 10_000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
) -> None:
    """Paired B − A over targets scored in both, with a bootstrap 95 % CI, as JSON."""
    from oplm.fold.eval.monomer import compare_scores, read_scores

    def table(path: Path) -> list[Any]:
        rows = read_scores(path / "targets.csv" if path.is_dir() else path)
        return [r for r in rows if target_set is None or r.target_set == target_set]

    try:
        result = compare_scores(table(a), table(b), metric=metric, samples=samples, seed=seed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=1))


@app.command("preprocess")
def preprocess_cmd(
    archive: Annotated[Path, typer.Option("--archive", help="Root of the mmCIF tree")],
    ccd: Annotated[Path, typer.Option("--ccd", help="components.cif.gz or a ccd_table.json")],
    out: Annotated[Path, typer.Option("--out", help="Dataset directory to create")],
    workers: Annotated[int, typer.Option(help="Parser processes")] = max(
        1, (os.cpu_count() or 2) - 1
    ),
    shard_bytes: Annotated[int, typer.Option(help="Target bytes per shard")] = 1 << 30,
    row_group_size: Annotated[int, typer.Option(help="Rows per Parquet row group")] = 16,
    cutoff: Annotated[str, typer.Option(help="Training release-date cutoff (ISO)")] = "2021-09-30",
    valid_size: Annotated[int, typer.Option(help="Post-cutoff monomer chains held for valid")] = 64,
    valid_seed: Annotated[int, typer.Option()] = 0,
    heldout: Annotated[
        list[str] | None, typer.Option("--heldout", help="name=path of a target list; repeatable")
    ] = None,
    ids: Annotated[Path | None, typer.Option(help="Restrict to the entry ids in this file")] = None,
) -> None:
    """Parse an mmCIF archive into structure shards, index, splits and manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from oplm.fold.data.dataset import PreprocessConfig, preprocess

    lists: dict[str, Path] = {}
    for item in heldout or []:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise typer.BadParameter(f"--heldout expects name=path, got {item!r}")
        lists[name] = Path(path)
    manifest = preprocess(
        PreprocessConfig(
            archive=archive,
            ccd=ccd,
            out=out,
            workers=workers,
            shard_bytes=shard_bytes,
            row_group_size=row_group_size,
            cutoff=cutoff,
            valid_size=valid_size,
            valid_seed=valid_seed,
            heldout=lists,
            ids=ids,
        )
    )
    table = Table(title=f"preprocess -> {out}")
    table.add_column("key")
    table.add_column("value")
    table.add_row("inputs", str(manifest["inputs"]))
    table.add_row("accepted", str(manifest["accepted"]))
    table.add_row("rejected", json.dumps(manifest["rejected"]))
    table.add_row("dropped", json.dumps(manifest["dropped"]))
    table.add_row("chains", str(manifest["chains"]))
    table.add_row("splits", json.dumps(manifest["splits"]))
    table.add_row("shards", str(len(manifest["shards"])))
    console.print(table)


@app.command("train")
def train(
    overrides: Annotated[
        list[str] | None, typer.Argument(help="Dotlist overrides, e.g. train.max_steps=10")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Run config YAML with a fold: block")
    ] = None,
    name: Annotated[
        str | None, typer.Option("--name", help="W&B run name and ./<name> output dir")
    ] = None,
) -> None:
    """Train the structure head.

    Single process; use `accelerate launch -m oplm.fold.train` for DDP.
    """
    from oplm.config import load_config

    argv: list[str] = []
    if config is not None:
        argv += ["--config", str(config)]
    if name is not None:
        argv += ["--name", name]
    argv += list(overrides or [])
    cfg = load_config(argv)
    if cfg.fold is None:
        raise typer.BadParameter("the config has no fold: block (docs/FOLD.md §9)")
    model = cfg.fold.model or {}
    console.print(
        f"[bold]Head:[/bold] pair_width {model.get('pair_width', 256)} / trunk_blocks "
        f"{model.get('trunk_blocks', 24)}  [bold]LM:[/bold] {cfg.fold.lm}  "
        f"[bold]crop:[/bold] {cfg.fold.crop_size}"
    )
    console.print(f"[bold]Output:[/bold] {cfg.train.output_dir}")
    from oplm.fold.train import main as train_main

    train_main(cfg)

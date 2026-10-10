"""``oplm fold`` command group. Milestone 0 ships ``bench-kernels`` (design §5.2).

Keep torch out of module scope: ``oplm.cli`` imports this module eagerly.
"""

from __future__ import annotations

import copy
import itertools
import json
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

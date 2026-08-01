"""`oplm slurm` command surface: turn a training config into job scripts."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003  # Typer resolves annotations at runtime.
from typing import Annotated, Any

import typer
from rich.console import Console

from oplm.slurm.config import load_slurm_config
from oplm.slurm.render import JobSpec, SubmitEntry, accelerate_command, render_job
from oplm.slurm.submit import running_job_ids, submit_all

app = typer.Typer(name="slurm", help="Generate and submit Slurm jobs", add_completion=False)
console = Console()

_MANIFEST = "jobs.json"


def _require_manifest(directory: Path) -> dict[str, Any]:
    """Load ``directory``'s job manifest.

    Args:
        directory: Directory a prior ``generate`` call wrote into.

    Returns:
        The parsed manifest (``{"scripts": [...], "job_ids": {...}}``).

    Raises:
        typer.BadParameter: If ``directory`` has no manifest, i.e. `generate` has not
            been run there yet.
    """
    manifest_path = directory / _MANIFEST
    if not manifest_path.exists():
        raise typer.BadParameter(f"no {_MANIFEST} in {directory}; run `oplm slurm generate` first")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    return manifest


@app.command()
def generate(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    nodes: Annotated[int | None, typer.Option("--nodes")] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
    time_limit: Annotated[str | None, typer.Option("--time-limit")] = None,
) -> None:
    """Write one sbatch script, plus a job manifest, for a training config."""
    # The rendered command runs on a compute node with a job-scoped $JOB_WORK_DIR as its
    # container working directory, not the directory `generate` was invoked from -- a relative
    # --config would silently resolve to the wrong (likely nonexistent) file there.
    config = config.resolve()
    out = out.resolve()
    try:
        slurm = load_slurm_config(config)
        job_name = name or f"oplm-{preset or 'run'}"
        resolved_nodes = (
            nodes if nodes is not None else slurm.nodes.resolve(phase=None, preset=preset)
        )
        resolved_time = time_limit or slurm.time_limit.resolve(phase=None, preset=preset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyError as exc:
        # KeyError.__str__ wraps its message in repr() quoting; args[0] is the plain message.
        raise typer.BadParameter(str(exc.args[0])) from exc

    args = f"--config {config}"
    if preset is not None:
        args += f" --preset {preset}"
    spec = JobSpec(
        name=job_name,
        nodes=resolved_nodes,
        time_limit=resolved_time,
        command=accelerate_command(
            module="oplm.train", gpus_per_node=slurm.gpus_per_node, args=args
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    script = out / f"{job_name}.sbatch"
    script.write_text(render_job(spec, slurm))
    (out / _MANIFEST).write_text(
        json.dumps({"scripts": [script.name], "job_ids": {}}, indent=2) + "\n"
    )
    console.print(f"wrote {script} ({resolved_nodes} nodes, {resolved_time})")


@app.command()
def submit(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Submit every script a previous `generate` wrote into DIRECTORY."""
    manifest = _require_manifest(directory)
    entries = [
        SubmitEntry(var=f"JOB_{index}", script=Path(script))
        for index, script in enumerate(manifest["scripts"])
    ]
    ids = submit_all(entries, base_dir=directory)
    manifest["job_ids"] = ids
    (directory / _MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    for var, job_id in ids.items():
        console.print(f"{var}: {job_id}")


@app.command()
def status(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Report which submitted jobs are still known to the scheduler."""
    manifest = _require_manifest(directory)
    ids: dict[str, str] = manifest.get("job_ids", {})
    if not ids:
        console.print("no jobs submitted yet")
        raise typer.Exit()
    query = running_job_ids(list(ids.values()))
    if not query.reachable:
        # `query.ids` is empty both when nothing is running AND when the scheduler could not be
        # reached (squeue absent, or its controller wedged past the query timeout); without this
        # check, every job here would print as "finished or unknown" purely because the
        # scheduler can't be queried, which looks identical to everything actually having
        # finished. Say so explicitly instead.
        console.print(
            "[yellow]scheduler unreachable[/yellow]: cannot query squeue (not found, or its "
            "controller did not answer before the timeout); listing submitted job ids only"
        )
        for var, job_id in ids.items():
            console.print(f"{var}: {job_id}")
        raise typer.Exit()
    for var, job_id in ids.items():
        state = "active" if job_id in query.ids else "finished or unknown"
        console.print(f"{var} ({job_id}): {state}")

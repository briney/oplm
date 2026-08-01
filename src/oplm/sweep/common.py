"""Shared coordinate-check geometry and phase artifacts for the μP scripts.

The width→(layers, heads) mapping lives here so coordinate checks and sweep phases
use the same model geometry. The phase helpers provide the small JSON artifact and
launcher contract shared by the sweep phases and runner.

``head_dim`` is fixed at 64 across OPLM presets (only the head *count* grows), so
every width is a multiple of it and the attention softmax scaling stays
width-invariant. ``preset_ray`` co-scales depth with width at the 32:1 preset
aspect ratio (the 50M preset is 512/16).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer

JsonScalar = str | int | float | bool | None
Params = dict[str, JsonScalar]

# head_dim is fixed across presets: every width must be a positive multiple of it.
HEAD_DIM = 64
# Preset aspect ratio hidden/layers (the 50M preset is 512/16 = 32).
PRESET_ASPECT_RATIO = 32


class Scaling(StrEnum):
    """How the per-width geometry is built."""

    width = "width"  # μP gate: vary hidden_size only, depth fixed
    preset_ray = "preset_ray"  # vary hidden_size AND depth at the preset ratio


class Optimizer(StrEnum):
    """Optimizer choices shared by the μP scripts."""

    muon = "muon"
    adamw = "adamw"


@dataclass(frozen=True)
class RunSpec:
    id: str
    config: str
    result: str
    params: Params


@dataclass
class PhaseManifest:
    version: int
    phase: str
    metric: str
    source: str | None
    runs: list[RunSpec]
    ranking: list[dict[str, Any]]
    selected: list[Params]
    oplm_version: str | None = None
    generated_at: str | None = None
    job_ids: dict[str, str] | None = None


def parse_widths(raw: str) -> list[int]:
    """Parse a comma-separated width list and validate the head-dim constraint."""
    try:
        widths = [int(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"widths must be comma-separated ints, got {raw!r}") from exc
    if not widths:
        raise typer.BadParameter("widths must list at least one width")
    bad = [w for w in widths if w % HEAD_DIM != 0 or w // HEAD_DIM < 1]
    if bad:
        raise typer.BadParameter(
            f"each width must be a positive multiple of head_dim={HEAD_DIM}; got {bad}"
        )
    return widths


def parse_floats(raw: str, *, name: str) -> list[float]:
    """Parse a comma-separated float list (e.g. an LR grid)."""
    try:
        values = [float(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be comma-separated floats, got {raw!r}") from exc
    if not values:
        raise typer.BadParameter(f"{name} must list at least one value")
    return values


def num_layers_for(width: int, depth: int, scaling: Scaling) -> int:
    """Layer count for a width: fixed ``depth`` (width mode) or co-scaled (preset_ray)."""
    if scaling is Scaling.preset_ray:
        return max(1, round(width / PRESET_ASPECT_RATIO))
    return depth


def gradient_accumulation_steps(
    global_examples: int, *, per_device_batch: int, world_size: int
) -> int:
    denominator = per_device_batch * world_size
    if global_examples < 1 or denominator < 1 or global_examples % denominator != 0:
        raise ValueError(
            "global examples must be a positive multiple of "
            f"per_device_batch * world_size ({denominator}), got {global_examples}"
        )
    return global_examples // denominator


def parse_candidates(raw: str, names: tuple[str, ...]) -> list[Params]:
    candidates: list[Params] = []
    for item in raw.split(","):
        values = item.split(":")
        if len(values) != len(names):
            raise typer.BadParameter(
                f"each candidate must contain {len(names)} colon-separated values: {item!r}"
            )
        try:
            candidates.append(dict(zip(names, (float(value) for value in values), strict=True)))
        except ValueError as exc:
            raise typer.BadParameter(f"candidate values must be numeric: {item!r}") from exc
    if not candidates:
        raise typer.BadParameter("candidates must not be empty")
    return candidates


def relative_path(path: Path, start: Path) -> Path:
    path_parts = path.resolve().parts
    start_parts = start.resolve().parts
    common = 0
    for left, right in zip(path_parts, start_parts, strict=False):
        if left != right:
            break
        common += 1
    return Path(*([".."] * (len(start_parts) - common)), *path_parts[common:])


def write_phase(path: Path, phase: PhaseManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(phase), indent=2, sort_keys=True) + "\n")


def load_phase(path: Path) -> PhaseManifest:
    raw = json.loads(path.read_text())
    return PhaseManifest(
        version=int(raw["version"]),
        phase=str(raw["phase"]),
        metric=str(raw["metric"]),
        source=raw["source"],
        runs=[RunSpec(**run) for run in raw["runs"]],
        ranking=list(raw["ranking"]),
        selected=list(raw["selected"]),
        # `.get(...)` defaults, not `raw[...]`, so a `phase.json` written before this task
        # (which introduced these three provenance fields) still loads.
        oplm_version=raw.get("oplm_version"),
        generated_at=raw.get("generated_at"),
        job_ids=raw.get("job_ids"),
    )


def result_metric(phase_dir: Path, run: RunSpec, metric: str) -> float | None:
    path = phase_dir / run.result
    if not path.exists():
        return None
    value = (json.loads(path.read_text()).get("eval") or {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def accelerate_argv(
    *, config: Path, result: Path, num_processes: int, accelerate_config: Path | None
) -> list[str]:
    if num_processes < 1:
        raise ValueError(f"num_processes must be >= 1, got {num_processes}")
    argv = ["accelerate", "launch"]
    if accelerate_config is not None:
        argv.extend(["--config_file", str(accelerate_config)])
    argv.extend(
        [
            "--num_processes",
            str(num_processes),
            "-m",
            "oplm.sweep.run",
            "--config",
            str(config),
            "--result",
            str(result),
        ]
    )
    return argv

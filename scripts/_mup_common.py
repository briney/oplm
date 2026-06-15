"""Shared geometry + parsing for the μP scripts (coord-check + LR sweep).

The width→(layers, heads) mapping lives here so the coordinate check
(:mod:`scripts.mup_coord_check`) validates the *same* model geometry the sweep
(:mod:`scripts.mup_pilot_run` / :mod:`scripts.mup_sweep`) trains — otherwise the
gate would certify a different model than the one whose LR is being tuned.

``head_dim`` is fixed at 64 across OPLM presets (only the head *count* grows), so
every width is a multiple of it and the attention softmax scaling stays
width-invariant. ``preset_ray`` co-scales depth with width at the 32:1 preset
aspect ratio (the 50M preset is 512/16).
"""

from __future__ import annotations

from enum import StrEnum

import typer

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

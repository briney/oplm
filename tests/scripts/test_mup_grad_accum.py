"""Tests that the μP sweep/pilot CLIs expose ``--grad-accum`` (default 1).

Gradient accumulation is how a single-GPU pilot reaches the production global batch
(``batch_size × grad_accum``) so the LR is tuned at the production batch — μP
transfers LR across width, not batch size (see ``docs/MUP.md``). These guard the
flag (and its no-op default) against regressions.

The flag name is read from the typer app's underlying click command (the actual
parsed CLI), not the rendered ``--help`` text — the latter is Rich-formatted and
line-wraps by terminal width, so substring checks are flaky under CI's narrow
non-TTY console.
"""

from __future__ import annotations

import inspect

import typer
from scripts import mup_pilot_run, mup_sweep


def _cli_flags(app: typer.Typer) -> set[str]:
    """Return every option flag string the typer app exposes (via its click command)."""
    command = typer.main.get_command(app)
    subcommands = getattr(command, "commands", None)
    commands = list(subcommands.values()) if subcommands else [command]
    return {opt for cmd in commands for param in cmd.params for opt in param.opts}


def test_pilot_exposes_grad_accum_default_one() -> None:
    """``mup_pilot_run`` registers ``--grad-accum`` defaulting to 1 (a no-op)."""
    assert inspect.signature(mup_pilot_run.main).parameters["grad_accum"].default == 1
    assert "--grad-accum" in _cli_flags(mup_pilot_run.app)


def test_sweep_exposes_grad_accum_default_one() -> None:
    """``mup_sweep`` registers ``--grad-accum`` defaulting to 1 (a no-op)."""
    assert inspect.signature(mup_sweep.main).parameters["grad_accum"].default == 1
    assert "--grad-accum" in _cli_flags(mup_sweep.app)

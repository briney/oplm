"""Tests that the μP sweep/pilot CLIs expose ``--grad-accum`` (default 1).

Gradient accumulation is how a single-GPU pilot reaches the production global batch
(``batch_size × grad_accum``) so the LR is tuned at the production batch — μP
transfers LR across width, not batch size (see ``docs/MUP.md``). These guard the
flag (and its no-op default) against regressions.
"""

from __future__ import annotations

import inspect

from scripts import mup_pilot_run, mup_sweep
from typer.testing import CliRunner

runner = CliRunner()


def test_pilot_exposes_grad_accum_default_one() -> None:
    """``mup_pilot_run`` registers ``--grad-accum`` defaulting to 1 (a no-op)."""
    assert inspect.signature(mup_pilot_run.main).parameters["grad_accum"].default == 1
    result = runner.invoke(mup_pilot_run.app, ["--help"])
    assert result.exit_code == 0
    assert "--grad-accum" in result.stdout


def test_sweep_exposes_grad_accum_default_one() -> None:
    """``mup_sweep`` registers ``--grad-accum`` defaulting to 1 (a no-op)."""
    assert inspect.signature(mup_sweep.main).parameters["grad_accum"].default == 1
    result = runner.invoke(mup_sweep.app, ["--help"])
    assert result.exit_code == 0
    assert "--grad-accum" in result.stdout

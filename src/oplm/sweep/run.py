"""Run one fully resolved μP sweep cell under Accelerate."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves annotations at runtime.
from typing import Annotated

import typer

from oplm.training.checkpoint import latest_checkpoint

app = typer.Typer(name="mup-run", help=__doc__, add_completion=False)

__all__ = ["app", "latest_checkpoint", "main"]


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result: Annotated[Path, typer.Option("--result", dir_okay=False)],
) -> None:
    """Train one resolved config and write its sweep result.

    Idempotent under Slurm ``--requeue``: every cell opts into ``Trainer``'s auto-resume (see
    ``oplm.training.trainer.Trainer.__init__``), so a requeued relaunch of the same cell picks up
    from the newest committed checkpoint under its output directory rather than restarting at
    step 0. An explicit ``resume_from`` pinned in the cell's config still wins over the scan.
    """
    from oplm.train import _bootstrap_training_environment

    _bootstrap_training_environment()

    from oplm.config import load_config
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config)])
    cfg.train.auto_resume = True
    Trainer(cfg, callbacks=[SweepMetricsCallback(result)]).train()


if __name__ == "__main__":
    app()

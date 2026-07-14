"""Run one fully resolved μP sweep cell under Accelerate."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(name="mup-run", help=__doc__, add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result: Annotated[Path, typer.Option("--result", dir_okay=False)],
) -> None:
    """Train one resolved config and write its sweep result."""
    from oplm.train import _bootstrap_training_environment

    _bootstrap_training_environment()

    from oplm.config import load_config
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config)])
    Trainer(cfg, callbacks=[SweepMetricsCallback(result)]).train()


if __name__ == "__main__":
    app()

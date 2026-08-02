"""Run one fully resolved μP sweep cell under Accelerate."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves annotations at runtime.
from typing import Annotated

import typer

app = typer.Typer(name="mup-run", help=__doc__, add_completion=False)

_CHECKPOINT_PREFIX = "checkpoint-"


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the highest-step checkpoint under ``output_dir``, or ``None``.

    Checkpoints are named ``checkpoint-<global_step>`` (see
    ``oplm.training.trainer.Trainer._save_checkpoint``), so ordering is numeric on the suffix —
    lexicographic ordering would rank ``checkpoint-9000`` above ``checkpoint-10000``.

    Args:
        output_dir: The cell's training output directory (may not exist yet).

    Returns:
        The path to the checkpoint directory with the highest numeric step, or ``None`` if
        ``output_dir`` does not exist or holds no well-formed checkpoint directory.
    """
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{_CHECKPOINT_PREFIX}*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix(_CHECKPOINT_PREFIX)
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return None
    return max(candidates)[1]


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result: Annotated[Path, typer.Option("--result", dir_okay=False)],
) -> None:
    """Train one resolved config and write its sweep result.

    Idempotent under Slurm ``--requeue``: if the cell's output directory already holds a
    checkpoint and the config does not pin ``resume_from``, training resumes from the newest one
    rather than restarting at step 0.
    """
    from oplm.train import _bootstrap_training_environment

    _bootstrap_training_environment()

    from oplm.config import load_config
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config)])
    if cfg.train.resume_from is None:
        checkpoint = latest_checkpoint(Path(cfg.train.output_dir))
        if checkpoint is not None:
            cfg.train.resume_from = str(checkpoint)
    Trainer(cfg, callbacks=[SweepMetricsCallback(result)]).train()


if __name__ == "__main__":
    app()

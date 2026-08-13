"""Worker for the 2-rank async-checkpoint dedicated-process-group e2e (Task 5.1b). Run
under ``torch.distributed.run``.

Reproduces Task 5.1's HSDP-pilot abort shape on plain DDP, without needing HSDP itself:
2 CPU/gloo ranks train with an aggressive periodic-async-save cadence (``save_every=1``)
so a background ``dcp.async_save`` write's own DCP collectives are very likely still in
flight when the training loop's next step issues its own collective (the per-step
control-bundle reduce, then the gradient all-reduce) from the main thread. Before Task
5.1b's fix, both ran on the same default process group and could interleave
nondeterministically -- exactly the ``gloo::EnforceNotMet ... collective mismatch`` abort
documented in ``task-5.1-report.md``'s Concern 1. After the fix, checkpoint I/O's
collectives run on their own dedicated GLOO group (``Trainer._checkpoint_process_group``)
and cannot interleave with the training loop's collectives on the default group.

Two phases, matching the parent test's two subprocess launches:
  - ``auto_resume=False``: fresh run, ``max_steps`` steps at ``save_every=1``, commits one
    checkpoint per step (``save_total_limit=max_steps`` so nothing rotates away yet).
  - ``auto_resume=True``: second launch, same ``run_dir``, picks up the newest committed
    checkpoint and trains onward at the same aggressive cadence.

Each rank records its own resumed-from step and post-training ``global_step`` to
``out_dir/rank<i>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir: str, train_data: str, out_dir: str, max_steps: int, auto_resume: bool) -> None:
    """Train ``max_steps`` steps on 2 ranks at ``save_every=1``, one checkpoint per step.

    Args:
        run_dir: Training output dir checkpoints are committed under.
        train_data: Parquet fixture path for the real dataloader.
        out_dir: Directory to write ``rank<i>.json`` into (must already exist).
        max_steps: ``cfg.train.max_steps`` for this launch (also ``save_total_limit``, so
            every step's checkpoint survives rotation within this launch).
        auto_resume: ``cfg.train.auto_resume`` -- ``True`` for the second launch, which
            must pick up the first launch's newest committed checkpoint.
    """
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(
        Path(run_dir),
        Path(train_data),
        max_steps=max_steps,
        save_every=1,
        save_total_limit=max_steps,
        auto_resume=auto_resume,
        log_every=1,
    )
    trainer = Trainer(cfg, callbacks=[])
    resumed_from_step = trainer.global_step
    trainer.train()

    # trainer.train() force-finalizes any still-pending async save and barriers before
    # returning, so every checkpoint triggered during this launch must already be
    # committed here on every rank.
    rank = trainer.accelerator.process_index
    payload = {
        "resumed_from_step": resumed_from_step,
        "global_step": trainer.global_step,
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "true")

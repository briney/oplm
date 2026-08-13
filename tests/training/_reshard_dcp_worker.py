"""Worker for the world-size-2-to-1 DCP reshard test (Task 2.4). Run under
``torch.distributed.run``.

Constructs a real ``Trainer`` with 2 CPU/gloo ranks, trains a few steps, and lets the
``save_every == max_steps`` cadence commit exactly one DCP checkpoint on the final step.
Rank 0 additionally dumps a clone of a stable, untied weight tensor
(``lm_head.decoder.bias``, not shared with the input embeddings -- see
``OplmForMaskedLM``'s tied-weights map) to ``out_dir/reference_weight.pt`` so the parent
test process can assert the world-size-1 resume restores the exact same values. Each rank
also writes its own post-training ``global_step`` to ``out_dir/rank<i>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main(run_dir: str, train_data: str, out_dir: str, max_steps: int) -> None:
    """Train ``max_steps`` steps on 2 ranks, commit one checkpoint, dump a reference weight.

    Args:
        run_dir: Training output dir the checkpoint is committed under.
        train_data: Parquet fixture path for the real dataloader.
        out_dir: Directory to write ``rank<i>.json``/``reference_weight.pt`` into (must
            already exist).
        max_steps: ``cfg.train.max_steps`` (also used as ``save_every`` so the final step's
            periodic save is the checkpoint under test, with no redundant final save).
    """
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(
        Path(run_dir),
        Path(train_data),
        max_steps=max_steps,
        save_every=max_steps,
        auto_resume=False,
        log_every=1,
    )
    trainer = Trainer(cfg, callbacks=[])
    trainer.train()

    # trainer.train() force-finalizes any still-pending async save and barriers before
    # returning, so the checkpoint must already be committed here on every rank -- assert
    # it explicitly (rather than only implicitly relying on the parent test's own check)
    # so a regression in that finalize-before-return guarantee fails fast, attributed to
    # this worker, instead of surfacing later as a confusing missing-reference-weight
    # error in the parent process.
    assert (Path(run_dir) / f"checkpoint-{max_steps}").is_dir()

    rank = trainer.accelerator.process_index
    if rank == 0:
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        reference_weight = unwrapped.lm_head.decoder.bias.detach().clone()
        torch.save(reference_weight, Path(out_dir) / "reference_weight.pt")

    payload = {"global_step": trainer.global_step}
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))

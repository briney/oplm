"""Worker for the HSDP end-to-end pilot (Task 5.1). Run under ``torch.distributed.run``.

Constructs a real ``Trainer`` with ``train.parallelism=hsdp`` on 2 CPU/gloo ranks (mesh
``(1, 2)`` -- one node, so the replicate dim is trivial and this is plain FSDP over the
2-rank shard dim), trains ``max_steps`` optimizer steps, and lets the ``save_every ==
max_steps`` cadence commit exactly one DCP checkpoint on the final step.

Every rank records what it observed -- the resumed-from step, the post-training step, and
proof that the model really is FSDP2-sharded (``lm_head.decoder.bias`` is a ``DTensor``
over a 2-D mesh named ``("replicate", "shard")``) -- to ``out_dir/rank<i>.json``. Without
that sharding proof the pilot would pass identically under plain DDP, i.e. it would assert
nothing about HSDP.

Only *local* attributes are read after ``trainer.train()`` returns: Accelerate's
``end_training`` (called in ``Trainer.train``'s ``finally``) destroys the process group,
so any collective here -- e.g. ``DTensor.full_tensor()`` -- would fail with "could not
resolve the process group". The parent test gets its reference weights from the committed
checkpoint's own gathered ``hf/`` export instead.

Gradient checkpointing, gradient accumulation, and gradient clipping are all switched on
here deliberately: each one interacts with ``fully_shard`` (activation-checkpoint ordering,
the per-micro-batch reduce-scatter, and ``clip_grad_norm_`` over ``DTensor`` grads), and
this pilot is the gate that they compose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir: str, train_data: str, out_dir: str, max_steps: int, auto_resume: bool) -> None:
    """Train ``max_steps`` steps under HSDP on 2 ranks and commit one checkpoint.

    Args:
        run_dir: Training output dir the checkpoint is committed under.
        train_data: Parquet fixture path for the real dataloader.
        out_dir: Directory to write ``rank<i>.json`` into (must already exist).
        max_steps: ``cfg.train.max_steps`` (also used as ``save_every`` so the final
            step's periodic save is the checkpoint under test, with no redundant final
            save).
        auto_resume: ``cfg.train.auto_resume`` -- ``True`` for the second launch, which
            must pick up the first launch's committed checkpoint.
    """
    from torch.distributed.tensor import DTensor

    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(
        Path(run_dir),
        Path(train_data),
        max_steps=max_steps,
        save_every=max_steps,
        auto_resume=auto_resume,
        log_every=1,
        parallelism="hsdp",
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        # The grad-norm-only diagnostic (probe_every=0) is the one stability-diagnostic
        # combination TrainConfig still allows under hsdp -- exercise it here so the
        # carve-out is proven rather than assumed: clip_grad_norm_ returns a DTensor norm
        # under FSDP2, which the callback then converts to a float on rank 0 alone.
        stability_diagnostics=True,
        stability_probe_every=0,
    )
    trainer = Trainer(cfg, callbacks=[])
    resumed_from_step = trainer.global_step
    trainer.train()

    # trainer.train() force-finalizes any still-pending async save and barriers before
    # returning, so the checkpoint must already be committed here on every rank.
    assert (Path(run_dir) / f"checkpoint-{max_steps}").is_dir()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    weight = unwrapped.lm_head.decoder.bias
    is_dtensor = isinstance(weight, DTensor)

    if is_dtensor:
        mesh = weight.device_mesh
        mesh_shape = list(mesh.shape)
        mesh_dim_names = list(mesh.mesh_dim_names or ())
        # Class names, not repr(): DTensor's placement repr is abbreviated ("R", "S(0)")
        # and is not a stable API to assert on.
        placements = [type(placement).__name__ for placement in weight.placements]
    else:
        mesh_shape, mesh_dim_names, placements = [], [], []

    rank = trainer.accelerator.process_index
    payload = {
        "resumed_from_step": resumed_from_step,
        "global_step": trainer.global_step,
        "is_dtensor": is_dtensor,
        "mesh_shape": mesh_shape,
        "mesh_dim_names": mesh_dim_names,
        "placements": placements,
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "true")

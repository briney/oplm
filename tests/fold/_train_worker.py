"""Two-rank fold training worker: trains from a resolved YAML and records the example stream."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path


def main(config_path: str, result_dir: str) -> None:
    import torch

    from oplm.config import load_config
    from oplm.fold.training import FoldTask
    from oplm.training.trainer import Trainer
    from tests.training.conftest import FullRecordingCallback

    cfg = load_config(["--config", config_path])
    callback = FullRecordingCallback()

    class _Recording(FoldTask):
        ids: list[list[str]] = []
        # Per micro-step in-model draws; the callback's train_logs only fire on the main process.
        loops: list[float] = []
        sigma_mean: list[float] = []

        def step(self, model, batch):
            result = super().step(model, batch)
            self.ids.append(list(batch.ids))
            self.loops.append(result.metrics["loops"])
            self.sigma_mean.append(result.metrics["sigma_mean"])
            return result

    task = _Recording()
    trainer = Trainer(cfg, task=task, callbacks=[callback])
    rank = trainer.accelerator.process_index
    seed = torch.initial_seed()  # FoldTask.build_model reseeds as train.seed + rank
    resumed_from = trainer.global_step
    trainer.train()
    losses = [m["train/loss"] for _, m in callback.train_logs]
    payload = {
        "rank": rank,
        "world_size": trainer.accelerator.num_processes,
        "resumed_from_step": resumed_from,
        "global_step": trainer.global_step,
        "ids": task.ids,
        "all_losses_finite": all(math.isfinite(x) for x in losses),
        "initial_seed": seed,
        "loops": task.loops,
        "sigma_mean": task.sigma_mean,
        "ema_n_averaged": None if trainer._ema is None else int(trainer._ema.n_averaged),
        "is_ddp": type(trainer.model).__name__ == "DistributedDataParallel",
    }
    # Trainer.train() already ran accelerator.end_training(), so no barrier/teardown here.
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except (ValueError, RuntimeError) as exc:
        Path(sys.argv[2], f"rank{os.environ.get('RANK', '0')}.error").write_text(str(exc))
        raise

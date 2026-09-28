"""Instrument the real Trainer and inject failures only in a dedicated drill job."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from oplm.config import OplmConfig, load_config
from oplm.train import _bootstrap_training_environment
from oplm.training.trainer import Trainer


class DrillTrainer(Trainer):
    """Test-only Trainer subclass; ordinary training never imports this module."""

    def __init__(self, cfg: OplmConfig, directory: Path, manifest: dict[str, Any]) -> None:
        self.directory = directory
        self.drill = manifest
        self.attempt = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.injected = False
        super().__init__(cfg)

    def record(self, kind: str, **values: Any) -> None:
        """Append and fsync one rank's evidence before a possible SIGKILL."""
        event = dict(kind=kind, attempt=self.attempt, rank=self.rank, time=time.time(), **values)
        with (self.directory / "events" / f"{self.attempt}-{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def snapshot(self) -> dict[str, Any]:
        """Capture counters and learning rates at save time or immediately after load."""
        return dict(
            step=self.global_step,
            epoch=self.epoch,
            samples=self._samples_seen,
            tokens=self.tokens_seen,
            batches=self._batches_in_epoch,
            lr=[s.get_last_lr() for s in self.schedulers],
            world_size=self.accelerator.num_processes,
            global_batch=self._global_effective_batch_size(),
        )

    def kill(self, **details: Any) -> None:
        """Kill this exact training rank, bypassing Trainer cleanup and final saves."""
        self.record("inject", step=self.global_step, **details)
        os.kill(os.getpid(), signal.SIGKILL)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        is_train = "train/loss" in metrics
        if is_train:
            metrics = metrics | {
                "recovery/attempt": self.attempt,
                "recovery/step": self.global_step,
            }
        super()._log_metrics(metrics)
        if is_train and self.rank == 0:
            self.record("log", step=self.global_step, loss=metrics["train/loss"])
        mode = self.drill["mode"]
        if not is_train or self.attempt != 0 or self.injected or mode == "checkpoint_write":
            return
        if self.global_step != self.drill["fail_step"]:
            return
        # Finish the first periodic checkpoint before the deliberate rollback window.
        if self._pending_save is not None:
            self._finalize_pending_save()
        self.accelerator.wait_for_everyone()
        self.injected = True
        if mode in {"worker_node_loss", "batch_node_loss"}:
            if self.rank == 0:
                self.record("node_ready", step=self.global_step)
            # All ranks pause so no newer checkpoint races the operator's node failure.
            time.sleep(self.drill["node_wait_seconds"])
            raise RuntimeError("node failure did not interrupt the job before the drill deadline")
        if mode == "graceful_drain":
            if self.rank == 0:
                self.record("inject", step=self.global_step)
                os.kill(os.getpid(), signal.SIGUSR1)
        elif self.rank == self.drill["kill_rank"]:
            self.kill()

    def _save_checkpoint(self, *, blocking: bool = True) -> None:
        self.record("save", **self.snapshot())
        if (
            self.drill["mode"] == "checkpoint_write"
            and self.attempt == 0
            and not blocking
            and self.global_step >= 2 * self.drill["save_every"]
            and self.rank == self.drill["kill_rank"]
            and not self.injected
        ):
            from torch.distributed.checkpoint import FileSystemWriter

            self.injected = True
            original = FileSystemWriter.write_data
            trainer = self
            checkpoint_step = self.global_step

            def interrupted_write(writer: Any, plan: Any, planner: Any) -> Any:
                result = original(writer, plan, planner)
                result.wait()
                # Real DCP shard files exist, but this rank never acknowledges its write;
                # DCP metadata finalization and the Trainer's commit cannot finish.
                staging = Path(trainer.cfg.train.output_dir) / f"checkpoint-{checkpoint_step}.tmp"
                shards = [p.name for p in staging.glob("*.distcp") if p.stat().st_size > 0]
                trainer.record("inject", step=checkpoint_step, shards_written=shards)
                os.kill(os.getpid(), signal.SIGKILL)
                return result

            # The writer runs asynchronously, so the patch must outlive this method.
            patch.object(FileSystemWriter, "write_data", interrupted_write).start()
        super()._save_checkpoint(blocking=blocking)
        if blocking and self.rank == 0:
            self.record("commit", step=self.global_step)

    def _finalize_pending_save(self) -> None:
        assert self._pending_save is not None
        step = self._pending_save.global_step
        super()._finalize_pending_save()
        if self.rank == 0:
            self.record("commit", step=step)


def main() -> None:
    """Run under the generated production srun/Accelerate launch stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / "drill.json").read_text())
    _bootstrap_training_environment()
    cfg = load_config(["--config", str(directory / "train.yaml")])
    trainer = DrillTrainer(cfg, directory, manifest)
    identity = None
    if cfg.train.wandb_enabled and trainer.rank == 0:
        import wandb

        if wandb.run is not None:
            identity = dict(
                id=wandb.run.id,
                name=wandb.run.name,
                project=wandb.run.project,
                entity=wandb.run.entity,
            )
    trainer.record(
        "start",
        **trainer.snapshot(),
        wandb=identity,
        node=os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        resume=trainer._resolved_resume_target,
    )
    trainer.accelerator.wait_for_everyone()
    if manifest["mode"] == "crash_loop" and trainer.attempt >= 1:
        if trainer.rank == manifest["kill_rank"]:
            trainer.kill()
        # No peer may take an optimizer step before the second deliberate crash.
        trainer.accelerator.wait_for_everyone()
    trainer.train()
    if trainer.rank == 0:
        trainer.record("complete", step=trainer.global_step)


if __name__ == "__main__":
    main()

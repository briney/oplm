"""Trainer class for OPLM masked language model pretraining."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig
    from oplm.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for OPLM with accelerate, wandb, and rich progress.

    Args:
        cfg: Full OPLM configuration. If ``data.eval`` is configured, an
            :class:`~oplm.eval.Evaluator` is built automatically.
    """

    def __init__(
        self,
        cfg: OplmConfig,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.utils import set_seed

        from oplm.data.loader import build_train_dataloader
        from oplm.model.transformer import OplmForMLM
        from oplm.training.flops import estimate_flops_per_token
        from oplm.training.optim import build_optimizer, build_scheduler

        self.cfg = cfg

        # Build evaluator from config if eval datasets are specified
        self.evaluator: Evaluator | None = None
        if cfg.data.eval is not None:
            from oplm.eval import Evaluator

            self.evaluator = Evaluator(cfg)

        # Seed everything
        set_seed(cfg.train.seed)

        # Accelerator
        log_with = "wandb" if cfg.train.wandb_enabled else None
        self.accelerator = Accelerator(
            mixed_precision=cfg.train.mixed_precision,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            log_with=log_with,
            project_dir=cfg.train.output_dir,
        )

        # Model
        model = OplmForMLM(cfg.model)
        if cfg.model.gradient_checkpointing:
            model.encoder.gradient_checkpointing = True

        # Optimizer and scheduler
        optimizer = build_optimizer(model, cfg.train)
        dataloader = build_train_dataloader(cfg)

        # Compute total_steps
        self.total_steps = self._compute_total_steps(cfg, dataloader)
        scheduler = build_scheduler(optimizer, cfg.train, self.total_steps)

        # Prepare with accelerate
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self._samples_seen = 0
        self._last_eval_loss: float | None = None

        # FLOP estimation
        self.flops_per_token = estimate_flops_per_token(cfg.model)

        # Dataset size for fractional epoch computation
        self._dataset_size = self._get_dataset_size()

        # Resume from checkpoint
        if cfg.train.resume_from is not None:
            self._resume_from_checkpoint(cfg.train.resume_from)

        # Init wandb
        if cfg.train.wandb_enabled:
            init_kwargs: dict[str, Any] = {}
            if cfg.train.wandb_run_name is not None:
                init_kwargs["wandb"] = {"name": cfg.train.wandb_run_name}
            self.accelerator.init_trackers(
                project_name=cfg.train.wandb_project,
                config=_config_to_flat_dict(cfg),
                init_kwargs=init_kwargs,
            )

    def train(self) -> None:
        """Run the training loop."""
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
        )

        cfg = self.cfg.train

        # Rich progress bar (main process only)
        progress: Progress | None = None
        task_id: Any = None
        if self.accelerator.is_main_process:
            progress = Progress(
                TextColumn("{task.fields[status]}"),
                BarColumn(),
                TextColumn("{task.fields[metrics]}"),
                TimeRemainingColumn(),
            )
            task_id = progress.add_task(
                "Training",
                total=self.total_steps,
                completed=self.global_step,
                status=f"{self.global_step}/{self.total_steps}",
                metrics="loss=N/A eval=N/A",
            )
            progress.start()

        self.model.train()
        data_iter = iter(self.dataloader)
        current_loss = float("nan")

        try:
            while self.global_step < self.total_steps:
                # Get next batch, handle epoch boundaries
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.epoch += 1
                    self._set_dataset_epoch(self.epoch)
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Convert (B, T) int mask -> (B, 1, 1, T) bool for SDPA
                attention_mask = batch["attention_mask"].unsqueeze(1).unsqueeze(1).bool()

                # Forward + backward inside accumulation context
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=attention_mask,
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    if cfg.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            cfg.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Track tokens and samples
                tokens_in_batch = batch["attention_mask"].sum().item()
                self.tokens_seen += int(tokens_in_batch) * self.accelerator.num_processes
                self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes

                # Only act on optimizer steps (accumulation boundary)
                if not self.accelerator.sync_gradients:
                    continue

                self.global_step += 1
                current_loss = loss.item()

                # Logging
                if self.global_step % cfg.log_every == 0:
                    self._log_step(current_loss)

                # Evaluation
                eval_metrics = self._run_eval()
                if eval_metrics:
                    if "eval/loss" in eval_metrics:
                        self._last_eval_loss = eval_metrics["eval/loss"]
                    self.accelerator.log(eval_metrics, step=self.global_step)

                # Checkpointing
                if self.global_step % cfg.save_every == 0:
                    self._save_checkpoint()

                # Update progress bar
                if progress is not None and task_id is not None:
                    eval_str = (
                        f"{self._last_eval_loss:.4f}" if self._last_eval_loss is not None else "N/A"
                    )
                    progress.update(
                        task_id,
                        completed=self.global_step,
                        status=f"{self.global_step}/{self.total_steps}",
                        metrics=f"loss={current_loss:.4f} eval={eval_str}",
                    )

            # Final checkpoint
            self._save_checkpoint()

        finally:
            if progress is not None:
                progress.stop()
            self.accelerator.end_training()

    def _run_eval(self) -> dict[str, float]:
        """Run all due evaluations for the current step.

        Delegates to the :class:`~oplm.eval.Evaluator`, which handles per-task
        scheduling internally. Returns an empty dict (no-op) when no evals are
        due or no evaluator is configured.
        """
        if self.evaluator is None:
            return {}

        unwrapped = self.accelerator.unwrap_model(self.model)
        return self.evaluator(unwrapped, self.accelerator, self.global_step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_total_steps(self, cfg: OplmConfig, dataloader: DataLoader) -> int:  # type: ignore[type-arg]
        """Compute total training steps from config."""
        if cfg.train.max_epochs is not None:
            dataset_size = self._get_dataset_size_from_dataloader(dataloader)
            effective_batch = cfg.train.batch_size * cfg.train.gradient_accumulation_steps
            steps_per_epoch = max(1, dataset_size // effective_batch)
            return steps_per_epoch * cfg.train.max_epochs
        return cfg.train.max_steps

    @staticmethod
    def _get_dataset_size_from_dataloader(dataloader: DataLoader) -> int:  # type: ignore[type-arg]
        """Get the dataset size from the dataloader."""
        try:
            return len(dataloader.dataset)  # type: ignore[arg-type]
        except TypeError:
            return 0

    def _get_dataset_size(self) -> int:
        """Get dataset size for fractional epoch computation."""
        return self._get_dataset_size_from_dataloader(self.dataloader)

    def _fractional_epoch(self) -> float:
        """Compute the current fractional epoch."""
        if self._dataset_size <= 0:
            return float(self.epoch)
        return self._samples_seen / self._dataset_size

    def _set_dataset_epoch(self, epoch: int) -> None:
        """Propagate epoch to the dataset for deterministic shuffling."""
        dataset = self.dataloader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def _log_step(self, loss: float) -> None:
        """Log training metrics to wandb."""
        fractional_epoch = self._fractional_epoch()
        cumulative_flops = self.flops_per_token * self.tokens_seen

        metrics = {
            "train/loss": loss,
            "train/epoch": fractional_epoch,
            "train/tokens": self.tokens_seen,
            "train/flops": cumulative_flops,
            "train/lr": self.scheduler.get_last_lr()[0],
        }
        self.accelerator.log(metrics, step=self.global_step)

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        from oplm.training.checkpoint import save_checkpoint

        save_checkpoint(
            accelerator=self.accelerator,
            cfg=self.cfg,
            output_dir=self.cfg.train.output_dir,
            global_step=self.global_step,
            epoch=self.epoch,
            tokens_seen=self.tokens_seen,
            save_total_limit=self.cfg.train.save_total_limit,
        )

    def _resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume training state from a checkpoint."""
        from oplm.training.checkpoint import load_checkpoint

        state = load_checkpoint(self.accelerator, checkpoint_dir)
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.tokens_seen = state["tokens_seen"]

        # Approximate samples_seen from tokens and dataset for epoch tracking
        if self._dataset_size > 0:
            self._samples_seen = int(self._fractional_epoch() * self._dataset_size)

        logger.info(
            "Resumed from checkpoint %s (step=%d, epoch=%d, tokens=%d)",
            checkpoint_dir,
            self.global_step,
            self.epoch,
            self.tokens_seen,
        )


def _config_to_flat_dict(cfg: OplmConfig) -> dict[str, Any]:
    """Flatten OplmConfig to a single-level dict for wandb init."""
    from dataclasses import asdict

    flat: dict[str, Any] = {}
    for section_name, section in asdict(cfg).items():
        if isinstance(section, dict):
            for key, value in section.items():
                flat[f"{section_name}/{key}"] = value
        else:
            flat[section_name] = section
    return flat

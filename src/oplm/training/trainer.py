"""Trainer class for OPLM masked language model pretraining."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig
    from oplm.eval.evaluator import Evaluator
    from oplm.training.callbacks import TrainerCallback

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
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.utils import DataLoaderConfiguration, set_seed
        from rich.console import Console

        from oplm.data.loader import build_train_dataloader
        from oplm.model.transformer import OplmForMLM
        from oplm.training.flops import estimate_flops_per_token
        from oplm.training.optim import build_optimizers, build_schedulers

        self.cfg = cfg
        self.callbacks = list(callbacks or [])

        # Seed everything
        set_seed(cfg.train.seed)

        # Accelerator
        log_with = "wandb" if cfg.train.wandb_enabled else None
        self.accelerator = Accelerator(
            mixed_precision=cfg.train.mixed_precision,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            log_with=log_with,
            project_dir=cfg.train.output_dir,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            step_scheduler_with_optimizer=False,
        )

        # Status helper for user-facing messages (main process only)
        _console = Console()

        def _status(msg: str) -> None:
            if self.accelerator.is_main_process:
                _console.print(msg)

        # Init wandb early so login prompt appears before slow setup steps
        if cfg.train.wandb_enabled:
            _status("[dim]Initializing wandb...[/dim]")
            init_kwargs: dict[str, Any] = {}
            if cfg.train.wandb_run_name is not None:
                init_kwargs["wandb"] = {"name": cfg.train.wandb_run_name}
            self.accelerator.init_trackers(
                project_name=cfg.train.wandb_project,
                config=_config_to_flat_dict(cfg),
                init_kwargs=init_kwargs,
            )

        # Build evaluator from config if eval datasets are specified
        self.evaluator: Evaluator | None = None
        if cfg.data.eval is not None:
            _status("[dim]Building evaluator...[/dim]")
            from oplm.eval import Evaluator

            self.evaluator = Evaluator(cfg)

        # Model
        _status("[dim]Building model...[/dim]")
        model = OplmForMLM(cfg.model)
        if cfg.model.gradient_checkpointing:
            model.encoder.gradient_checkpointing = True

        # Optimizer and dataloader
        optimizers = build_optimizers(model, cfg.train)
        _status("[dim]Loading training data...[/dim]")
        dataloader = build_train_dataloader(cfg)
        raw_dataset_size = self._get_dataset_size_from_dataloader(dataloader)

        # Compute total_steps
        self.total_steps = self._compute_total_steps(cfg, dataloader)
        schedulers = build_schedulers(optimizers, cfg.train, self.total_steps)

        # Prepare with accelerate
        _status("[dim]Preparing for training...[/dim]")
        prepared = self.accelerator.prepare(model, *optimizers, dataloader, *schedulers)
        num_optimizers = len(optimizers)
        self.model = prepared[0]
        self.optimizers = list(prepared[1 : 1 + num_optimizers])
        self.optimizer = self.optimizers[0]
        self.dataloader = prepared[1 + num_optimizers]
        self.schedulers = list(prepared[2 + num_optimizers :])
        self.scheduler = self.schedulers[0]

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self._samples_seen = 0
        self._last_eval_loss: float | None = None

        # FLOP estimation
        self.flops_per_token = estimate_flops_per_token(cfg.model)

        # Dataset size for fractional epoch computation
        self._dataset_size = raw_dataset_size

        # Resume from checkpoint
        if cfg.train.resume_from is not None:
            _status("[dim]Resuming from checkpoint...[/dim]")
            self._resume_from_checkpoint(cfg.train.resume_from)

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

        self._emit_train_start()
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

                # Forward + backward inside accumulation context
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    if cfg.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            cfg.max_grad_norm,
                        )

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                # Track tokens and samples
                tokens_in_batch = batch["attention_mask"].sum().item()
                self.tokens_seen += int(tokens_in_batch) * self.accelerator.num_processes
                self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes

                # Only act on optimizer steps (accumulation boundary)
                if not self.accelerator.sync_gradients:
                    continue

                for scheduler in self.schedulers:
                    scheduler.step()
                self.global_step += 1
                current_loss = loss.item()

                # Logging
                if self.global_step % cfg.log_every == 0:
                    self._log_step(current_loss)

                # Evaluation
                eval_metrics = self._run_eval()
                if eval_metrics:
                    eval_loss = self._extract_eval_loss(eval_metrics)
                    if eval_loss is not None:
                        self._last_eval_loss = eval_loss
                    self._log_metrics(eval_metrics)
                    self._emit_eval_end(eval_metrics)

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
            self._emit_train_end()
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
            effective_batch = (
                cfg.train.batch_size
                * cfg.train.gradient_accumulation_steps
                * self.accelerator.num_processes
            )
            steps_per_epoch = max(1, math.ceil(dataset_size / effective_batch))
            return int(steps_per_epoch * cfg.train.max_epochs)
        return cfg.train.max_steps

    @staticmethod
    def _get_dataset_size_from_dataloader(dataloader: DataLoader) -> int:  # type: ignore[type-arg]
        """Get the dataset size from the dataloader."""
        dataset = getattr(dataloader, "dataset", None)
        return _resolve_total_length(dataset)

    def _fractional_epoch(self) -> float:
        """Compute the current fractional epoch."""
        if self._dataset_size <= 0:
            return float(self.epoch)
        return self._samples_seen / self._dataset_size

    def _set_dataset_epoch(self, epoch: int) -> None:
        """Propagate epoch to the dataset for deterministic shuffling."""
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)
            return

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
            "train/samples": self._samples_seen,
            "train/tokens": self.tokens_seen,
            "train/flops": cumulative_flops,
            "train/lr": self.scheduler.get_last_lr()[0],
        }
        self._log_metrics(metrics)

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        from oplm.training.checkpoint import save_checkpoint

        checkpoint_dir = Path(self.cfg.train.output_dir) / f"checkpoint-{self.global_step}"
        save_checkpoint(
            accelerator=self.accelerator,
            cfg=self.cfg,
            output_dir=self.cfg.train.output_dir,
            global_step=self.global_step,
            epoch=self.epoch,
            samples_seen=self._samples_seen,
            tokens_seen=self.tokens_seen,
            save_total_limit=self.cfg.train.save_total_limit,
        )
        self._emit_checkpoint_saved(checkpoint_dir)

    def _resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume training state from a checkpoint."""
        from oplm.training.checkpoint import load_checkpoint

        state = load_checkpoint(self.accelerator, checkpoint_dir)
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.tokens_seen = state["tokens_seen"]
        self._samples_seen = int(
            state.get("samples_seen", self.global_step * self._global_effective_batch_size())
        )
        self._set_dataset_epoch(self.epoch)

        logger.info(
            "Resumed from checkpoint %s (step=%d, epoch=%d, samples=%d, tokens=%d)",
            checkpoint_dir,
            self.global_step,
            self.epoch,
            self._samples_seen,
            self.tokens_seen,
        )

    def _global_effective_batch_size(self) -> int:
        """Return the batch size represented by one optimizer step."""
        return int(
            self.cfg.train.batch_size
            * self.cfg.train.gradient_accumulation_steps
            * self.accelerator.num_processes
        )

    @staticmethod
    def _extract_eval_loss(metrics: dict[str, float]) -> float | None:
        """Extract a progress-bar loss from evaluator output."""
        if "eval/loss" in metrics:
            return metrics["eval/loss"]

        losses = [value for key, value in metrics.items() if key.endswith("/loss")]
        if not losses:
            return None
        return sum(losses) / len(losses)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics and notify callbacks."""
        self.accelerator.log(metrics, step=self.global_step)
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_log(self, dict(metrics), self.global_step)

    def _emit_train_start(self) -> None:
        """Notify callbacks that training is starting."""
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_train_start(self)

    def _emit_eval_end(self, metrics: dict[str, float]) -> None:
        """Notify callbacks that evaluation completed."""
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_eval_end(self, dict(metrics), self.global_step)

    def _emit_checkpoint_saved(self, checkpoint_dir: Path) -> None:
        """Notify callbacks that a checkpoint was saved."""
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_checkpoint_saved(self, checkpoint_dir, self.global_step)

    def _emit_train_end(self) -> None:
        """Notify callbacks that training has ended."""
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_train_end(self)


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


def _resolve_total_length(dataset: object) -> int:
    """Resolve the raw dataset length through wrapper layers."""
    if dataset is None:
        return 0

    total_length = getattr(dataset, "total_length", None)
    if total_length is not None:
        return int(total_length)

    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
        child_length = _resolve_total_length(child_dataset)
        if child_length > 0:
            return child_length

    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return 0

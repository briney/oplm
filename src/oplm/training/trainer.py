"""Trainer class for OPLM masked language model pretraining."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig
    from oplm.eval.context import EvalContext
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

        from oplm.data import build_train_dataloader
        from oplm.model import OplmForMaskedLM
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

        # Consolidate all run artifacts (checkpoints, config copy, wandb logs)
        # under output_dir. Create it up front so wandb can log into it.
        if self.accelerator.is_main_process:
            Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)

        # Init wandb early so login prompt appears before slow setup steps
        if cfg.train.wandb_enabled:
            _status("[dim]Initializing wandb...[/dim]")
            # accelerate's WandBTracker ignores project_dir (requires_logging_directory
            # is False), so point wandb's local logs into output_dir explicitly via
            # `dir` -> they land in output_dir/wandb/ instead of ./wandb.
            wandb_kwargs: dict[str, Any] = {"dir": cfg.train.output_dir}
            if cfg.train.wandb_run_name is not None:
                wandb_kwargs["name"] = cfg.train.wandb_run_name
            self.accelerator.init_trackers(
                project_name=cfg.train.wandb_project,
                config=_config_to_flat_dict(cfg),
                init_kwargs={"wandb": wandb_kwargs},
            )

        # Drop a top-level copy of the fully resolved config alongside the run.
        if self.accelerator.is_main_process:
            from oplm.config import serialize_config

            (Path(cfg.train.output_dir) / "config.yaml").write_text(serialize_config(cfg))

        # Build evaluator from config if eval datasets are specified
        self.evaluator: Evaluator | None = None
        if cfg.data.eval is not None:
            _status("[dim]Building evaluator...[/dim]")
            from oplm.eval import Evaluator

            self.evaluator = Evaluator(cfg)

        # Model
        _status("[dim]Building model...[/dim]")
        # Read the flag before constructing the model: transformers strips
        # ``config.gradient_checkpointing`` during ``PreTrainedModel.__init__`` (and
        # auto-enables checkpointing on the model), so reading it afterward raises
        # AttributeError. Re-enabling here is idempotent and keeps the wiring
        # explicit across transformers versions.
        gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
        model = OplmForMaskedLM(cfg.model)  # cfg.model is the HF OplmConfig
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()  # propagates to every OplmBlock

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

        # Apply torch.compile AFTER accelerator.prepare so that accelerate always
        # receives the raw model and wraps it with DDP normally. Compiling the
        # DDP-wrapped model produces OptimizedModule(_orig_mod=DDP(model)), which
        # accelerate.unwrap_model can unambiguously peel. Compiling before prepare
        # causes some accelerate versions to strip the compile wrapper during prepare,
        # leaving self.model = DDP(model) with no _orig_mod — and unwrap_model then
        # fails with KeyError: '_orig_mod' at the first eval call.
        if cfg.train.compile:
            # Selective activation checkpointing (SAC) is incompatible with the
            # default DDPOptimizer: it splits the compiled graph at gradient-bucket
            # boundaries to overlap allreduce with backward, which fragments each
            # block's activation-checkpoint higher-order op across subgraphs. AOT
            # autograd's min-cut partitioner then can't honor the SAC MUST_SAVE set,
            # so `selective` silently collapses to full recompute under DDP+compile
            # (single-GPU is unaffected — DDPOptimizer never engages). Disabling
            # graph-splitting keeps SAC intact; the only cost is the lost comm/compute
            # overlap (a few ms of allreduce on a ~500 ms step over NVLink — negligible
            # next to the recompute SAC saves back). Gate on `selective` only so
            # `full`/`none` keep the default overlap. The flag is a process-global
            # dynamo config, set once before compile, and is numerically transparent.
            if getattr(cfg.model, "gradient_checkpointing_mode", "full") == "selective":
                # `from torch import _dynamo` (not `import torch._dynamo`) so the local
                # binding is `_dynamo`, not `torch` — the latter would shadow the
                # module-level `torch` and break the `torch.compile` call just below.
                from torch import _dynamo

                _dynamo.config.optimize_ddp = False
            # When compile_dynamic is not True (False or None/auto with bucketed static
            # shapes), raise the Dynamo recompile budget so bucketed shapes do not
            # silently fall back to eager. Use the same local `from torch import _dynamo`
            # pattern to avoid shadowing the module-level `torch`.
            if cfg.train.compile_dynamic is not True:
                from torch import _dynamo

                if cfg.data.pad_to_multiple_of is not None:
                    buckets = math.ceil(
                        cfg.model.max_position_embeddings / cfg.data.pad_to_multiple_of
                    )
                    _dynamo.config.cache_size_limit = max(
                        _dynamo.config.cache_size_limit, buckets + 8
                    )
                    logger.info(
                        "compile_dynamic=%s, pad_to_multiple_of=%d: "
                        "expecting ~%d sequence-length buckets; "
                        "raised cache_size_limit to %d",
                        cfg.train.compile_dynamic,
                        cfg.data.pad_to_multiple_of,
                        buckets,
                        _dynamo.config.cache_size_limit,
                    )
                else:
                    logger.warning(
                        "compile_dynamic=%s with pad_to_multiple_of=None: "
                        "batch-max padding produces unbounded shapes under non-dynamic "
                        "compile and will thrash Dynamo recompiles. "
                        "Set pad_to_multiple_of or compile_dynamic=True.",
                        cfg.train.compile_dynamic,
                    )
            _status(
                f"[dim]Compiling model (torch.compile, dynamic={cfg.train.compile_dynamic})...[/dim]"
            )
            # torch.compile stubs return Callable; cast to nn.Module so downstream
            # calls are well-typed — OptimizedModule IS an nn.Module at runtime.
            self.model = cast(
                "nn.Module",
                torch.compile(self.model, dynamic=cfg.train.compile_dynamic, mode=cfg.train.compile_mode),
            )
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
        self._epoch_at_last_opt_step = 0
        self._step_local_tokens = 0  # local tokens accumulated across the current opt step

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
        step_loss_sum = 0.0

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
                        optimizer.zero_grad()

                # Accumulate the per-micro-batch loss so logging reports the mean
                # across the optimizer step, not just the final micro-batch.
                step_loss_sum += loss.detach().item()

                # Track tokens (local; reduced across ranks on the opt-step boundary
                # below) and samples
                self._step_local_tokens += int(batch["attention_mask"].sum().item())
                self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes

                # Only act on optimizer steps (accumulation boundary)
                if not self.accelerator.sync_gradients:
                    continue

                for scheduler in self.schedulers:
                    scheduler.step()
                self.global_step += 1
                current_loss = step_loss_sum / cfg.gradient_accumulation_steps
                step_loss_sum = 0.0

                # Rank-reduce this step's tokens so tokens_seen / tokens_delta are
                # rank-identical (the EvalContext rank-sync invariant; see design §3.2).
                # Unconditional: a per-rank estimate would diverge on ragged batches.
                tokens_tensor = torch.tensor(
                    self._step_local_tokens, device=self.accelerator.device, dtype=torch.long
                )
                tokens_delta = int(self.accelerator.reduce(tokens_tensor, reduction="sum").item())
                self.tokens_seen += tokens_delta
                self._step_local_tokens = 0

                # Logging
                if self.global_step % cfg.log_every == 0:
                    self._log_step(current_loss)

                # Evaluation
                eval_metrics = self._run_eval(tokens_delta)
                if eval_metrics:
                    eval_loss = self._extract_eval_loss(eval_metrics)
                    if eval_loss is not None:
                        self._last_eval_loss = eval_loss
                    self._log_metrics(eval_metrics)
                    self._emit_eval_end(eval_metrics)

                # Checkpointing
                if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
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

            # Final checkpoint — guaranteed unless disabled. Skip when the last
            # step already triggered a periodic save (avoids a redundant re-write).
            last_step_saved = cfg.save_every > 0 and self.global_step % cfg.save_every == 0
            if cfg.save_final and not last_step_saved:
                self._save_checkpoint()

        finally:
            if progress is not None:
                progress.stop()
            self._emit_train_end()
            self.accelerator.end_training()

    def _build_eval_context(self, tokens_delta: int) -> EvalContext:
        """Build a rank-identical EvalContext for the current optimizer step."""
        from oplm.eval.context import EvalContext

        epoch_delta = self.epoch - self._epoch_at_last_opt_step
        self._epoch_at_last_opt_step = self.epoch
        return EvalContext(
            global_step=self.global_step,
            epoch=self.epoch,
            tokens_seen=self.tokens_seen,
            steps_delta=1,
            tokens_delta=tokens_delta,
            epoch_delta=epoch_delta,
            is_final=(self.global_step >= self.total_steps),
        )

    def _run_eval(self, tokens_delta: int) -> dict[str, float]:
        """Run all due evaluations for the current optimizer step.

        Builds a rank-identical :class:`~oplm.eval.context.EvalContext` and delegates
        to :meth:`~oplm.eval.Evaluator.run_due`, which handles per-task scheduling and
        unwraps the (wrapped) model only when a task is due. Returns an empty dict when
        no evaluator is configured or nothing is due.
        """
        if self.evaluator is None:
            return {}
        ctx = self._build_eval_context(tokens_delta)
        return self.evaluator.run_due(ctx, self.model, self.accelerator)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_total_steps(self, cfg: OplmConfig, dataloader: DataLoader) -> int:
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
    def _get_dataset_size_from_dataloader(dataloader: DataLoader) -> int:
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
            model=self.model,
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

        # Reset per-opt-step snapshot markers so the first post-resume step computes
        # correct deltas. tokens_seen is already restored from trainer_state.json; the
        # half-open crossing test neither re-fires at the resumed step nor skips a multiple.
        self._epoch_at_last_opt_step = self.epoch
        self._step_local_tokens = 0

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
    for key, value in cfg.model.to_dict().items():
        flat[f"model/{key}"] = value
    for section_name in ("train", "data"):
        section = asdict(getattr(cfg, section_name))
        for key, value in section.items():
            flat[f"{section_name}/{key}"] = value
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
        return len(dataset)  # ty: ignore[invalid-argument-type]  # guarded by except TypeError
    except TypeError:
        return 0

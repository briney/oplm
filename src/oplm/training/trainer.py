"""Trainer class for OPLM masked language model pretraining.

Native PyTorch distributed stack (no Accelerate): a NCCL/Gloo process group,
FSDP2 (``fully_shard``) for weight/optimizer sharding, optional ``torch.compile``,
and optional torchao FP8. The fixed initialization order is

1. ``dist.init_process_group``
2. build model on CPU
3. [fp8] ``apply_fp8_training(model)``          (before sharding)
4. ``fully_shard`` per block, then the root      (or ``.to(device)`` if unsharded)
5. [compile] ``torch.compile(model)``            (after sharding)
6. ``build_optimizers(model)``                   (after sharding)

Launch via ``torchrun`` (see ``oplm.train``); a bare ``Trainer(cfg)`` also runs
single-process by filling in single-rank defaults.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch.nn as nn
    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig
    from oplm.eval.context import EvalContext
    from oplm.eval.evaluator import Evaluator
    from oplm.training.callbacks import TrainerCallback

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for OPLM with native FSDP2, wandb, and rich progress.

    Args:
        cfg: Full OPLM configuration. If ``data.eval`` is configured, an
            :class:`~oplm.eval.Evaluator` is built automatically.
        callbacks: Optional lifecycle callbacks (fired on rank 0 only).
    """

    def __init__(
        self,
        cfg: OplmConfig,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        from rich.console import Console

        from oplm.data import build_train_dataloader
        from oplm.model import OplmForMaskedLM
        from oplm.training.flops import estimate_flops_per_token
        from oplm.training.optim import build_optimizers, build_schedulers

        self.cfg = cfg
        self.callbacks = list(callbacks or [])

        # 1. Process group + per-rank device.
        self._init_distributed()

        # Seed identically on every rank. Model construction (below) must produce
        # identical weights on all ranks because fully_shard shards each rank's local
        # replica with no cross-rank weight broadcast — a per-rank seed here would
        # create inconsistent shards. Per-rank DATA diversity is handled separately by
        # ShardedProteinDataset's (rank, worker) row striping, not the global RNG seed.
        torch.manual_seed(cfg.train.seed)

        # Status helper for user-facing messages (main process only).
        _console = Console()

        def _status(msg: str) -> None:
            if self.is_main:
                _console.print(msg)

        # wandb (rank 0 only); init early so any login prompt precedes slow setup.
        self._wandb_enabled = bool(cfg.train.wandb_enabled)
        if self.is_main and self._wandb_enabled:
            _status("[dim]Initializing wandb...[/dim]")
            import wandb

            wandb.init(
                project=cfg.train.wandb_project,
                name=cfg.train.wandb_run_name,
                config=_config_to_flat_dict(cfg),
            )

        # Evaluator from config if eval datasets are specified.
        self.evaluator: Evaluator | None = None
        if cfg.data.eval is not None:
            _status("[dim]Building evaluator...[/dim]")
            from oplm.eval import Evaluator

            self.evaluator = Evaluator(cfg)

        # 2. Model on CPU. Read gradient_checkpointing before construction: transformers
        # strips ``config.gradient_checkpointing`` during ``PreTrainedModel.__init__``, so
        # reading it afterward raises. Re-enabling here is idempotent.
        _status("[dim]Building model...[/dim]")
        gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
        model = OplmForMaskedLM(cfg.model)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()  # propagates to every OplmBlock

        # 3. FP8 conversion (BEFORE sharding). Gated by a fail-fast capability check so
        # an unsupported device errors here, not deep inside the first FP8 matmul.
        if cfg.train.precision == "fp8":
            from oplm.training.precision import apply_fp8_training, is_fp8_supported

            if not is_fp8_supported():
                raise RuntimeError(
                    "train.precision='fp8' requires an sm90+ GPU (Blackwell / H100+); "
                    "the current device does not support FP8 matmuls. Use precision='bf16'."
                )
            _status("[dim]Converting Linear layers to FP8 (torchao)...[/dim]")
            apply_fp8_training(model)

        # 4. FSDP2 sharding (or move-to-device for the unsharded debug path).
        _status("[dim]Sharding model (FSDP2)...[/dim]")
        model = self._apply_sharding(model)

        # 5. torch.compile (AFTER sharding). dynamic=True: protein batches have variable
        # sequence lengths, so a static shape would recompile on every new (B, L).
        if cfg.train.compile:
            _status("[dim]Compiling model (torch.compile)...[/dim]")
            # OptimizedModule IS an nn.Module at runtime; cast keeps callers well-typed.
            self.model = cast(
                "nn.Module",
                torch.compile(model, dynamic=True, mode=cfg.train.compile_mode),
            )
        else:
            self.model = model

        # 6. Optimizers (AFTER sharding) — built from the sharded ``model`` so they hold
        # the FSDP2-managed DTensor params. AdamW and Muon both handle DTensors natively.
        optimizers = build_optimizers(model, cfg.train)

        # Dataloader: no accelerate.prepare — ShardedProteinDataset stripes rows by
        # ``(rank, worker)`` off the live process group, so batches are already disjoint.
        _status("[dim]Loading training data...[/dim]")
        self.dataloader = build_train_dataloader(cfg)
        raw_dataset_size = self._get_dataset_size_from_dataloader(self.dataloader)

        self.total_steps = self._compute_total_steps(cfg, self.dataloader)
        schedulers = build_schedulers(optimizers, cfg.train, self.total_steps)

        self.optimizers = list(optimizers)
        self.optimizer = self.optimizers[0]
        self.schedulers = list(schedulers)
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

    # ------------------------------------------------------------------
    # Distributed / sharding setup
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Initialize the process group and resolve this rank's device.

        Fills in single-rank defaults for the ``torchrun`` env vars so a bare
        ``Trainer(cfg)`` (no launcher) still forms a one-rank group; under a real
        ``torchrun`` launch every var is already set, so ``setdefault`` is a no-op.
        Uses NCCL on CUDA and Gloo on CPU (the latter only for local/debug runs).
        """
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        if not dist.is_initialized():
            # Pin the collective device up front (NCCL): lets barrier()/all_reduce pick
            # the right device without guessing and silences the device-context warning.
            backend = "nccl" if use_cuda else "gloo"
            init_kwargs: dict[str, Any] = {"backend": backend}
            if use_cuda:
                init_kwargs["device_id"] = self.device
            dist.init_process_group(**init_kwargs)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main = self.rank == 0

    def _apply_sharding(self, model: nn.Module) -> nn.Module:
        """Apply the configured FSDP2 sharding strategy and return the model.

        ``"full"`` shards every transformer block independently then the root
        (embedding, final norm, MLM head), keeping all-gather granularity manageable.
        ``"none"`` skips sharding entirely and just moves the model to the device
        (single-GPU / CPU debugging). ``"hybrid"`` is a documented follow-on.
        """
        strategy = self.cfg.train.fsdp_sharding_strategy
        if strategy == "none":
            return model.to(self.device)
        if strategy == "hybrid":
            raise NotImplementedError(
                "fsdp_sharding_strategy='hybrid' is not yet implemented; "
                "use 'full' (sharded) or 'none' (single-GPU/debug)."
            )

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("dp",))
        # BF16 params for compute, FP32 gradient reduction for stability. With FP8 the
        # high-precision master weights stay BF16; FP8 is applied at matmul time.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs: dict[str, Any] = {"mesh": mesh, "mp_policy": mp_policy}

        # Shard each transformer block independently, then the root (embedding,
        # final norm, MLM head). The block list lives at oplm.backbone.layers; cast
        # because the param is typed nn.Module (the model is an OplmForMaskedLM here).
        blocks = cast(
            "Sequence[nn.Module]",
            model.oplm.backbone.layers,  # ty: ignore[unresolved-attribute]
        )
        for block in blocks:
            fully_shard(block, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        return model

    def _set_grad_sync(self, enabled: bool) -> None:
        """Toggle FSDP2 gradient reduce-scatter (no-op when the model is unsharded).

        On a ``fully_shard``-ed module, ``set_requires_gradient_sync(False)`` makes
        grads accumulate locally without a reduce-scatter — the FSDP2 analogue of
        DDP's ``no_sync()``. The unsharded ("none") path is a plain module with no
        such method, so this is a no-op there. Resolves through a ``torch.compile``
        wrapper via attribute delegation.
        """
        set_sync = getattr(self.model, "set_requires_gradient_sync", None)
        if set_sync is not None:
            set_sync(enabled)

    @property
    def _checkpoint_model(self) -> nn.Module:
        """The FSDP2 model with any ``torch.compile`` wrapper peeled.

        DCP reads/writes the sharded state dict from this module; the optimizer was
        built over the same (pre-compile) parameters, so model and optimizer state
        stay consistent.
        """
        return getattr(self.model, "_orig_mod", self.model)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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
        if self.is_main:
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
        micro_step = 0  # micro-batches accumulated toward the current optimizer step

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

                is_last_micro_step = (micro_step + 1) % cfg.gradient_accumulation_steps == 0
                micro_step += 1

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Skip the FSDP2 reduce-scatter until the final micro-step so grads
                # accumulate locally; sync on the last micro-step. Loss is divided by
                # the accumulation depth so the gradient magnitude is depth-invariant.
                self._set_grad_sync(is_last_micro_step)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
                (loss / cfg.gradient_accumulation_steps).backward()

                # Accumulate the unscaled per-micro-batch loss so logging reports the
                # mean across the optimizer step, not just the final micro-batch.
                step_loss_sum += loss.detach().item()
                self._step_local_tokens += int(attention_mask.sum().item())
                self._samples_seen += len(batch["input_ids"]) * self.world_size

                # Only act on optimizer steps (accumulation boundary)
                if not is_last_micro_step:
                    continue

                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                # FP8 (dynamic rowwise): precompute next-iter weight scales AFTER the
                # step (weights must be updated first). No-op without FSDP2-sharded
                # Float8Linear weights, so it is safe whenever precision == "fp8".
                if cfg.precision == "fp8":
                    from oplm.training.precision import sync_fp8_history

                    sync_fp8_history(self.model)

                for scheduler in self.schedulers:
                    scheduler.step()

                self.global_step += 1
                micro_step = 0
                current_loss = step_loss_sum / cfg.gradient_accumulation_steps
                step_loss_sum = 0.0

                # Rank-reduce this step's tokens so tokens_seen / tokens_delta are
                # rank-identical (the EvalContext rank-sync invariant; see design §3.2).
                tokens_tensor = torch.tensor(
                    self._step_local_tokens, device=self.device, dtype=torch.long
                )
                dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
                tokens_delta = int(tokens_tensor.item())
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
            self._teardown()

    def _teardown(self) -> None:
        """Finalize wandb (rank 0) and destroy the process group."""
        if self._wandb_enabled and self.is_main:
            import wandb

            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

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

        Builds a rank-identical :class:`~oplm.eval.context.EvalContext` and a
        :class:`~oplm.eval.context.DistContext`, then delegates to
        :meth:`~oplm.eval.Evaluator.run_due`, which handles per-task scheduling and
        peels the (possibly compiled) model only when a task is due. Returns an empty
        dict when no evaluator is configured or nothing is due.
        """
        if self.evaluator is None:
            return {}
        from oplm.eval.context import DistContext

        ctx = self._build_eval_context(tokens_delta)
        dist_ctx = DistContext(
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            is_main=self.is_main,
        )
        return self.evaluator.run_due(ctx, self.model, dist_ctx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_total_steps(self, cfg: OplmConfig, dataloader: DataLoader) -> int:
        """Compute total training steps from config."""
        if cfg.train.max_epochs is not None:
            dataset_size = self._get_dataset_size_from_dataloader(dataloader)
            effective_batch = (
                cfg.train.batch_size * cfg.train.gradient_accumulation_steps * self.world_size
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
        """Propagate epoch to the dataloader or its dataset for deterministic shuffling."""
        loader_set_epoch = getattr(self.dataloader, "set_epoch", None)
        if callable(loader_set_epoch):
            loader_set_epoch(epoch)
            return

        dataset_set_epoch = getattr(self.dataloader.dataset, "set_epoch", None)
        if callable(dataset_set_epoch):
            dataset_set_epoch(epoch)

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
            "train/lr": float(self.scheduler.get_last_lr()[0]),
        }
        self._log_metrics(metrics)

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint via PyTorch Distributed Checkpoint."""
        from oplm.training.checkpoint import save_checkpoint

        checkpoint_dir = Path(self.cfg.train.output_dir) / f"checkpoint-{self.global_step}"
        save_checkpoint(
            model=self._checkpoint_model,
            optimizer=self.optimizer,
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

        state = load_checkpoint(self._checkpoint_model, self.optimizer, checkpoint_dir)
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
            self.cfg.train.batch_size * self.cfg.train.gradient_accumulation_steps * self.world_size
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
        """Log metrics to wandb (rank 0) and notify callbacks."""
        if self.is_main and self._wandb_enabled:
            import wandb

            wandb.log(metrics, step=self.global_step)
        if not self.is_main:
            return

        for callback in self.callbacks:
            callback.on_log(self, dict(metrics), self.global_step)

    def _emit_train_start(self) -> None:
        """Notify callbacks that training is starting."""
        if not self.is_main:
            return

        for callback in self.callbacks:
            callback.on_train_start(self)

    def _emit_eval_end(self, metrics: dict[str, float]) -> None:
        """Notify callbacks that evaluation completed."""
        if not self.is_main:
            return

        for callback in self.callbacks:
            callback.on_eval_end(self, dict(metrics), self.global_step)

    def _emit_checkpoint_saved(self, checkpoint_dir: Path) -> None:
        """Notify callbacks that a checkpoint was saved."""
        if not self.is_main:
            return

        for callback in self.callbacks:
            callback.on_checkpoint_saved(self, checkpoint_dir, self.global_step)

    def _emit_train_end(self) -> None:
        """Notify callbacks that training has ended."""
        if not self.is_main:
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

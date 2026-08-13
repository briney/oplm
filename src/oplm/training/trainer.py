"""Trainer class for OPLM masked language model pretraining."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from oplm.training.signals import DRAIN_EXIT_CODE, DrainSignal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig
    from oplm.eval.context import EvalContext
    from oplm.eval.evaluator import Evaluator
    from oplm.training.callbacks import TrainerCallback
    from oplm.training.checkpoint import PendingSave

logger = logging.getLogger(__name__)

# Wall-clock margin (seconds) before SLURM_JOB_END_TIME at which the env drain
# clock starts reporting requested=True. Fixed module constant for now; may
# become a config knob later if a use case needs it.
_DRAIN_MARGIN_SECONDS = 600

# Loss-spike warn-only detector (Task 1.8, spec §6): same 0.98/0.02 EMA smoothing
# oplm.training.mup's logging helper uses. Warns (never aborts) once the EMA has had
# _LOSS_SPIKE_WARMUP_LOGS logged steps to settle, so a cold-start EMA (which starts
# equal to the very first loss) can't trivially trip the 3x threshold.
_LOSS_EMA_DECAY = 0.98
_LOSS_SPIKE_MULTIPLIER = 3.0
_LOSS_SPIKE_WARMUP_LOGS = 50


@dataclass(frozen=True)
class _StepFlags:
    """Rank-synchronized control bundle for the just-completed optimizer step.

    Every rank computes its own local inputs (raw token count, whether it
    observed a drain signal, whether its own loss was non-finite, whether its
    own ``save_every_minutes`` timer fired, whether its own pending async save's
    local write is done) and :meth:`Trainer._reduce_step_flags` sum-reduces them
    into one rank-identical bundle: token counts add up normally, and any boolean
    flag true on *any* rank becomes true on *every* rank (sum > 0). This is the
    single rank-synchronization point later resilience features plug into — the
    drain signal (Task 1.5) and the non-finite-loss guardrail (Task 1.8) only
    need to change their local input to this reduce, not add a new collective.

    ``pending_done_count`` is deliberately kept as the raw sum rather than
    booleanized like the others: an async checkpoint save (Task 2.3) may only be
    safely committed once *every* rank's local write is done, so the trainer
    needs to compare this count against ``accelerator.num_processes`` exactly --
    a partial count (some but not all ranks done) must never look like "true" the
    way the other flags' any-rank-trips-it-for-everyone semantics intend.
    """

    tokens_delta: int
    drain: bool
    nonfinite: bool
    save_due: bool
    pending_done_count: int


@dataclass(frozen=True)
class _PendingAsyncSave:
    """Trainer-local wrapper around a checkpoint.py ``PendingSave`` handle.

    ``checkpoint.py``'s :func:`~oplm.training.checkpoint.save_checkpoint` /
    :func:`~oplm.training.checkpoint.finalize_pending_save` know nothing about the
    ``keep_every_n_hours`` permanent-marker rule (Task 1.3) -- that decision is made
    entirely on the trainer side, from wall-clock state ``save_checkpoint`` never sees.
    This wrapper carries the decision alongside the handle: ``should_mark_permanent``
    is computed at TRIGGER time in :meth:`Trainer._save_checkpoint` (identically to the
    synchronous path, which also decides it before saving), and applied at FINALIZE
    time in :meth:`Trainer._finalize_pending_save`, once the checkpoint has actually
    been renamed into its final, markable location.

    ``global_step`` is also captured at TRIGGER time -- by the time
    :meth:`Trainer._finalize_pending_save` runs, ``self.global_step`` may already have
    advanced past the step this checkpoint was actually saved at (that is the entire
    point of deferring the commit), so ``on_checkpoint_saved`` must be told the
    checkpoint's own step explicitly rather than reading the trainer's live counter.
    """

    handle: PendingSave
    should_mark_permanent: bool
    global_step: int


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
        from accelerate.utils import DataLoaderConfiguration, InitProcessGroupKwargs, set_seed
        from rich.console import Console

        from oplm.data import DeviceDataLoader, build_train_dataloader
        from oplm.model import OplmForMaskedLM
        from oplm.training.flops import estimate_flops_per_token
        from oplm.training.optim import build_optimizers, build_schedulers
        from oplm.training.preflight import run_preflight

        self.cfg = cfg
        self.callbacks = list(callbacks or [])

        # Drain trigger (Task 1.5): SIGUSR1/SIGTERM handlers plus the Slurm
        # SLURM_JOB_END_TIME wall-clock margin. Installed immediately so a signal
        # arriving during the (potentially slow) rest of __init__ is still caught.
        self._drain_signal = DrainSignal(margin_seconds=_DRAIN_MARGIN_SECONDS)
        self._drain_signal.install()

        # Latest pre-clip gradient norm, captured in the training loop when
        # clipping is active; read by StabilityDiagnosticsCallback. Stays None
        # when max_grad_norm <= 0 (no clip) so the diagnostic simply omits it.
        self._last_grad_norm: torch.Tensor | None = None

        # Opt-in deep-model stability diagnostics (docs/LR_SWEEP.md). Attached
        # here so plain `oplm train ... train.stability_diagnostics=true` runs
        # (the probe/control) get it without a bespoke entry point.
        if cfg.train.stability_diagnostics:
            from oplm.training.mup import StabilityDiagnosticsCallback

            self.callbacks.append(
                StabilityDiagnosticsCallback(probe_every=cfg.train.stability_probe_every)
            )

        # Seed everything
        set_seed(cfg.train.seed)

        # Accelerator. kwargs_handlers=[InitProcessGroupKwargs(timeout=...)] (Task 1.8)
        # bounds every NCCL/gloo collective's wait, converting a genuine hang into a
        # raised exception (-> nonzero exit -> requeue) within cfg.train.dist_timeout_minutes
        # instead of wedging until the Slurm time limit; on a single process it is a harmless
        # no-op (there is no process group to time out).
        log_with = "wandb" if cfg.train.wandb_enabled else None
        self.accelerator = Accelerator(
            mixed_precision=cfg.train.mixed_precision,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            log_with=log_with,
            project_dir=cfg.train.output_dir,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(minutes=cfg.train.dist_timeout_minutes))
            ],
        )

        # Preflight (Task 1.8): allocate + matmul on every rank, right after the
        # Accelerator exists and BEFORE anything below touches checkpoints or data. A
        # sick node must fail here, in seconds and attributably, rather than hanging
        # mid-training or corrupting a resume by racing the stale-checkpoint cleanup
        # below. Unconditional on every rank -- like the resume-target broadcast further
        # down, run_preflight contains a collective (a gather_object exchange of every
        # rank's local pass/fail, when num_processes > 1) that every rank must reach
        # together, REGARDLESS of its own local outcome, or a locally-failing rank would
        # leave the healthy ranks hanging in it instead of failing fast and attributably.
        run_preflight(self.accelerator)

        # Status helper for user-facing messages (main process only)
        _console = Console()

        def _status(msg: str) -> None:
            if self.accelerator.is_main_process:
                _console.print(msg)

        # Consolidate all run artifacts (checkpoints, config copy, wandb logs)
        # under output_dir. Create it up front so wandb can log into it.
        if self.accelerator.is_main_process:
            Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)

            # A checkpoint-<step>.tmp/ staging dir surviving to trainer start means the
            # process that created it was killed mid-save (torn, unconditionally deleted).
            # A checkpoint-<step>.old/ dir means it was killed mid-replace (the previous
            # commit at that step, moved aside before the new one landed) — recovered back
            # onto checkpoint-<step>/ if the new commit never landed. Resolve both before
            # any resume logic below can consider the directory listing.
            from oplm.training.checkpoint import clean_stale_checkpoint_dirs

            clean_stale_checkpoint_dirs(Path(cfg.train.output_dir))

        # Barrier: no rank proceeds to the resume-target resolution below until the main
        # process's clean_stale_checkpoint_dirs recovery renames (checkpoint-<N>.old ->
        # checkpoint-<N>) have landed. Without it, a rank that reaches the auto_resume scan
        # before the main process finishes an interrupted same-step recovery could see a
        # directory listing the main process itself never would have. A no-op on a single
        # process.
        self.accelerator.wait_for_everyone()

        # Resolve the resume target *before* wandb init (Task 1.7) so a resumed run's id can
        # be threaded into wandb_kwargs below. _resolve_resume_target scans for the newest
        # *committed* checkpoint (oplm.training.checkpoint.latest_checkpoint) ON THE MAIN
        # PROCESS ONLY and broadcasts the result to every other rank, rather than letting
        # every rank scan the directory itself: the barrier above only orders the main
        # process's recovery renames ahead of the scan, it does not guarantee every rank's
        # own directory listing agrees with the main process's -- multi-node shared
        # filesystems (NFS/Lustre) commonly serve a stale/cached listing to a non-writer
        # rank for a window after a writer's rename lands. Scanning once and broadcasting
        # makes the resume target rank-identical BY CONSTRUCTION, independent of filesystem
        # visibility semantics -- the actual property this branch exists to provide. A fresh
        # output_dir with no committed checkpoint yet is a no-op: training starts at step 0,
        # exactly as if auto_resume were unset. The heavy state restore
        # (_resume_from_checkpoint) happens later, once the model/optimizer/dataloader exist.
        resume_target = _resolve_resume_target(
            self.accelerator,
            cfg.train.resume_from,
            cfg.train.auto_resume,
            cfg.train.output_dir,
            cfg,
            status=_status,
        )
        self._resolved_resume_target = resume_target  # exposed for tests/observability

        # Run id persisted right after init_trackers below; reused as the wandb_run_id
        # extra_state key on every checkpoint save. Stays None when wandb is disabled or
        # (main-process-only) on non-main ranks, so save_checkpoint's extra_state omits it.
        self._wandb_run_id: str | None = None

        # Init wandb early so login prompt appears before slow setup steps
        if cfg.train.wandb_enabled:
            _status("[dim]Initializing wandb...[/dim]")
            # accelerate's WandBTracker ignores project_dir (requires_logging_directory
            # is False), so point wandb's local logs into output_dir explicitly via
            # `dir` -> they land in output_dir/wandb/ instead of ./wandb.
            wandb_kwargs: dict[str, Any] = {"dir": cfg.train.output_dir}
            if cfg.train.wandb_run_name is not None:
                wandb_kwargs["name"] = cfg.train.wandb_run_name
            if resume_target is not None:
                # Task 1.7: continue the same W&B run across a requeue instead of
                # starting a new one. Checkpoint's trainer_state.json is the first
                # choice (authoritative for that checkpoint); the output_dir marker
                # file is the fallback (e.g. a checkpoint saved before this field
                # existed, or a save that landed between the id write and the next
                # checkpoint).
                run_id = _read_resume_wandb_run_id(resume_target, cfg.train.output_dir)
                if run_id is not None:
                    wandb_kwargs |= {"id": run_id, "resume": "allow"}
            self.accelerator.init_trackers(
                project_name=cfg.train.wandb_project,
                config=_config_to_flat_dict(cfg),
                init_kwargs={"wandb": wandb_kwargs},
            )

            # Persist the run id immediately so a mid-run kill still leaves a
            # resumable marker. Lazy/guarded import: wandb is an optional dependency,
            # only needed when wandb_enabled. `wandb.run` is set by init_trackers just
            # above (WandBTracker.start() calls wandb.init()); it is main-process-only
            # (WandBTracker methods are @on_main_process) and can stay None in some
            # offline edge cases, which is skipped gracefully.
            if self.accelerator.is_main_process:
                import wandb

                active_run = wandb.run
                if active_run is not None:
                    self._wandb_run_id = active_run.id
                    (Path(cfg.train.output_dir) / "wandb_run_id").write_text(
                        f"{self._wandb_run_id}\n"
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

        # Prepare with accelerate. The dataloader is deliberately excluded: it is
        # built from ShardedProteinDataset, which already stripes rows over the
        # joint (rank, worker) index. Passing it through accelerator.prepare would
        # wrap it in accelerate's own IterableDatasetShard, striping a second time
        # and silently dropping rows (confirmed in
        # .superpowers/sdd/TODOS/task-0.1-report.md). DeviceDataLoader below
        # supplies the only two things prepare was providing for the dataloader:
        # device placement and set_epoch forwarding.
        #
        # Gradient-accumulation caveat: without a prepared dataloader, accelerate
        # cannot detect end-of-dataloader for a partial final accumulation window.
        # The training stream is effectively infinite (epoch rollover re-iterates
        # via _set_dataset_epoch below), so no partial window ever needs to sync —
        # identical semantics to before for gradient_accumulation_steps == 1, and
        # for >1 the tail micro-batches of an epoch simply roll into the next
        # window instead of forcing an early sync.
        _status("[dim]Preparing for training...[/dim]")
        prepared = self.accelerator.prepare(model, *optimizers, *schedulers)
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
                elif cfg.train.compile_dynamic is False:
                    logger.warning(
                        "compile_dynamic=False with pad_to_multiple_of=None: "
                        "batch-max padding produces unbounded shapes under static "
                        "compile and will thrash Dynamo recompiles. "
                        "Set pad_to_multiple_of or compile_dynamic=True.",
                    )
            # torch 2.10/2.11 enable Inductor's mix-order-reduction pass by
            # default. It produces FusedMixOrderReductions scheduler nodes that
            # crash the combo-kernel fusion path during backward compilation:
            #   scheduler.py: assert not isinstance(node1, FusedMixOrderReductions)
            # (pytorch/pytorch#169811). The crash is build/cache-sensitive — it
            # surfaces on a cold Inductor cache (e.g. ephemeral SUNK/CoreWeave
            # pods) but is masked by a warm cache on-prem. Disable the pass; it's
            # a fusion-throughput optimization, not a correctness feature. Set
            # unconditionally (the bug is independent of SAC mode and
            # compile_mode) and guarded so torch versions lacking the flag won't
            # raise (the inductor config module raises AttributeError on unknown
            # attributes). Remove once #169811 is fixed and the torch floor is
            # raised past the fix.
            from torch._inductor import config as _inductor_config

            if hasattr(_inductor_config.triton, "mix_order_reduction"):
                _inductor_config.triton.mix_order_reduction = False
            _status(
                f"[dim]Compiling model (torch.compile, dynamic={cfg.train.compile_dynamic})"
                "...[/dim]"
            )
            # torch.compile stubs return Callable; cast to nn.Module so downstream
            # calls are well-typed — OptimizedModule IS an nn.Module at runtime.
            self.model = cast(
                "nn.Module",
                torch.compile(
                    self.model, dynamic=cfg.train.compile_dynamic, mode=cfg.train.compile_mode
                ),
            )
        self.optimizers = list(prepared[1 : 1 + num_optimizers])
        self.optimizer = self.optimizers[0]
        self.dataloader = DeviceDataLoader(dataloader, self.accelerator.device)
        self.schedulers = list(prepared[1 + num_optimizers :])
        self.scheduler = self.schedulers[0]

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self._samples_seen = 0
        self._last_eval_loss: float | None = None
        self._epoch_at_last_opt_step = 0
        self._step_local_tokens = 0  # local tokens accumulated across the current opt step

        # Loss-spike warn-only detector state (Task 1.8). Not persisted across a
        # requeue/resume: it is a diagnostic-only trend, not training state, and simply
        # re-warms from the resumed run's own logged losses.
        self._loss_ema: float | None = None
        self._loss_log_count = 0

        # Wall-clock/time-based checkpoint cadence state (Task 1.3). _last_save_at
        # uses time.monotonic() and is never persisted (meaningless across process
        # restarts); _first_checkpoint_at/_last_time_keep_index anchor the
        # keep_every_n_hours rule and ARE persisted (trainer_state.json) so a
        # requeue doesn't reset the hours anchor.
        self._last_save_at = time.monotonic()
        self._first_checkpoint_at: float | None = None
        self._last_time_keep_index = 0

        # At most one outstanding async periodic save (Task 2.3): set by
        # _save_checkpoint(blocking=False), cleared by _finalize_pending_save once its
        # deferred commit tail (barrier + rename + pointer + rotation) has run. Never
        # persisted across a resume -- a Future can't survive a process restart, and a
        # kill while this is set intentionally leaves only a .tmp dir (see
        # clean_stale_checkpoint_dirs), never a resume candidate.
        self._pending_save: _PendingAsyncSave | None = None

        # FLOP estimation
        self.flops_per_token = estimate_flops_per_token(cfg.model)

        # Throughput timing state (steady-state window; warmup steps excluded)
        self._step_timer_start: float | None = None
        self._tput_window_tokens = 0
        self._tput_window_seconds = 0.0
        self._tput_window_steps = 0

        # Dataset size for fractional epoch computation
        self._dataset_size = raw_dataset_size

        # Resume from checkpoint. resume_target was already resolved above (ahead of wandb
        # init, so a resumed run can reuse its wandb id); only the heavy state restore
        # happens here, now that the model/optimizer/dataloader exist.
        if resume_target is not None:
            self._resume_from_checkpoint(resume_target)

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
        last_step_did_save = False

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
                        # Retain the pre-clip total norm so StabilityDiagnosticsCallback
                        # can log it (grads are zeroed before the log step fires).
                        self._last_grad_norm = self.accelerator.clip_grad_norm_(
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

                # Rank-sync the control bundle for this optimizer step in a single
                # reduce: tokens (so tokens_seen / tokens_delta are rank-identical —
                # the EvalContext rank-sync invariant; see design §3.2) plus the
                # drain, non-finite, and time-based-save flags later resilience
                # tasks plug into. Unconditional: a per-rank token estimate would
                # diverge on ragged batches.
                local_tokens = self._step_local_tokens
                self._step_local_tokens = 0
                flags = self._reduce_step_flags(
                    local_tokens,
                    drain=self._drain_signal.requested,
                    nonfinite=not math.isfinite(current_loss),
                    save_due=self._save_timer_due(),
                    pending_done=(
                        self._pending_save is not None and self._pending_save.handle.future.done()
                    ),
                )
                tokens_delta = flags.tokens_delta
                self.tokens_seen += tokens_delta

                # Opportunistically commit a pending async save (Task 2.3) once every
                # rank's local write has actually finished, not merely started --
                # pending_done_count is the raw sum, so this only fires once every rank
                # agrees (== num_processes), never on a partial-done count. A cheap no-op
                # guard when there is nothing pending.
                if (
                    self._pending_save is not None
                    and flags.pending_done_count == self.accelerator.num_processes
                ):
                    self._finalize_pending_save()

                # Non-finite loss guard (Task 1.8, spec §6): a NaN/inf loss on ANY rank trips
                # this flag on EVERY rank (it is post-reduce), so all ranks raise together
                # here -- after this step's tokens_seen accounting is complete (consistent
                # with where the drain branch sits below), but before eval/checkpoint can act
                # on a poisoned step. The requeue loop then resumes from the last checkpoint,
                # an automatic rollback; two such restarts with no step advance trips the
                # no-progress crash-loop guard instead of requeueing forever. A still-pending
                # async save is force-finalized first (blocks on its future) so the process
                # doesn't raise/exit with a background write in flight (Task 2.3): the commit
                # protocol invariants demand that, absent this, a torn write left behind would
                # be a kill mid-async -- fine on a true kill, but not on a controlled raise.
                if flags.nonfinite:
                    if self._pending_save is not None:
                        self._finalize_pending_save()
                    logger.error(
                        "non-finite training loss %s at step %d", current_loss, self.global_step
                    )
                    raise RuntimeError(f"non-finite training loss at step {self.global_step}")

                # Accumulate throughput window, excluding warmup steps
                now = time.perf_counter()
                if (
                    self._step_timer_start is not None
                    and self.global_step > cfg.throughput_warmup_steps
                ):
                    self._tput_window_seconds += now - self._step_timer_start
                    self._tput_window_tokens += tokens_delta
                    self._tput_window_steps += 1

                # Drain takes priority over logging/eval/periodic-save: this step's
                # bookkeeping (tokens_seen, throughput window) is already complete above,
                # so the checkpoint below carries tokens_seen consistent with global_step
                # across a drain+resume -- but we still check before eval so a drain never
                # starts a long eval. A pending async save is finalized (blocking) FIRST
                # (Task 2.3): the drain's own save must always be blocking, and must never
                # be preceded by a torn/uncommitted async save left dangling behind it.
                if flags.drain:
                    if self._pending_save is not None:
                        self._finalize_pending_save()
                    self._save_checkpoint()
                    logger.warning(
                        "Drain requested: checkpoint saved at step %d; exiting %d",
                        self.global_step,
                        DRAIN_EXIT_CODE,
                    )
                    raise SystemExit(DRAIN_EXIT_CODE)

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

                # Checkpointing: step cadence OR the rank-synced save_every_minutes
                # timer (flags.save_due is identical on every rank after the reduce
                # above, so this decision is too). Periodic saves default to async
                # (Task 2.3): a prior pending save that hasn't been finalized yet (the
                # opportunistic gate above didn't catch it this step) is force-finalized
                # first -- at most one outstanding async save at a time -- then this
                # save is triggered without blocking on its own write.
                step_saved = cfg.save_every > 0 and self.global_step % cfg.save_every == 0
                last_step_did_save = step_saved or flags.save_due
                if last_step_did_save:
                    if self._pending_save is not None:
                        self._finalize_pending_save()
                    self._save_checkpoint(blocking=False)

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

                # Restart the step timer AFTER eval/checkpoint so their wall time
                # is excluded from the next step's measurement.
                self._step_timer_start = time.perf_counter()

            # A still-outstanding async save must be finalized before the run ends
            # normally (Task 2.3) -- otherwise its background dcp.async_save write
            # could still be in flight when the process exits, and its commit rename
            # would never happen at all. This does not apply to the drain/non-finite
            # exit paths above: those raise/exit through the `finally` below without
            # reaching this point, by design (their own pending-save handling already
            # ran, and a genuine kill is meant to leave nothing but a `.tmp`).
            if self._pending_save is not None:
                self._finalize_pending_save()

            # Final checkpoint — guaranteed unless disabled. Skip when the last
            # optimizer step already triggered a save, for either reason (step
            # cadence or the save_every_minutes timer), to avoid a redundant
            # re-write.
            if cfg.save_final and not last_step_did_save:
                self._save_checkpoint()

        finally:
            # Deliberately NOT finalizing self._pending_save here. This finally runs on
            # every exit from the try above, including an unrelated crash (an exception
            # this method didn't raise itself -- e.g. an OOM, a NCCL/dist error, a
            # keyboard interrupt) with a pending async save still outstanding. Forcing a
            # finalize (which blocks on the future, then renames tmp_dir onto the final
            # name) on that path would be wrong: it would let a checkpoint commit AFTER
            # the crash that was supposed to kill the run, on a filesystem state that may
            # not reflect what every rank actually finished writing (a crash on one rank
            # doesn't guarantee every other rank's write, or its own barrier, resolves
            # cleanly). Every DELIBERATE stop (drain, the non-finite guard, a new
            # save trigger, or a normal end-of-training) already force-finalizes at its
            # own call site, above, before reaching here -- so by the time control
            # reaches this finally through one of those paths, self._pending_save is
            # already None. An unrelated crash instead leaves self._pending_save's
            # tmp_dir exactly where clean_stale_checkpoint_dirs expects a torn,
            # never-committed staging dir: invisible to discovery/rotation, and deleted
            # unconditionally the next time a Trainer starts against this output_dir.
            # This is the commit protocol's core invariant working as designed, not a
            # gap: "a kill mid-async leaves only a .tmp" is supposed to include kills
            # that happen to land while an async save is pending.
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

    def _reduce_step_flags(
        self,
        local_tokens: int,
        *,
        drain: bool = False,
        nonfinite: bool = False,
        save_due: bool = False,
        pending_done: bool = False,
    ) -> _StepFlags:
        """Rank-sync the control bundle for this optimizer step in one reduce.

        Packs ``[local_tokens, drain, nonfinite, save_due, pending_done]`` into a
        single 5-element long tensor and sum-reduces it across ranks: the token count
        accumulates normally, ``drain``/``nonfinite``/``save_due`` each become True
        everywhere if True on *any* rank (sum > 0), and ``pending_done`` is instead
        returned as the raw sum (see :class:`_StepFlags`'s docstring for why it alone
        needs the exact count rather than any-rank-trips-it semantics). Per-rank clock
        or signal skew on the boolean inputs is harmless by construction — one rank
        tripping a flag trips it for all ranks.

        Args:
            local_tokens: This rank's token count for the just-completed optimizer
                step.
            drain: This rank's local drain signal (SIGUSR1/SIGTERM/SLURM end-time
                margin), from :attr:`_drain_signal` (Task 1.5).
            nonfinite: This rank's local non-finite-loss signal, i.e.
                ``not math.isfinite(current_loss)`` for this rank's just-computed loss
                (Task 1.8).
            save_due: This rank's local ``save_every_minutes`` timer signal.
            pending_done: This rank's local signal that its pending async save's
                write has finished (Task 2.3): ``self._pending_save is not None and
                self._pending_save.handle.future.done()``.

        Returns:
            A rank-identical :class:`_StepFlags`.
        """
        bundle = torch.tensor(
            [int(local_tokens), int(drain), int(nonfinite), int(save_due), int(pending_done)],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        reduced = self.accelerator.reduce(bundle, reduction="sum").tolist()
        tokens_delta, drain_sum, nonfinite_sum, save_due_sum, pending_done_sum = (
            int(x) for x in reduced
        )
        return _StepFlags(
            tokens_delta=tokens_delta,
            drain=drain_sum > 0,
            nonfinite=nonfinite_sum > 0,
            save_due=save_due_sum > 0,
            pending_done_count=pending_done_sum,
        )

    def _save_timer_due(self) -> bool:
        """Return True if this rank's ``save_every_minutes`` wall-clock timer fired.

        Uses ``time.monotonic()`` (never persisted — meaningless across process
        restarts); ``self._last_save_at`` is reset every time a checkpoint is
        actually saved, in :meth:`_save_checkpoint`.
        """
        save_every_minutes = self.cfg.train.save_every_minutes
        if save_every_minutes is None:
            return False
        return (time.monotonic() - self._last_save_at) >= save_every_minutes * 60

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

        # Steady-state throughput (window resets after each log emission).
        # Note: flops_per_token omits attention-score FLOPs, so achieved_tflops/mfu
        # undercount padding waste — tokens_per_sec is the headline metric.
        if self._tput_window_steps > 0 and self._tput_window_seconds > 0:
            tokens_per_sec = self._tput_window_tokens / self._tput_window_seconds
            step_time_s = self._tput_window_seconds / self._tput_window_steps
            achieved_tflops = (
                self.flops_per_token * self._tput_window_tokens / self._tput_window_seconds / 1e12
            )
            metrics["train/tokens_per_sec"] = tokens_per_sec
            metrics["train/step_time_s"] = step_time_s
            metrics["train/achieved_tflops"] = achieved_tflops
            if self.cfg.train.peak_tflops:
                metrics["train/mfu"] = achieved_tflops / self.cfg.train.peak_tflops
            # Reset window accumulators
            self._tput_window_tokens = 0
            self._tput_window_seconds = 0.0
            self._tput_window_steps = 0

        self._log_metrics(metrics)

        # Main-process-only: _log_step itself runs on every rank (accelerator.log/
        # _log_metrics' callback dispatch is what's already gated), and `loss` is this
        # rank's own local value, not a cross-rank reduction -- gating here keeps the EMA
        # state and any warning single-sourced instead of one (possibly divergent) copy
        # per rank.
        if self.accelerator.is_main_process:
            self._check_loss_spike(loss)

    def _check_loss_spike(self, loss: float) -> None:
        """Warn (never abort) when a logged loss deviates sharply from its EMA trend.

        Maintains ``self._loss_ema`` with the same 0.98/0.02 smoothing
        ``oplm.training.mup``'s logging helper uses, and flags ``loss > 3 * ema`` once
        at least ``_LOSS_SPIKE_WARMUP_LOGS`` logged steps have built up a trend (a
        cold-start EMA equals the very first loss, which would otherwise trivially
        clear the 3x threshold on step one). Spikes self-recover often enough that
        acting on them (abort/rollback, as the non-finite guard does) would cause more
        harm than the spike itself, so this only logs.

        Args:
            loss: This step's logged training loss.
        """
        self._loss_log_count += 1
        if self._loss_ema is None:
            self._loss_ema = loss
        else:
            self._loss_ema = _LOSS_EMA_DECAY * self._loss_ema + (1.0 - _LOSS_EMA_DECAY) * loss

        if self._loss_log_count <= _LOSS_SPIKE_WARMUP_LOGS:
            return
        if loss > _LOSS_SPIKE_MULTIPLIER * self._loss_ema:
            logger.warning(
                "loss spike: %.4f vs EMA %.4f at step %d", loss, self._loss_ema, self.global_step
            )

    def _save_checkpoint(self, *, blocking: bool = True) -> None:
        """Save a training checkpoint.

        Updates ``self._last_save_at`` (monotonic) so the ``save_every_minutes``
        timer restarts from this save. On the main process, also anchors
        ``self._first_checkpoint_at`` (wall clock, on the first checkpoint ever)
        and applies the ``keep_every_n_hours`` rule: once wall-clock time since
        that anchor crosses a new multiple of ``keep_every_n_hours * 3600``,
        this checkpoint is marked permanent via
        :func:`~oplm.training.checkpoint.mark_permanent`. Both anchor and index
        are persisted into ``trainer_state.json`` (``first_checkpoint_unix`` /
        ``last_time_keep_index``) so a requeue resumes the hours anchor instead
        of restarting it. The ``keep_every_n_hours`` decision itself is always made
        here, at TRIGGER time, regardless of ``blocking`` -- only *applying* it
        (:func:`~oplm.training.checkpoint.mark_permanent`) is deferred for an async
        save (see :class:`_PendingAsyncSave`).

        Args:
            blocking: When ``True`` (default), calls ``save_checkpoint(...,
                blocking=True)`` and the checkpoint is fully committed --
                ``on_checkpoint_saved`` fires -- before this method returns. When
                ``False`` (the default for periodic saves, set by the caller in
                :meth:`train`), the write is handed to ``dcp.async_save`` and this
                method returns as soon as the (synchronous) sidecars are written and
                the model/optimizer write has started; the commit itself, and
                ``on_checkpoint_saved``, are deferred to :meth:`_finalize_pending_save`.
                Asserts ``self._pending_save is None`` on entry -- callers must finalize
                any existing pending save before triggering a new one (see
                :meth:`train`'s "new trigger while pending" handling).
        """
        from oplm.training.checkpoint import PendingSave, mark_permanent, save_checkpoint

        assert blocking or self._pending_save is None, (
            "at most one outstanding async save at a time -- caller must finalize first"
        )

        checkpoint_dir = Path(self.cfg.train.output_dir) / f"checkpoint-{self.global_step}"

        should_mark_permanent = False
        if self.accelerator.is_main_process:
            now_wall = time.time()
            if self._first_checkpoint_at is None:
                self._first_checkpoint_at = now_wall
            keep_every_n_hours = self.cfg.train.keep_every_n_hours
            if keep_every_n_hours is not None:
                interval_seconds = keep_every_n_hours * 3600
                elapsed = now_wall - self._first_checkpoint_at
                current_time_keep_index = int(elapsed // interval_seconds)
                if current_time_keep_index > self._last_time_keep_index:
                    should_mark_permanent = True
                    self._last_time_keep_index = current_time_keep_index

        result = save_checkpoint(
            accelerator=self.accelerator,
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            cfg=self.cfg,
            output_dir=self.cfg.train.output_dir,
            global_step=self.global_step,
            epoch=self.epoch,
            samples_seen=self._samples_seen,
            tokens_seen=self.tokens_seen,
            save_total_limit=self.cfg.train.save_total_limit,
            keep_every_n_steps=self.cfg.train.keep_every_n_steps,
            extra_state=self._checkpoint_extra_state(),
            blocking=blocking,
        )
        self._last_save_at = time.monotonic()

        if blocking:
            assert isinstance(result, Path)
            if should_mark_permanent:
                mark_permanent(checkpoint_dir)
            self._emit_checkpoint_saved(checkpoint_dir, self.global_step)
        else:
            assert isinstance(result, PendingSave)
            self._pending_save = _PendingAsyncSave(
                handle=result,
                should_mark_permanent=should_mark_permanent,
                global_step=self.global_step,
            )

    def _finalize_pending_save(self) -> None:
        """Block on the pending async save's future (a no-op if already done) and commit it.

        Runs the deferred commit tail via
        :func:`~oplm.training.checkpoint.finalize_pending_save` (barrier, rank-0 rename +
        pointer + rotation, second barrier), applies the ``keep_every_n_hours``
        permanent-marker decision made at trigger time (see :class:`_PendingAsyncSave`),
        fires ``on_checkpoint_saved`` with the checkpoint's own (trigger-time) step --
        NOT ``self.global_step``, which may have advanced past it by now -- and clears
        :attr:`_pending_save`. Called both from the opportunistic "every rank already
        reported its write done" gate in :meth:`train` (where the ``.result()`` inside
        is a no-op) and from the force-finalize paths (new trigger while pending,
        drain, and end-of-training), where it may actually block.
        """
        from oplm.training.checkpoint import finalize_pending_save, mark_permanent

        pending = self._pending_save
        assert pending is not None, "_finalize_pending_save called with no pending save"

        final_dir = finalize_pending_save(self.accelerator, pending.handle)
        if pending.should_mark_permanent:
            mark_permanent(final_dir)
        self._pending_save = None
        self._emit_checkpoint_saved(final_dir, pending.global_step)

    def _checkpoint_extra_state(self) -> dict[str, Any]:
        """Build the ``extra_state`` payload merged into ``trainer_state.json``.

        Includes the ``keep_every_n_hours`` bookkeeping keys and, when a wandb run is
        active (``self._wandb_run_id`` set on the main process), the run id (Task 1.7)
        so a resumed run can be threaded back into ``wandb.init(id=..., resume="allow")``
        without depending on the ``wandb_run_id`` marker file surviving.
        """
        extra_state: dict[str, Any] = {
            "first_checkpoint_unix": self._first_checkpoint_at,
            "last_time_keep_index": self._last_time_keep_index,
        }
        if self._wandb_run_id is not None:
            extra_state["wandb_run_id"] = self._wandb_run_id
        return extra_state

    def _resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume training state from a checkpoint."""
        from oplm.training.checkpoint import load_checkpoint

        state = load_checkpoint(
            self.accelerator,
            self.model,
            self.optimizers,
            self.schedulers,
            checkpoint_dir,
            self.cfg,
        )
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.tokens_seen = state["tokens_seen"]
        self._samples_seen = int(
            state.get("samples_seen", self.global_step * self._global_effective_batch_size())
        )
        # keep_every_n_hours anchor: restore rather than reset, so a requeue keeps
        # accumulating toward the same wall-clock boundary instead of starting over.
        self._first_checkpoint_at = state.get("first_checkpoint_unix")
        self._last_time_keep_index = int(state.get("last_time_keep_index", 0))
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

    def _emit_checkpoint_saved(self, checkpoint_dir: Path, step: int) -> None:
        """Notify callbacks that a checkpoint was saved.

        Args:
            checkpoint_dir: The committed checkpoint directory.
            step: The checkpoint's own step -- explicit rather than reading
                ``self.global_step``, since a deferred async commit (Task 2.3) may run
                after ``self.global_step`` has already advanced past the step the
                checkpoint was actually saved at.
        """
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_checkpoint_saved(self, checkpoint_dir, step)

    def _emit_train_end(self) -> None:
        """Notify callbacks that training has ended."""
        if not self.accelerator.is_main_process:
            return

        for callback in self.callbacks:
            callback.on_train_end(self)


# auto_resume fallback budget: the newest committed checkpoint plus this many
# next-newest candidates, before giving up and raising fatally.
_MAX_AUTO_RESUME_FALLBACK_ATTEMPTS = 2


def _select_auto_resume_candidate(
    output_dir: Path,
    cfg: OplmConfig,
    status: Callable[[str], None] | None,
) -> Path | None:
    """Pick the newest committed checkpoint that passes cheap pre-load validation.

    Called on the main process only, from within :func:`_resolve_resume_target`, before
    the auto-resume target is broadcast -- see that function's docstring for why the
    fallback decision has to be made and validated *before* the broadcast rather than
    after a failed ``dcp.load``.

    Walks ``checkpoint`` :func:`oplm.training.checkpoint.list_committed_checkpoints`
    (newest first) and returns the first one that passes
    :func:`oplm.training.checkpoint.validate_checkpoint_for_resume`, logging loudly (main
    process only) about every candidate skipped along the way. Gives up after the newest
    checkpoint plus :data:`_MAX_AUTO_RESUME_FALLBACK_ATTEMPTS` next-newest candidates.

    Args:
        output_dir: The training output directory to scan.
        cfg: The live, resolved config being trained/resumed with (for schedule-compat
            validation).
        status: Optional main-process-only status callback; ``None`` suppresses it.

    Returns:
        The selected checkpoint path, or ``None`` if there is no committed checkpoint at
        all (a fresh ``output_dir`` -- not an error, auto_resume is simply a no-op).

    Raises:
        Exception: Re-raises the *last* candidate's validation error if every candidate
            (newest plus fallbacks) fails validation. The caller
            (:func:`_resolve_resume_target`) turns this into a rank-identical
            ``RuntimeError`` broadcast to every rank instead of letting it propagate from
            the main process alone.
    """
    from oplm.training.checkpoint import list_committed_checkpoints, validate_checkpoint_for_resume

    candidates = list_committed_checkpoints(output_dir)
    if not candidates:
        return None

    attempts = candidates[: _MAX_AUTO_RESUME_FALLBACK_ATTEMPTS + 1]
    last_error: Exception | None = None
    for index, candidate in enumerate(attempts):
        try:
            validate_checkpoint_for_resume(candidate, cfg)
        except Exception as exc:  # noqa: BLE001 -- any validation failure triggers fallback
            remaining = len(attempts) - index - 1
            logger.error(
                "auto_resume: checkpoint %s failed pre-load validation (%s: %s); %s",
                candidate,
                type(exc).__name__,
                exc,
                f"trying the next-newest committed checkpoint ({remaining} attempt(s) left)"
                if remaining > 0
                else "no further fallback attempts left",
            )
            last_error = exc
            continue

        if index > 0:
            logger.warning(
                "auto_resume: falling back to checkpoint %s after %d newer checkpoint(s) "
                "failed pre-load validation",
                candidate,
                index,
            )
        if status is not None:
            status(f"[dim]Auto-resuming from {candidate.name}[/dim]")
        return candidate

    assert last_error is not None  # at least one attempt ran (candidates is non-empty)
    raise last_error


def _resolve_resume_target(
    accelerator: Accelerator,
    resume_from: str | None,
    auto_resume: bool,
    output_dir: str,
    cfg: OplmConfig,
    status: Callable[[str], None] | None = None,
) -> str | None:
    """Resolve the checkpoint path to resume from, rank-identical by construction.

    An explicit ``resume_from`` is already rank-identical (it's config), so no scan is
    needed for it -- and, per the binding constraint that explicit ``resume_from`` never
    falls back, it is never validated or substituted here either: if it turns out to be
    corrupt, ``Trainer._resume_from_checkpoint`` raises and the job dies (test (d)).

    When ``auto_resume`` applies instead (no explicit path given), ONLY the main process
    scans the filesystem and selects a candidate (:func:`_select_auto_resume_candidate`).
    A non-main rank never performs this scan itself: the recovery renames in
    :func:`oplm.training.checkpoint.clean_stale_checkpoint_dirs` (also main-only) plus
    shared-filesystem directory-listing caches on a multi-node cluster mean a rank's own
    scan is not guaranteed to see what the main rank sees, even after a barrier -- so the
    only way to guarantee a rank-identical result is to compute it once and hand it to
    everyone.

    Auto-resume fallback design (Task 2.2): ``dcp.load`` is collective, so retrying a
    *different* checkpoint after a failed ``dcp.load`` would require ranks to agree, mid
    failure, to retry together -- exactly the desynchronized-exception/hang risk
    rank-sync discipline exists to avoid. Instead, everything cheap to validate without
    the collective (``trainer_state.json`` readability, ``.metadata`` readability,
    schedule compatibility -- see
    :func:`oplm.training.checkpoint.validate_checkpoint_for_resume`) runs here, on the
    main process, *before* the broadcast: a torn or schedule-incompatible candidate is
    rejected identically on every rank, and the next-newest committed checkpoint is
    substituted (and broadcast) in its place, up to
    :data:`_MAX_AUTO_RESUME_FALLBACK_ATTEMPTS` times. If every candidate fails, the main
    process's last validation error is packaged into the broadcast payload (rather than
    raised locally, which would leave every other rank hanging at the broadcast call) and
    every rank raises an identical ``RuntimeError`` from it. A genuine failure *inside*
    ``dcp.load`` after a candidate has already passed this validation is treated as
    fatal: it is allowed to propagate straight out of ``Trainer._resume_from_checkpoint``,
    the job dies, and the external requeue loop retries -- which, given the pre-validation
    above, should be rare.

    Args:
        accelerator: The trainer's Accelerator, called after the ``wait_for_everyone``
            barrier that follows ``clean_stale_checkpoint_dirs``.
        resume_from: ``cfg.train.resume_from`` -- an explicit operator-pinned path, or
            ``None``.
        auto_resume: ``cfg.train.auto_resume``.
        output_dir: The training output directory to scan when ``auto_resume`` applies.
        cfg: The live, resolved config being trained/resumed with (threaded into
            candidate validation for schedule-compat checking).
        status: Optional main-process-only status callback (mirrors the trainer's
            ``_status`` helper); ``None`` suppresses it.

    Returns:
        The resolved checkpoint directory path, identical on every rank, or ``None`` if
        there is nothing to resume.

    Raises:
        RuntimeError: Every auto-resume candidate (newest plus fallbacks) failed pre-load
            validation. Raised identically on every rank.
    """
    from accelerate.utils import broadcast_object_list

    resume_target: str | None = resume_from
    resolve_error: str | None = None
    if resume_target is None and auto_resume and accelerator.is_main_process:
        try:
            found = _select_auto_resume_candidate(Path(output_dir), cfg, status)
        except Exception as exc:  # noqa: BLE001 -- packaged into the broadcast, see above
            resolve_error = (
                f"auto_resume: no usable checkpoint found under {output_dir} after "
                f"{_MAX_AUTO_RESUME_FALLBACK_ATTEMPTS + 1} attempt(s); last error: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            resume_target = str(found) if found is not None else None

    resume_target, resolve_error = broadcast_object_list([resume_target, resolve_error])
    if resolve_error is not None:
        raise RuntimeError(resolve_error)
    return resume_target


def _read_resume_wandb_run_id(resume_target: str, output_dir: str) -> str | None:
    """Read the wandb run id to resume, for threading into ``wandb.init(id=..., ...)``.

    Prefers the id recorded in the resume target checkpoint's own ``trainer_state.json``
    (authoritative for that exact checkpoint); falls back to the
    ``<output_dir>/wandb_run_id`` marker file (e.g. a checkpoint saved before this field
    existed, or a checkpoint dir moved/copied without its marker).

    Args:
        resume_target: Path to the checkpoint directory being resumed from.
        output_dir: The training output directory.

    Returns:
        The wandb run id string, or ``None`` if neither source has one.
    """
    state_path = Path(resume_target) / "trainer_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
        run_id = state.get("wandb_run_id")
        if run_id:
            return str(run_id)

    id_path = Path(output_dir) / "wandb_run_id"
    if id_path.is_file():
        text = id_path.read_text().strip()
        if text:
            return text
    return None


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

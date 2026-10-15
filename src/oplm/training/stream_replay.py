"""Replay a run's exact training stream at chosen steps and score a frozen model on it.

The data order is a pure function of the config, the step, and the ``(rank, worker)``
layout, so the batches one rank served at any global step can be rebuilt offline with the
same resume-skip machinery the trainer uses on requeue. Scoring a *frozen* checkpoint on
windows before and after an event (an epoch boundary, a loss jump) separates "the data got
harder" from "the model got worse". Rows are tagged with their source so the loss and the
mix can be split per ``data.train`` entry. See docs/LATE_LOSS_INVESTIGATION.md.
"""

from __future__ import annotations

import copy
import math
import os
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader

from oplm.data.sequence.dataset import InterleavedDataset

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torch import nn

    from oplm.config import OplmConfig

__all__ = [
    "StreamReplay",
    "compile_like_trainer",
    "compiled_step_loss",
    "score_batch",
    "steps_per_epoch",
    "trim_batch",
]


class _SourceTagged(InterleavedDataset):
    """:class:`InterleavedDataset` whose rows carry ``"source"``: their source's index."""

    def _next_or_refill(
        self, iters: list[Iterator[dict[str, object]]], passes: list[int], source_idx: int
    ) -> object:
        item = super()._next_or_refill(iters, passes, source_idx)
        return {**item, "source": source_idx} if isinstance(item, dict) else item


class _TaggingCollator:
    """Run the training collator, then attach each row's source index as ``"source"``."""

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(self, rows: list[Mapping[str, object]]) -> dict[str, Any]:
        sources = torch.tensor([int(r.get("source", 0)) for r in rows])  # ty: ignore[invalid-argument-type]
        batch = self.collator([{k: v for k, v in r.items() if k != "source"} for r in rows])
        batch["source"] = sources
        return batch


def steps_per_epoch(
    n_samples: int,
    *,
    rank: int,
    world_size: int,
    num_workers: int,
    batch_size: int,
    drop_last: bool = False,
) -> int:
    """Batches one rank's DataLoader yields per epoch.

    Rows are striped over the joint ``(rank, worker)`` index. Without ``drop_last``
    worker ``w`` serves ``len(range(rank * num_workers + w, n_samples, world_size *
    num_workers))`` rows and ends on a partial batch; with it every worker serves
    ``n_samples // (world_size * num_workers)`` rows in full batches only.
    """
    stride = world_size * num_workers
    if drop_last:
        return num_workers * ((n_samples // stride) // batch_size)
    return sum(
        math.ceil(len(range(rank * num_workers + w, n_samples, stride)) / batch_size)
        for w in range(num_workers)
    )


class StreamReplay:
    """One rank's training stream, re-positionable at any global step.

    Builds the run's training dataset once (its shard metadata reads dominate the cost) and
    re-arms it per window with the trainer's own epoch and resume-skip arithmetic. Assumes one
    batch per accumulation micro-step and the world size, worker count and batch size the run
    used throughout, as the trainer's resume does.

    Args:
        cfg: The run's resolved config (a checkpoint's ``config.yaml``).
        rank: Rank whose stream to rebuild.
        world_size: The run's world size (global batch / ``train.batch_size``).
    """

    def __init__(
        self, cfg: OplmConfig, *, rank: int, world_size: int, spawn_workers: bool = False
    ) -> None:
        from oplm.data import build_train_dataloader

        # Rows are tagged and scored one sequence per row. Packing happens after masking,
        # so an unpacked replay of a packed run serves the same sequences and masks.
        cfg = copy.deepcopy(cfg)
        cfg.data.pack_sequences = False
        template = build_train_dataloader(cfg)
        dataset: Any = template.dataset
        if isinstance(dataset, InterleavedDataset):
            dataset = _SourceTagged(
                dataset._datasets,
                dataset._fractions,
                num_samples=len(dataset),
                seed=dataset._seed,
                reshuffle_on_refill=dataset._reshuffle_on_refill,
                drop_remainder=dataset._drop_remainder,
            )
        self.cfg, self.rank, self.world_size = cfg, rank, world_size
        self.dataset = dataset
        self.collator = _TaggingCollator(template.collate_fn)
        self.num_workers = max(cfg.data.num_workers, 1)
        self.steps_per_epoch = steps_per_epoch(
            len(dataset),
            rank=rank,
            world_size=world_size,
            num_workers=self.num_workers,
            batch_size=cfg.train.batch_size,
            drop_last=cfg.data.drop_last,
        )
        stride = world_size * self.num_workers
        rows = [
            len(range(rank * self.num_workers + w, len(dataset), stride))
            for w in range(self.num_workers)
        ]
        # Each worker's last batch of an epoch: the short steps that close every epoch
        # (none with drop_last, which serves only full batches).
        self.partial_sizes = [
            n % cfg.train.batch_size
            for n in rows
            if n % cfg.train.batch_size and not cfg.data.drop_last
        ]
        # A process group in this process would override the RANK/WORLD_SIZE the stream reads;
        # spawned (not forked) workers do not inherit it.
        self.multiprocessing_context = "spawn" if spawn_workers else None

    def position(self, step: int) -> tuple[int, int]:
        """``(epoch, batches_in_epoch)`` of global optimizer step ``step``."""
        return divmod(step * self.cfg.train.gradient_accumulation_steps, self.steps_per_epoch)

    def batches(self, step: int, num_batches: int) -> Iterator[dict[str, Any]]:
        """The ``num_batches`` batches this rank trained on from global step ``step``.

        Every batch carries a ``"source"`` tensor: each row's ``data.train`` entry index.
        """
        epoch, in_epoch = self.position(step)
        self.dataset.set_epoch(epoch)
        self.dataset.set_resume_skip(in_epoch, self.cfg.train.batch_size, self.num_workers)
        workers = self.cfg.data.num_workers
        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            collate_fn=self.collator,
            num_workers=workers,
            prefetch_factor=self.cfg.data.prefetch_factor if workers > 0 else None,
            multiprocessing_context=self.multiprocessing_context if workers > 0 else None,
            drop_last=self.cfg.data.drop_last,
        )
        # With no process group the stream reads rank and world size from the environment
        # (DataLoader workers inherit it when they start); restored once iteration ends.
        saved = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE")}
        os.environ["RANK"], os.environ["WORLD_SIZE"] = str(self.rank), str(self.world_size)
        try:
            for i, batch in enumerate(loader):
                if i == num_batches:
                    return
                yield batch
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@torch.no_grad()
def score_batch(
    batch: Mapping[str, torch.Tensor], num_sources: int, model: nn.Module | None = None
) -> torch.Tensor:
    """Per-source ``[loss_sum, masked_tokens, rows, tokens]`` of one replayed batch.

    ``loss_sum`` is the frozen ``model``'s masked-token cross-entropy (zero without a model).

    Returns:
        A ``(num_sources, 4)`` float64 tensor on the CPU.
    """
    labels = batch["labels"]
    masked = labels != -100
    loss = torch.zeros(len(labels), dtype=torch.float64)
    if model is not None:
        device = next(model.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )["logits"]
        ce = torch.nn.functional.cross_entropy(
            logits.float().transpose(1, 2), labels.to(device), ignore_index=-100, reduction="none"
        )
        loss = ce.sum(1).double().cpu()  # ignored positions contribute zero
    per_row = torch.stack(
        [
            loss,
            masked.sum(1).double(),
            torch.ones(len(labels), dtype=torch.float64),
            batch["attention_mask"].sum(1).double(),
        ],
        dim=1,
    )
    out = torch.zeros(num_sources, 4, dtype=torch.float64)
    return out.index_add_(0, batch["source"].long(), per_row)


def compiled_step_loss(
    compiled: nn.Module, model: nn.Module, batch: Mapping[str, torch.Tensor]
) -> tuple[float, float]:
    """``(loss_sum, masked_tokens)`` of one training step through ``compiled``, weights untouched.

    Runs what the trainer runs: train mode, bf16 autocast on CUDA, the model's own loss, and a
    backward pass, so the compiled graph is the joint forward/backward training graph rather
    than an inference graph. Gradients are discarded; no optimizer step is taken.

    Args:
        compiled: ``torch.compile`` of ``model`` (optionally of a DDP wrapper around it).
        model: The underlying module (owns the parameters whose gradients are cleared).
        batch: A replayed batch.
    """
    device = next(model.parameters()).device
    labels = batch["labels"].to(device)
    model.train()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = compiled(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels,
        )["loss"]
    loss.backward()
    model.zero_grad(set_to_none=True)
    model.eval()
    masked = float((labels != -100).sum())
    return float(loss.detach()) * masked, masked


def compile_like_trainer(model: nn.Module, cfg: OplmConfig, *, ddp: bool) -> nn.Module:
    """``torch.compile`` the model the way the trainer does, optionally behind DDP.

    The trainer compiles the DDP-wrapped model (``torch.compile(model, dynamic=...,
    mode=...)``), and Dynamo splits a DDP module's graph at gradient-bucket boundaries. With
    ``ddp`` this sets up a single-process group so the compiled graph has that structure too.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    target = model
    if ddp:
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        device = next(model.parameters()).device
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )
        target = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    # torch.compile's stub returns a callable; the OptimizedModule it builds is an nn.Module.
    return cast(
        "nn.Module",
        torch.compile(target, dynamic=cfg.train.compile_dynamic, mode=cfg.train.compile_mode),
    )


def trim_batch(batch: Mapping[str, torch.Tensor], rows: int) -> dict[str, torch.Tensor]:
    """The first ``rows`` rows of a batch, padded only to their own longest row (as collated)."""
    width = int(batch["attention_mask"][:rows].sum(1).max())
    return {
        k: v[:rows, :width] if v.dim() == 2 else v[:rows]
        for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }

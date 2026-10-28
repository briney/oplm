"""FoldTask: the structure head's model, dataloader and step for the Trainer (spec §6.4).

The Trainer keeps acceleration, optimization, accumulation, EMA, checkpointing, resume and
eval cadence; this module supplies the three objective-specific pieces. ``build_model`` builds
the head from the ``fold:`` block (shape fields inferred from the loaded LM), attaches the frozen
LM on the process device (bf16 on CUDA) and enables block-level checkpointing; ``build_dataloader``
makes one :class:`StructureDataset` per ``data.train`` entry (every entry of a fold run is a
structure shard directory) mixed through :class:`InterleavedDataset`, seeded with the run seed so
the Trainer's ``set_epoch``/resume-skip forwarding restores the exact example stream; ``step``
runs the model's training branch and returns a :class:`StepResult`.
"""

from __future__ import annotations

import inspect
import time
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

import oplm.fold.eval  # noqa: F401  (registers the fold_monomer eval type, docs/FOLD.md §10)
from oplm.data.config import parse_train_configs
from oplm.data.sequence.dataset import InterleavedDataset
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.collate import FoldBatch, collate_rows
from oplm.fold.data.dataset import StructureDataset
from oplm.fold.lm_shim import lm_state_count
from oplm.fold.losses import LossWeights
from oplm.fold.modeling_fold import FoldTrainOutput, OplmForFolding, TrainSettings, load_frozen_lm
from oplm.training.callbacks import TrainerCallback
from oplm.training.initialization import resolve_initialization_source
from oplm.training.task import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from oplm.config import FoldRunConfig, OplmConfig
    from oplm.training.trainer import Trainer

__all__ = [
    "FoldTask",
    "PeakMemoryCallback",
    "build_fold_config",
    "load_initial_head",
    "require_fold",
]

_LM_RUN_FIELDS = ("lm_name_or_path", "lm_revision", "lm_max_chain_length")


def require_fold(cfg: OplmConfig) -> FoldRunConfig:
    """Return ``cfg.fold`` or raise a ``ValueError`` naming the missing block."""
    if cfg.fold is None:
        raise ValueError(
            "this is not a fold run: the config has no `fold:` block (docs/FOLD.md §9)"
        )
    return cfg.fold


def build_fold_config(run: FoldRunConfig, *, lm: nn.Module | None = None) -> FoldConfig:
    """``FoldConfig`` from the run block: ``run.model`` overrides, LM fields, inferred LM shape.

    Raises:
        ValueError: ``run.model`` names a key ``FoldConfig`` does not accept, or one of the
            LM fields that come from the run block itself.
    """
    overrides = dict(run.model or {})
    allowed = {
        name
        for name, p in inspect.signature(FoldConfig.__init__).parameters.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown FoldConfig key(s) in fold.model: {unknown}")
    clash = sorted(set(overrides) & set(_LM_RUN_FIELDS))
    if clash:
        raise ValueError(
            f"fold.model may not set {clash}; "
            "use fold.lm / fold.lm_revision / fold.lm_max_chain_length"
        )
    if lm is not None:
        overrides.setdefault("lm_hidden_size", int(lm.config.hidden_size))  # ty: ignore[unresolved-attribute, invalid-argument-type]  # nn.Module attrs
        overrides.setdefault("lm_num_hidden_states", lm_state_count(lm))
    return FoldConfig(
        **overrides,
        lm_name_or_path=run.lm,
        lm_revision=run.lm_revision,
        lm_max_chain_length=run.lm_max_chain_length,
    )


def load_initial_head(source: Path | str, target: FoldConfig) -> OplmForFolding:
    """Load every head tensor from a saved ``hf/`` or ``hf_ema/`` export (``train.init_from``).

    The saved architecture must equal ``target`` field for field, except the LM identity fields
    (``lm_name_or_path``, ``lm_revision``, ``lm_max_chain_length``), which the run owns and which
    are written onto the loaded config.

    Raises:
        ValueError: An architecture field differs, or tensors are missing or unexpected.
    """
    path = resolve_initialization_source(str(source))
    saved = FoldConfig.from_pretrained(path, local_files_only=True)
    architecture = [
        name
        for name, p in inspect.signature(FoldConfig.__init__).parameters.items()
        if p.kind == p.KEYWORD_ONLY and name not in _LM_RUN_FIELDS
    ]
    differences = [
        f"{k} (saved={getattr(saved, k)!r}, run={getattr(target, k)!r})"
        for k in architecture
        if getattr(saved, k) != getattr(target, k)
    ]
    if differences:
        raise ValueError(
            "init_from head does not match the run's fold.model: " + "; ".join(differences)
        )
    # ``from_pretrained`` would auto-load the LM named by the saved config; the run attaches its
    # own, so clear the name (an unset name attaches nothing) and write the run's identity below.
    saved.lm_name_or_path = None
    model, info = OplmForFolding.from_pretrained(
        path, config=saved, local_files_only=True, output_loading_info=True
    )
    problems = {
        k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(k)
    }
    if problems:
        raise ValueError(f"incomplete head initialization from {path}: {problems}")
    for name in _LM_RUN_FIELDS:
        setattr(model.config, name, getattr(target, name))
    return model


class FoldTask:
    """The structure head's :class:`~oplm.training.task.TrainTask`.

    See docs/training.md, "Training tasks".
    """

    def __init__(self) -> None:
        self.settings = TrainSettings()
        self.last_batch_ids: list[str] = []

    def build_model(self, cfg: OplmConfig, initialization_source: Path | None) -> nn.Module:
        """Build the head, attach the frozen LM on the process device, enable checkpointing."""
        from accelerate import PartialState

        run = require_fold(cfg)
        device = PartialState().device
        assert device is not None  # PartialState always resolves one
        dtype = torch.bfloat16 if device.type == "cuda" else None
        lm = load_frozen_lm(run.lm, revision=run.lm_revision, device=device, dtype=dtype)
        fold_cfg = build_fold_config(run, lm=lm)
        model = (
            OplmForFolding(fold_cfg)
            if initialization_source is None
            else load_initial_head(initialization_source, fold_cfg)
        )
        model.attach_lm(lm, name_or_path=run.lm, revision=run.lm_revision)
        model.train()
        model.gradient_checkpointing_enable()
        self.settings = TrainSettings(
            diffusion_samples=run.diffusion_samples,
            diffusion_microbatch=run.diffusion_microbatch,
            confidence_rollout_steps=run.confidence_rollout_steps,
            weights=LossWeights(
                diffusion_mse=run.loss_diffusion_mse,
                smooth_lddt=run.loss_smooth_lddt,
                distogram=run.loss_distogram,
                confidence=run.loss_confidence,
                pae=run.loss_pae,
            ),
        )
        # Decorrelate the in-model draws (loop count, sigma, noise, augmentation, dropout, LM
        # masking) across ranks. This comes after construction so every rank initialises the
        # head identically (DDP's parameter broadcast stays a no-op); a resume restores each
        # rank's RNG from its own sidecar afterwards. Seeds CUDA too; the data stream has its own
        # explicit (seed, epoch, shard, row) generators.
        torch.manual_seed(cfg.train.seed + PartialState().process_index)
        return model

    def build_dataloader(self, cfg: OplmConfig) -> DataLoader[Any]:
        """One ``StructureDataset`` per ``data.train`` entry, mixed by fraction, run-seeded."""
        run = require_fold(cfg)
        entries = parse_train_configs(cfg.data.train)
        if not entries:
            raise ValueError("cfg.data.train resolved to no datasets; nothing to train on")
        sources = [
            StructureDataset(
                entry.path,
                split="train",
                crop_size=run.crop_size,
                contiguous_prob=run.contiguous_prob,
                max_chain_length=run.lm_max_chain_length,
                seed=cfg.train.seed,
                shuffle_shards=cfg.data.shuffle_shards,
                shuffle_rows=cfg.data.shuffle_rows,
            )
            for entry in entries
        ]
        dataset: Any = (
            InterleavedDataset(sources, [e.fraction for e in entries], seed=cfg.train.seed)
            if len(sources) > 1
            else sources[0]
        )
        atom_pad_multiple = int(
            (run.model or {}).get("atom_pad_multiple", FoldConfig().atom_pad_multiple)
        )
        collate = partial(
            collate_rows,
            crop_size=run.crop_size,
            atoms_per_token_budget=run.atoms_per_token_budget,
            atom_pad_multiple=atom_pad_multiple,
        )
        num_workers = cfg.data.num_workers
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            prefetch_factor=cfg.data.prefetch_factor if num_workers > 0 else None,
        )

    def step(self, model: nn.Module, batch: Any) -> StepResult:
        """Run the training branch on one ``FoldBatch`` and return its loss and counts."""
        assert isinstance(batch, FoldBatch), type(batch)
        out = model(batch.features, targets=batch.targets, train=self.settings)
        assert isinstance(out, FoldTrainOutput)
        self.last_batch_ids = list(batch.ids)
        return StepResult(loss=out.loss, tokens=out.tokens, samples=out.chains, metrics=out.metrics)

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        """No honest estimate: cost depends on crop, loops and samples (spec §6.4)."""
        return None


class PeakMemoryCallback(TrainerCallback):
    """Record per-log step time and CUDA peak memory; ``oplm fold measure`` writes the result."""

    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []
        self._started = time.monotonic()
        self._cuda = False

    def on_train_start(self, trainer: Trainer) -> None:
        self._started = time.monotonic()
        # Gate on the trainer's device, not torch.cuda.is_available(): a CPU-forced run on a
        # GPU host has no CUDA peaks to report.
        self._cuda = trainer.accelerator.device.type == "cuda"
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        if "train/loss" not in metrics:
            return
        record = {"step": float(step), "elapsed_s": time.monotonic() - self._started}
        record.update(
            {
                k.removeprefix("train/"): float(v)
                for k, v in metrics.items()
                if k.startswith("train/")
            }
        )
        if self._cuda:
            record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        self.records.append(record)

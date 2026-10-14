"""Shared builders for the fold tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from oplm.fold.configuration_fold import FoldConfig
from oplm.model import OplmConfig, OplmModel

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.config import OplmConfig as RunConfig
    from oplm.fold.data.featurize import FoldFeatures, FoldTargets
    from oplm.fold.modeling_fold import OplmForFolding


def tiny_lm(num_loops: int = 1) -> OplmModel:
    """A 2-layer, 32-wide OplmModel in eval mode (seeded) for fold tests."""
    torch.manual_seed(1)
    cfg = OplmConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        num_loops=num_loops,
    )
    return OplmModel(cfg).eval()


def tiny_fold_config(**overrides: Any) -> FoldConfig:
    """A small, valid FoldConfig for CPU tests (dense attention, reference trimul)."""
    base: dict[str, Any] = dict(
        pair_width=32,
        token_width=64,
        atom_width=32,
        atom_encoder_blocks=1,
        atom_encoder_heads=2,
        uid_rope_pairs=2,
        trunk_blocks=1,
        lm_encoder_blocks=1,
        coda_blocks=1,
        diffusion_blocks=1,
        diffusion_heads=4,
        diffusion_atom_blocks=1,
        diffusion_atom_heads=2,
        fourier_dim=16,
        confidence_blocks=1,
        inference_num_steps=3,
        attention_backend="dense",
        trimul_backend="reference",
    )
    base.update(overrides)
    return FoldConfig(**base)


def tiny_folder() -> OplmForFolding:
    """An eval-mode tiny folding model with a tiny LM attached, for fold() tests."""
    from oplm.fold.modeling_fold import OplmForFolding

    torch.manual_seed(0)
    cfg = tiny_fold_config(
        plddt_bins=10,
        pae_bins=8,
        pde_bins=8,
        confidence_dist_bins=5,
        distogram_bins=8,
        inference_num_steps=2,
        inference_num_loops=1,
        lm_hidden_size=32,
        lm_num_hidden_states=3,
    )
    model = OplmForFolding(cfg).eval()
    model.attach_lm(tiny_lm())
    return model


def dummy_targets(features: FoldFeatures) -> FoldTargets:
    """All-unresolved targets with the right axes, for tests that need a FoldExample."""
    from oplm.fold.data.featurize import FoldTargets

    L, A = features.num_tokens, features.num_atoms
    return FoldTargets(
        coords=torch.zeros(1, A, 3),
        resolved=torch.zeros(1, A, dtype=torch.bool),
        bfactor=torch.zeros(1, A),
        frame_atom_idx=torch.zeros(1, L, 3, dtype=torch.long),
    )


TINY_FOLD_MODEL: dict[str, Any] = dict(
    pair_width=32,
    token_width=64,
    atom_width=32,
    atom_encoder_blocks=1,
    atom_encoder_heads=2,
    uid_rope_pairs=2,
    trunk_blocks=1,
    lm_encoder_blocks=1,
    coda_blocks=1,
    diffusion_blocks=1,
    diffusion_heads=4,
    diffusion_atom_blocks=1,
    diffusion_atom_heads=2,
    fourier_dim=16,
    confidence_blocks=1,
    inference_num_steps=2,
    inference_num_loops=2,
    attention_backend="dense",
    trimul_backend="reference",
    plddt_bins=10,
    pae_bins=8,
    pde_bins=8,
    confidence_dist_bins=5,
    distogram_bins=8,
)


def tiny_fold_run_cfg(
    output_dir: Path,
    dataset: Path,
    lm_dir: Path,
    *,
    max_steps: int = 4,
    batch_size: int = 2,
    save_every: int = 10_000,
    ema_decay: float | None = None,
    resume_from: str | None = None,
    init_from: str | None = None,
    seed: int = 0,
    gradient_accumulation_steps: int = 1,
    lr: float = 1e-3,
) -> RunConfig:
    """A CPU fold run on the preprocessed fixtures: tiny head, the saved tiny LM, crop 128."""
    # The run config shares its name with the HF model config imported above; alias it locally.
    from oplm.config import DataConfig, FoldRunConfig, TrainConfig
    from oplm.config import OplmConfig as RunConfig

    return RunConfig(
        # The checkpoint layer serializes ``cfg.model`` (an HF config) on every save, so a fold
        # run carries a default one even though FoldTask never reads it.
        model=OplmConfig(),
        train=TrainConfig(
            max_steps=max_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer="adamw",
            lr=lr,
            warmup_steps=0,
            max_grad_norm=1.0,
            log_every=1,
            save_every=save_every,
            save_final=False,
            ema_decay=ema_decay,
            resume_from=resume_from,
            init_from=init_from,
            seed=seed,
            output_dir=str(output_dir),
            mixed_precision="no",
            wandb_enabled=False,
        ),
        data=DataConfig(train=str(dataset), num_workers=0, pin_memory=False),
        fold=FoldRunConfig(
            lm=str(lm_dir),
            model=dict(TINY_FOLD_MODEL),
            crop_size=128,
            diffusion_samples=2,
            diffusion_microbatch=1,
            confidence_rollout_steps=2,
        ),
    )

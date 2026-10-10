"""Shared builders for the fold tests."""

from __future__ import annotations

from typing import Any

import torch

from oplm.fold.configuration_fold import FoldConfig
from oplm.model import OplmConfig, OplmModel


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

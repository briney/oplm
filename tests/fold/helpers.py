"""Shared builders for the fold tests."""

from __future__ import annotations

import torch

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

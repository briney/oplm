"""Token and value embedding modules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from oplm.model.norm import RMSNorm

if TYPE_CHECKING:
    from oplm.config import ModelConfig


class TokenEmbedding(nn.Module):
    """Scaled token embedding with optional post-embedding normalization.

    Scaling by ``sqrt(hidden_dim)`` follows Proust convention to prevent
    embedding magnitude from being dwarfed by hidden state magnitudes.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.scale = math.sqrt(config.hidden_dim)
        self.post_norm = (
            RMSNorm(config.hidden_dim, config.norm_eps) if config.post_embed_norm else None
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed and scale token IDs.

        Args:
            input_ids: Integer tensor of shape ``(B, T)``.

        Returns:
            Embedded tensor of shape ``(B, T, D)``.
        """
        x: Tensor = self.embed(input_ids) * self.scale
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


class ValueEmbedding(nn.Module):
    """Separate embedding tables injected into V at selected layers.

    From Proust: applied symmetrically to the first N and last N layers,
    where N = ``num_value_embeds``. Each layer's value embedding is gated
    by a learned projection from the first ``value_embed_gate_dim`` dimensions
    of the hidden state.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.head_dim or config.hidden_dim // config.num_heads
        kv_dim = config.num_kv_heads * head_dim
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = head_dim
        self.gate_dim = config.value_embed_gate_dim

        self.embeds = nn.ModuleList(
            [nn.Embedding(config.vocab_size, kv_dim) for _ in range(config.num_value_embeds)]
        )
        self.gates = nn.ModuleList(
            [
                nn.Linear(config.value_embed_gate_dim, config.num_kv_heads)
                for _ in range(config.num_value_embeds)
            ]
        )

        # Map layer index to embed index: first N and last N layers
        n = config.num_value_embeds
        self.layer_map: dict[int, int] = {}
        for i in range(n):
            self.layer_map[i] = i
            self.layer_map[config.num_layers - n + i] = i

    def uses_layer(self, layer_idx: int) -> bool:
        """Return whether a layer consumes a value embedding."""
        return layer_idx in self.layer_map

    def forward(self, input_ids: Tensor, x: Tensor, layer_idx: int) -> Tensor | None:
        """Compute gated value embedding for a given layer.

        Args:
            input_ids: Token IDs ``(B, T)``.
            x: Hidden state ``(B, T, D)``.
            layer_idx: Current layer index.

        Returns:
            Gated value embedding ``(B, T, KV_H, D_head)`` or None if this
            layer does not use value embeddings.
        """
        if not self.uses_layer(layer_idx):
            return None
        idx = self.layer_map[layer_idx]
        ve: Tensor = self.embeds[idx](input_ids)  # (B, T, kv_dim)
        gate_input = x[..., : self.gate_dim]  # (B, T, gate_dim)
        gate: Tensor = 2.0 * torch.sigmoid(self.gates[idx](gate_input))  # (B, T, num_kv_heads)
        ve = ve.view(*ve.shape[:-1], self.num_kv_heads, self.head_dim)  # (B, T, KV_H, D_head)
        return gate.unsqueeze(-1) * ve  # (B, T, KV_H, D_head)

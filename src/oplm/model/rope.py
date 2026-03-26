"""Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dimensions: ``[-x2, x1]``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Standard Rotary Position Embedding applied to Q and K.

    Precomputes inverse frequencies and cos/sin cache up to ``max_seq_len``.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to Q and K.

        Args:
            q: Query tensor of shape ``(B, T, H, D)``.
            k: Key tensor of shape ``(B, T, KV_H, D)``.
            position_ids: Optional position indices ``(B, T)`` or ``(total_tokens,)``.

        Returns:
            Rotated (q, k) with the same shapes.
        """
        if position_ids is None:
            seq_len = q.shape[1]
            # Extend cache if needed
            if seq_len > self.max_seq_len:
                self._build_cache(seq_len)
                self.max_seq_len = seq_len
            # (1, T, 1, D) for broadcasting over B and H
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(-2)  # (..., 1, D)
            sin = self.sin_cached[position_ids].unsqueeze(-2)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot


class PartialRotaryEmbedding(nn.Module):
    """Partial RoPE for GQA-S2 mode.

    Splits head dimensions into position-invariant (NoPE) and position-dependent
    (RoPE) portions. RoPE is applied only to the ``rope_dim`` portion of Q and K.
    """

    def __init__(
        self,
        rope_dim: int,
        nope_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.rope_dim = rope_dim
        self.nope_dim = nope_dim
        self.rotary = RotaryEmbedding(rope_dim, max_seq_len, theta)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply partial rotary embeddings.

        Args:
            q: Query tensor ``(B, T, H, D)`` where ``D = nope_dim + rope_dim``.
            k: Key tensor ``(B, T, KV_H, D)``.
            position_ids: Optional position indices.

        Returns:
            (q, k) with RoPE applied only to the rope_dim portion.
        """
        # Split into NoPE and RoPE portions
        q_nope, q_rope = q[..., : self.nope_dim], q[..., self.nope_dim :]
        k_nope, k_rope = k[..., : self.nope_dim], k[..., self.nope_dim :]

        # Apply RoPE only to the rope portion
        q_rope, k_rope = self.rotary(q_rope, k_rope, position_ids)

        return (
            torch.cat([q_nope, q_rope], dim=-1),
            torch.cat([k_nope, k_rope], dim=-1),
        )

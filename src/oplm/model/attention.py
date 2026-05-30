"""Dual-path multi-head attention (flex_attention fast path + manual fallback)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .masking import make_flex_block_mask
from .norm import make_norm
from .rope import RotaryEmbedding

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["OplmAttention"]


# `flex_attention` requires CUDA and a compatible torch build. Probe once at
# import time so the per-forward fast-path guard is a cheap boolean check.
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    _FLEX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on torch builds without flex
    _flex_attention = None  # ty: ignore[invalid-assignment]  # optional-import None sentinel
    _FLEX_AVAILABLE = False


class OplmAttention(nn.Module):
    """Multi-head self-attention with a flex_attention fast path and SDPA fallback.

    The two compute paths share one set of parameters and one set of
    pre-attention transformations (Q/K/V projections, optional QK/V norm, RoPE);
    only the attention kernel itself differs. The fast path uses
    `torch.nn.attention.flex_attention.flex_attention` with a closure-driven
    `BlockMask`; the fallback is a manual scaled-dot-product softmax that also
    returns the attention weights when `output_attentions=True`.

    Norm wiring (controlled by `config.norm_strategy`):

    * `qk_norm=True` always installs `q_norm` and `k_norm` on the per-head
      dimension (`d_head`).
    * Under `norm_strategy == "hybrid"`, an additional `v_norm` is installed on
      `d_head` — this realises the paper's "QKV-norm" main method.
    * Under every other strategy, `v_norm` is `nn.Identity()`.

    The output projection (`o_proj`) is marked `_is_residual_writer = True` so
    the `OplmPreTrainedModel._init_weights` hook can apply the
    `1/sqrt(2L)` residual-stream scaling defined in §15.1 of the architecture
    doc.
    """

    def __init__(self, config: OplmConfig) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        if head_dim * num_heads != hidden_size:
            raise ValueError(
                f"head_dim ({head_dim}) * num_attention_heads ({num_heads}) must equal "
                f"hidden_size ({hidden_size})."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.head_dim = head_dim
        self.attention_dropout = float(config.attention_dropout)
        self.hidden_dropout = float(config.hidden_dropout)
        self.use_flex_attention = bool(config.use_flex_attention)
        self.norm_strategy = config.norm_strategy
        self.qk_norm_enabled = bool(config.qk_norm)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Picked up by OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling.
        self.o_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

        if self.qk_norm_enabled:
            self.q_norm: nn.Module = make_norm(config.norm_type, head_dim, eps=config.norm_eps)
            self.k_norm: nn.Module = make_norm(config.norm_type, head_dim, eps=config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Hybrid strategy ("QKV-norm") also norms V; every other strategy leaves V alone.
        if config.norm_strategy == "hybrid":
            self.v_norm: nn.Module = make_norm(config.norm_type, head_dim, eps=config.norm_eps)
        else:
            self.v_norm = nn.Identity()

        self.rotary = RotaryEmbedding(
            head_dim=head_dim,
            rope_dim=config.rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to Q/K/V and reshape from `(B, T, D)` to `(B, H, T, d_head)`."""
        batch, seq_len, _ = x.shape
        h, d = self.num_attention_heads, self.head_dim

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, h, d).transpose(1, 2)

        return split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))

    def _qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head QK normalization (no-op when `qk_norm=False`).

        Norm internals already cast to fp32; we forward the result in the same
        dtype the norm returns (matches the input dtype).
        """
        return self.q_norm(q), self.k_norm(k)

    def _apply_v_norm(self, v: torch.Tensor) -> torch.Tensor:
        return self.v_norm(v)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary.apply_rotary(q, k)

    def _output_projection(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Reshape `(B, H, T, d_head)` back to `(B, T, D)` and project."""
        batch, _, seq_len, _ = attn_out.shape
        merged = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(merged)

    # ------------------------------------------------------------------
    # Fast-path guard / kernels
    # ------------------------------------------------------------------

    def _use_fast_path(self, output_attentions: bool, device: torch.device) -> bool:
        """Decide whether to dispatch to `flex_attention`.

        All conditions must hold; otherwise the manual fallback runs. The fast
        path returns no attention weights and applies no dropout, so requesting
        either forces the fallback. CUDA is required by the kernel itself.
        """
        if output_attentions:
            return False
        if not self.use_flex_attention:
            return False
        # flex_attention exposes no `dropout_p` argument; honouring the
        # configured attention_dropout exactly requires the manual fallback.
        if self.attention_dropout != 0.0:
            return False
        if device.type != "cuda":
            return False
        return _FLEX_AVAILABLE

    def _flex_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run `flex_attention` with a closure-built `BlockMask`."""
        block_mask = make_flex_block_mask(attention_mask, num_heads=self.num_attention_heads)
        # torch's flex_attention return type is a union; this path returns a Tensor.
        return _flex_attention(q, k, v, block_mask=block_mask)  # ty: ignore[invalid-return-type]

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Manual scaled-dot-product attention. Returns `(out, attn_fp32)`."""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        # Pad positions: mask is (B, T) with 1 for real tokens; broadcast as KV mask.
        mask = (attention_mask == 0)[:, None, None, :]
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_dropped = F.dropout(attn, p=self.attention_dropout, training=self.training)
        out = torch.matmul(attn_dropped.to(v.dtype), v)  # (B, H, T, d_head)
        return out, attn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run dual-path multi-head attention.

        Args:
            x: `(B, T, D)` residual-stream input.
            attention_mask: `(B, T)` tensor with `1` at real tokens, `0` at pads.
            output_attentions: When `True`, return the fp32 attention weights;
                forces the manual fallback (the fast path returns none).

        Returns:
            A `(output, attn_weights_or_None)` tuple. `output` has shape
            `(B, T, D)`; `attn_weights_or_None` has shape `(B, H, T, T)` in fp32
            when requested, otherwise `None`.
        """
        q, k, v = self._project_qkv(x)
        q, k = self._qk_norm(q, k)
        v = self._apply_v_norm(v)
        q, k = self._apply_rope(q, k)

        if self._use_fast_path(output_attentions, q.device):
            out = self._flex_attention(q, k, v, attention_mask)
            attn = None
        else:
            out, attn = self._manual_attention(q, k, v, attention_mask)

        out = self._output_projection(out)
        out = F.dropout(out, p=self.hidden_dropout, training=self.training)
        return out, attn

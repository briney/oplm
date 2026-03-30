"""Grouped-query attention with configurable sub-features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from oplm.model.norm import RMSNorm
from oplm.model.rope import PartialRotaryEmbedding, RotaryEmbedding

if TYPE_CHECKING:
    from oplm.config import ModelConfig

logger = logging.getLogger(__name__)

# Select FlashAttention backend once at import time
_flash_attn_func = None
try:
    from flash_attn import (  # type: ignore[no-redef]
        flash_attn_func as _flash_attn_func,
    )

    logger.info("Using flash_attn backend for attention")
except ImportError:
    logger.info("flash_attn not available, using PyTorch SDPA backend")


class Attention(nn.Module):
    """Grouped-query attention supporting all optional sub-features.

    All optional features are resolved at ``__init__`` time — no dynamic
    branching in ``forward()``. Disabled features create no parameters.
    The ``need_weights`` flag is an exception: it is a per-call runtime choice
    (typically False for training, True for eval) following the convention of
    ``torch.nn.MultiheadAttention``.

    Features (all togglable via config):
        - GQA (``num_kv_heads < num_heads``)
        - Shared K/V projection (``shared_kv``)
        - Q/K RMSNorm (``qk_norm``)
        - Partial or full RoPE (``partial_rope``)
        - Output gating — static or query-dependent (``output_gate``)
        - Post-SDPA normalization (``post_sdpa_norm``)
        - Cross-layer value residuals (``value_residual``)
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim or config.hidden_dim // config.num_heads
        self.gqa_ratio = config.num_heads // config.num_kv_heads
        self.hidden_dim = config.hidden_dim

        q_dim = config.num_heads * self.head_dim
        kv_dim = config.num_kv_heads * self.head_dim

        # Projections
        self.q_proj = nn.Linear(config.hidden_dim, q_dim, bias=False)
        if config.shared_kv:
            self.kv_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
        else:
            self.k_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
        self.shared_kv = config.shared_kv
        self.o_proj = nn.Linear(q_dim, config.hidden_dim, bias=False)

        # Q/K normalization
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None

        # Rotary embeddings
        self.rope: RotaryEmbedding | PartialRotaryEmbedding
        if config.partial_rope:
            self.rope = PartialRotaryEmbedding(
                rope_dim=config.rope_dim or 32,
                nope_dim=config.nope_dim or (self.head_dim - 32),
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        else:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )

        # Output gating
        self.output_gate = config.output_gate
        if config.output_gate:
            if config.query_dependent_gate:
                self.gate_proj = nn.Linear(config.hidden_dim, config.num_heads, bias=False)
            else:
                self.gate_param = nn.Parameter(torch.zeros(config.num_heads))
            self.query_dependent_gate = config.query_dependent_gate
        else:
            self.query_dependent_gate = False

        # Post-SDPA normalization
        self.post_sdpa_norm = (
            RMSNorm(self.head_dim, config.norm_eps) if config.post_sdpa_norm else None
        )

        # Cross-layer value residuals
        self.value_residual = config.value_residual
        if config.value_residual and layer_idx > 0:
            init = config.value_residual_lambda_init
            self.value_lambda = nn.Parameter(torch.tensor([init, -init], dtype=torch.float32))

    def forward(
        self,
        x: Tensor,
        v_first: Tensor | None = None,
        attention_mask: Tensor | None = None,
        value_embed: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Compute attention.

        Args:
            x: Hidden state ``(B, T, D)``.
            v_first: First layer's V for cross-layer value residuals.
            attention_mask: Normalized 4D attention mask. Boolean masks are
                interpreted as keep-masks; floating masks are interpreted as
                additive bias terms.
            value_embed: Gated value embedding ``(B, T, KV_H, D_head)`` from ValueEmbedding.
            need_weights: If True, compute and return per-head attention weights
                using manual attention (bypasses FlashAttention/SDPA). Default False.

        Returns:
            Tuple of (output ``(B, T, D)``, v_first_out, attn_weights).
            ``v_first_out`` is non-None only at layer 0 when value_residual is enabled.
            ``attn_weights`` is ``(B, H, T, T)`` when need_weights=True, else None.
        """
        B, T, _ = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        if self.shared_kv:
            kv = self.kv_proj(x)  # (B, T, num_kv_heads * head_dim)
            k = v = kv
        else:
            k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
            v = self.v_proj(x)

        # 2. Reshape to (B, T, H, D_head)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # 3. Q/K normalization
        if self.q_norm is not None:
            assert self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4. Apply RoPE
        q, k = self.rope(q, k)

        # 5. GQA expansion
        if self.gqa_ratio > 1:
            k = k.repeat_interleave(self.gqa_ratio, dim=2)  # (B, T, H, D_head)
            v = v.repeat_interleave(self.gqa_ratio, dim=2)

        # 6. Value embeddings
        if value_embed is not None:
            if self.gqa_ratio > 1:
                value_embed = value_embed.repeat_interleave(self.gqa_ratio, dim=2)
            v = v + value_embed

        # 7. Cross-layer value residuals
        v_first_out: Tensor | None = None
        if self.value_residual:
            if self.layer_idx == 0:
                v_first_out = v.detach().clone()
            elif v_first is not None:
                lambdas = torch.sigmoid(self.value_lambda)  # (2,)
                lambda_v, lambda_first = lambdas[0], lambdas[1]
                v = lambda_v * v + lambda_first * v_first

        # 8. Compute attention
        attn_out, attn_weights = self._attention(  # (B, T, H, D_head)
            q, k, v, attention_mask, need_weights
        )

        # 9. Post-SDPA normalization
        if self.post_sdpa_norm is not None:
            attn_out = self.post_sdpa_norm(attn_out)

        # 10. Output gating
        if self.output_gate:
            if self.query_dependent_gate:
                gate = torch.sigmoid(self.gate_proj(x))  # (B, T, H)
                gate = gate.unsqueeze(-1)  # (B, T, H, 1)
            else:
                gate = torch.sigmoid(self.gate_param)  # (H,)
                gate = gate.view(1, 1, -1, 1)  # (1, 1, H, 1)
            attn_out = gate * attn_out

        # 11. Output projection
        attn_out = attn_out.reshape(B, T, -1)  # (B, T, num_heads * head_dim)
        output = self.o_proj(attn_out)  # (B, T, D)

        return output, v_first_out, attn_weights

    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor | None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Dispatch to manual attention, FlashAttention, or PyTorch SDPA.

        Args:
            q, k, v: Tensors of shape ``(B, T, H, D_head)``.
            attention_mask: Optional normalized 4D mask.
            need_weights: If True, compute attention manually and return
                per-head weight matrix. Bypasses FlashAttention/SDPA.

        Returns:
            Tuple of (attention output ``(B, T, H, D_head)``, attn_weights).
            ``attn_weights`` is ``(B, H, T, T)`` when need_weights=True, else None.
        """
        # Manual path: materialize attention weights for eval metrics
        if need_weights:
            q = q.transpose(1, 2)  # (B, H, T, D_head)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = q.size(-1) ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
                else:
                    scores = scores + attention_mask.to(dtype=scores.dtype)
            weights = scores.softmax(dim=-1)  # (B, H, T, T)
            attn_out = torch.matmul(weights, v)  # (B, H, T, D_head)
            return attn_out.transpose(1, 2), weights  # (B, T, H, D_head), (B, H, T, T)

        # FlashAttention path
        if _flash_attn_func is not None and attention_mask is None:
            # flash_attn expects (B, T, H, D) layout and returns same
            return _flash_attn_func(q, k, v, causal=False), None

        # PyTorch SDPA path
        q = q.transpose(1, 2)  # (B, H, T, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        )
        return attn_out.transpose(1, 2), None  # (B, T, H, D_head)

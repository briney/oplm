"""Multi-head self-attention: SDPA compute path with a manual softmax for attention weights."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .norm import make_norm
from .rope import RotaryEmbedding

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["OplmAttention"]


class OplmAttention(nn.Module):
    """Multi-head self-attention with an SDPA compute path and a manual-softmax path.

    Both paths share one set of parameters and one set of pre-attention
    transformations (Q/K/V projections, optional QK/V norm, RoPE); only the
    attention kernel differs. The default path calls
    `torch.nn.functional.scaled_dot_product_attention` with a `(B, 1, 1, T)`
    boolean key-padding mask, dispatching to a fused FlashAttention / memory-
    efficient kernel on CUDA (and the math backend on CPU). When
    `output_attentions=True` the manual scaled-dot-product softmax runs instead,
    since SDPA does not expose the attention weights.

    Norm wiring (controlled by `config.norm_strategy`, `config.qk_norm`,
    `config.qk_norm_mode`):

    * `qk_norm=True, qk_norm_mode="channel"` (default) installs per-head
      `q_norm`/`k_norm` modules on `d_head` and uses the canonical
      `1/sqrt(d_head)` attention-kernel scale.
    * `qk_norm=True, qk_norm_mode="l2"` instead L2-normalizes Q and K over
      `d_head` in fp32 and multiplies Q by a learned per-head scale
      (`qk_l2_scale`, shape `(num_attention_heads,)`); the attention kernel then
      runs with scale `1.0` (the score scale is folded into `qk_l2_scale`).
    * `qk_norm=False` leaves Q and K unnormalized (Identity) with the
      `1/sqrt(d_head)` kernel scale.
    * Under `norm_strategy == "hybrid"`, an additional `v_norm` is installed on
      `d_head` — this realises the paper's "QKV-norm" main method.
    * Under every other strategy, `v_norm` is `nn.Identity()`.

    Output gating (controlled by `config.attn_output_gate`): when `"sigmoid"` or
    `"silu"`, a `gate_proj` linear computes an elementwise, head-specific gate
    from the attention input — the post-SDPA G1 multiplicative gate of "Gated
    Attention for Large Language Models" (arXiv:2505.06708) — and the merged
    attention output is multiplied by `act(gate_proj(x))` before `o_proj`.
    `gate_proj` uses the standard trunc-normal init (as in the paper), so at
    init the gate sits near `act(0)` (0.5 for sigmoid, 0.0 for SiLU) and
    attenuates the attention write. `"none"` (default) adds no parameters.

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
        self.norm_strategy = config.norm_strategy
        self.qk_norm_enabled = bool(config.qk_norm)
        self.qk_norm_mode = getattr(config, "qk_norm_mode", "channel")
        # True L2 QK-norm: a learned per-head temperature replaces the channel
        # norm + fixed score scale.
        self.qk_l2 = self.qk_norm_enabled and self.qk_norm_mode == "l2"

        # Attention-kernel score scale. L2 mode folds the scale into the learned
        # per-head multiplier on Q, so the kernel runs at 1.0; every other path
        # keeps the canonical 1/sqrt(head_dim).
        self.attn_scale = 1.0 if self.qk_l2 else 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Picked up by OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling.
        self.o_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

        # Post-SDPA output gate (arXiv:2505.06708, G1): elementwise multiplicative
        # gate on the merged attention output, computed from the attention input.
        # `gate_proj` is a plain Linear (generic trunc-normal init, regular Muon
        # matrix); `o_proj` remains the sole residual writer.
        self.attn_output_gate = getattr(config, "attn_output_gate", "none")
        if self.attn_output_gate not in {"none", "sigmoid", "silu"}:
            raise ValueError(
                f"Unknown attn_output_gate {self.attn_output_gate!r}; "
                "expected one of 'none', 'sigmoid', 'silu'."
            )
        if self.attn_output_gate != "none":
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Channel-mode QK norm installs per-head norm modules; L2 mode and the
        # disabled path leave them as Identity (L2 normalization is functional).
        if self.qk_norm_enabled and not self.qk_l2:
            self.q_norm: nn.Module = make_norm(config.norm_type, head_dim, eps=config.norm_eps)
            self.k_norm: nn.Module = make_norm(config.norm_type, head_dim, eps=config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # L2 mode's learned per-head scale. Initialized directly (not through the
        # generic _init_weights hook) to qk_norm_l2_scale_init or sqrt(head_dim);
        # 1D so it lands in the no-decay AdamW group.
        if self.qk_l2:
            scale_init_cfg = getattr(config, "qk_norm_l2_scale_init", None)
            scale_init = (
                float(scale_init_cfg) if scale_init_cfg is not None else math.sqrt(head_dim)
            )
            self.qk_l2_scale = nn.Parameter(torch.full((num_heads,), scale_init))

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
        """Apply QK normalization for the configured mode (no-op when `qk_norm=False`).

        Channel mode uses the per-head `q_norm`/`k_norm` modules (their internals
        already cast to fp32 and return the input dtype). L2 mode dispatches to
        `_qk_l2_norm`. When `qk_norm=False`, both norms are `nn.Identity`.
        """
        if self.qk_l2:
            return self._qk_l2_norm(q, k)
        return self.q_norm(q), self.k_norm(k)

    def _qk_l2_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """L2-normalize Q,K over `d_head` in fp32, then scale Q per head.

        Q and K are unit-normalized over the head dimension so each logit is a
        scaled cosine similarity; Q is then multiplied by the learned per-head
        scale `qk_l2_scale`. The accompanying attention kernel uses scale `1.0`.
        Computed in fp32 for stability and returned in the inputs' dtype.

        Shapes: q, k are `(B, H, T, d_head)`; `qk_l2_scale` is `(H,)` broadcast
        as `(1, H, 1, 1)`.
        """
        q_hat = F.normalize(q.float(), p=2.0, dim=-1)
        k_hat = F.normalize(k.float(), p=2.0, dim=-1)
        scale = self.qk_l2_scale.to(torch.float32).view(1, -1, 1, 1)  # (1, H, 1, 1)
        q_hat = q_hat * scale
        return q_hat.to(q.dtype), k_hat.to(k.dtype)

    def _apply_v_norm(self, v: torch.Tensor) -> torch.Tensor:
        return self.v_norm(v)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary.apply_rotary(q, k)

    def _output_projection(self, attn_out: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """Reshape `(B, H, T, d_head)` back to `(B, T, D)`, gate (optional), project.

        `gate_input` is the `(B, T, D)` attention input; with the output gate
        enabled, the merged attention output is multiplied elementwise by
        `act(gate_proj(gate_input))` before `o_proj` (head-specific G1 gating,
        since `D = H * d_head`).
        """
        batch, _, seq_len, _ = attn_out.shape
        merged = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        if self.attn_output_gate != "none":
            act = torch.sigmoid if self.attn_output_gate == "sigmoid" else F.silu
            merged = merged * act(self.gate_proj(gate_input))  # (B, T, D)
        return self.o_proj(merged)

    # ------------------------------------------------------------------
    # Attention kernels
    # ------------------------------------------------------------------

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scaled-dot-product attention via `F.scaled_dot_product_attention`.

        The `(B, T)` padding mask becomes a `(B, 1, 1, T)` boolean *key* mask
        (`True` = attend). Masking only keys keeps every query row non-empty —
        pad-query rows stay finite (and are discarded downstream) — so no row is
        all-masked and the softmax cannot produce NaNs. Dropout is applied inside
        the kernel during training only. The explicit `scale=self.attn_scale`
        matches the manual path (`1/sqrt(d_head)` in channel/disabled modes, `1.0`
        in L2 mode where the scale is folded into `qk_l2_scale`).
        """
        attn_mask = (attention_mask == 1)[:, None, None, :]  # (B, 1, 1, T) bool
        dropout_p = self.attention_dropout if self.training else 0.0
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=self.attn_scale
        )

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Manual scaled-dot-product attention. Returns `(out, attn_fp32)`."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale  # (B, H, T, T)
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
                selects the manual softmax path (SDPA exposes no weights).

        Returns:
            A `(output, attn_weights_or_None)` tuple. `output` has shape
            `(B, T, D)`; `attn_weights_or_None` has shape `(B, H, T, T)` in fp32
            when requested, otherwise `None`.
        """
        q, k, v = self._project_qkv(x)
        q, k = self._qk_norm(q, k)
        v = self._apply_v_norm(v)
        q, k = self._apply_rope(q, k)

        if output_attentions:
            out, attn = self._manual_attention(q, k, v, attention_mask)
        else:
            out = self._sdpa_attention(q, k, v, attention_mask)
            attn = None

        out = self._output_projection(out, x)
        out = F.dropout(out, p=self.hidden_dropout, training=self.training)
        return out, attn

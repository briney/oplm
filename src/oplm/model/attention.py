"""Multi-head self-attention: SDPA compute path with a manual softmax for attention weights."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .conv import CanonConv
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

    Value residual (controlled by `config.value_residual`): the ResFormer
    cross-layer value residual of arXiv:2410.17897. Layer 0 exposes its
    post-V-norm values v1 (third element of the forward return); every later
    layer blends its own values toward them after the V projection:
    `v' = lambda * v + (1 - lambda) * v1`. Under `"learnable"`, `lambda` is a
    per-layer scalar parameter (`value_residual_lambda`, layers > 0 only)
    initialized to `config.value_residual_lambda_init`; under `"fixed"` it is a
    constant buffer. `"none"` (default) adds no parameters and keeps the
    two-element forward return.

    Canon-B (controlled by `config.canon_enabled` + `"B" in
    config.canon_positions`): the "inside attention" Canon conv of "Physics of
    Language Models 4.1" (arXiv:2512.17351), equivalent to Primer's multi-DConv-
    head attention (arXiv:2109.08668). One depthwise `CanonConv` per stream
    (`conv_b_q`/`conv_b_k`/`conv_b_v`) runs on the projected, normed Q/K/V —
    after QK/V-norm and before the value residual and RoPE. The conv output is
    added back to its stream when `canon_residual=True` (default). `"none"` (the
    default empty `canon_positions`) adds no parameters.

    The output projection (`o_proj`) is marked `_is_residual_writer = True` so
    the `OplmPreTrainedModel._init_weights` hook can apply the
    `1/sqrt(2L)` residual-stream scaling defined in §15.1 of the architecture
    doc.
    """

    def __init__(self, config: OplmConfig, layer_idx: int = 0) -> None:
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

        # ResFormer value residual (arXiv:2410.17897): layer 0 exposes its values,
        # later layers blend toward them. The mixing scalar lives only on layers
        # > 0; "none" adds no parameters or buffers. Initialized directly at
        # construction (like qk_l2_scale), not through _init_weights; 1D, so it
        # lands in the no-decay AdamW group.
        self.layer_idx = layer_idx
        self.value_residual = getattr(config, "value_residual", "none")
        if self.value_residual not in {"none", "fixed", "learnable"}:
            raise ValueError(
                f"Unknown value_residual {self.value_residual!r}; "
                "expected one of 'none', 'fixed', 'learnable'."
            )
        self.value_residual_enabled = self.value_residual != "none"
        if self.value_residual_enabled and layer_idx > 0:
            lam_init = float(getattr(config, "value_residual_lambda_init", 0.5))
            if self.value_residual == "learnable":
                self.value_residual_lambda = nn.Parameter(torch.full((1,), lam_init))
            else:  # "fixed" — persistent buffer (compile/DDP + HF fast-init safe)
                self.register_buffer(
                    "value_residual_lambda", torch.tensor(lam_init), persistent=True
                )

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

        # Canon-B (Physics of LM 4.1, arXiv:2512.17351): depthwise convs on Q/K/V
        # after the norms and before RoPE, residual by default. One CanonConv per
        # stream is equivalent to a single conv over concatenated 3*hidden_size
        # channels since the conv is depthwise (no cross-channel mixing).
        # Constructed here (not in OplmBlock) because it must see projected Q/K/V.
        self.canon_b_enabled = bool(getattr(config, "canon_enabled", False)) and (
            "B" in set(config.canon_positions or [])
        )
        self.canon_residual = bool(getattr(config, "canon_residual", True))
        if self.canon_b_enabled:
            kernel_sizes = config.canon_kernel_sizes
            if not isinstance(kernel_sizes, list) or len(kernel_sizes) != config.num_hidden_layers:
                raise ValueError(
                    "config.canon_kernel_sizes must be a list of length num_hidden_layers; "
                    "resolve it via resolve_canon_kernel_sizes() before instantiating "
                    "OplmAttention."
                )
            kernel_size = kernel_sizes[layer_idx]
            activation = getattr(config, "canon_activation", "none")
            self.conv_b_q = CanonConv(hidden_size, kernel_size, activation=activation)
            self.conv_b_k = CanonConv(hidden_size, kernel_size, activation=activation)
            self.conv_b_v = CanonConv(hidden_size, kernel_size, activation=activation)

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

    def _apply_canon_b(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the per-stream Canon-B depthwise conv to Q/K/V.

        Each tensor is `(B, H, T, d_head)`; `CanonConv` operates on a
        `(B, T, D)` view, so the heads are merged before the conv and split back
        after. The conv runs along the time axis per channel, which commutes
        with the head reshape, so this is identical to convolving the flat
        `(B, T, D)` projection output.
        """
        batch, heads, seq_len, head_dim = q.shape

        def conv(t: torch.Tensor, layer: CanonConv) -> torch.Tensor:
            # (B, H, T, d) -> (B, T, D) -> conv -> (B, H, T, d)
            merged = t.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
            convolved = layer(merged, attention_mask)
            merged = merged + convolved if self.canon_residual else convolved
            return merged.view(batch, seq_len, heads, head_dim).transpose(1, 2)

        return conv(q, self.conv_b_q), conv(k, self.conv_b_k), conv(v, self.conv_b_v)

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
        value_residual: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ):
        """Run dual-path multi-head attention.

        Args:
            x: `(B, T, D)` residual-stream input.
            attention_mask: `(B, T)` tensor with `1` at real tokens, `0` at pads.
            output_attentions: When `True`, return the fp32 attention weights;
                selects the manual softmax path (SDPA exposes no weights).
            value_residual: `(B, H, T, d_head)` layer-0 values v1 to blend into
                this layer's values (`value_residual != "none"` and
                `layer_idx > 0` only); `None` disables the blend.

        Returns:
            A `(output, attn_weights_or_None)` tuple. `output` has shape
            `(B, T, D)`; `attn_weights_or_None` has shape `(B, H, T, T)` in fp32
            when requested, otherwise `None`. When the value residual is
            enabled, a third element carries this layer's post-V-norm values at
            `layer_idx == 0` (`(B, H, T, d_head)`) and `None` on later layers.
        """
        q, k, v = self._project_qkv(x)
        q, k = self._qk_norm(q, k)
        v = self._apply_v_norm(v)
        # Canon-B: depthwise conv on the projected, normed Q/K/V before RoPE.
        # Applied before the value residual so layer 0 exposes (and later layers
        # blend toward) the conv'd values.
        if self.canon_b_enabled:
            q, k, v = self._apply_canon_b(q, k, v, attention_mask)
        # ResFormer value residual: blend this layer's values toward layer 0's.
        if value_residual is not None:
            lam = self.value_residual_lambda
            v = lam * v + (1.0 - lam) * value_residual
        q, k = self._apply_rope(q, k)

        if output_attentions:
            out, attn = self._manual_attention(q, k, v, attention_mask)
        else:
            out = self._sdpa_attention(q, k, v, attention_mask)
            attn = None

        out = self._output_projection(out, x)
        out = F.dropout(out, p=self.hidden_dropout, training=self.training)
        if self.value_residual_enabled:
            return out, attn, (v if self.layer_idx == 0 else None)
        return out, attn

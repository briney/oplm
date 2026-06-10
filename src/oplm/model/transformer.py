"""OplmBlock and OplmStack — the repeating block and backbone holder."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils import checkpoint as torch_checkpoint

from .attention import OplmAttention
from .conv import CanonConv
from .embedding import OplmEmbedding
from .ffn import make_ffn
from .masking import prepare_attention_mask
from .norm import make_norm

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["OplmBlock", "OplmStack"]


_CANON_POSITIONS = ("A", "B", "C", "D")


# CheckpointPolicy is public in torch.utils.checkpoint from torch>=2.4; guard the
# import so this module still loads on older torch (the selective branch in
# OplmBlock.forward raises a clear error if SAC is then requested).
try:
    from torch.utils.checkpoint import CheckpointPolicy as _CheckpointPolicy
except ImportError:  # torch < 2.4
    _CheckpointPolicy = None  # ty: ignore[invalid-assignment]  # torch<2.4 fallback


# Selective Activation Checkpointing (SAC) — ops whose outputs are FLOP-heavy to
# recompute but cheap to store. Saving these avoids re-running the expensive
# kernels during the backward recompute, while everything else (norms, SiLU/GELU,
# RoPE, softmax, dropout, residual scaling, reshapes) is recomputed to free
# memory. See docs/TRAIN.md for the memory/compute tradeoff rationale.
def _build_sac_save_ops() -> frozenset:
    """Collect the aten overloads SAC keeps in memory (matmuls + SDPA backends).

    Built lazily through getattr so a missing SDPA overload on an older torch
    simply drops out of the set (those ops then fall to PREFER_RECOMPUTE) rather
    than raising at import time.
    """
    aten = torch.ops.aten
    candidates = (
        "mm",  # Linear (2D) / matmuls
        "addmm",  # Linear with bias (MLM head, etc.)
        "bmm",  # batched matmul (manual-attention path)
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_attention_math",  # CPU / math backend
    )
    ops = set()
    for name in candidates:
        overload_packet = getattr(aten, name, None)
        if overload_packet is not None:
            ops.add(overload_packet.default)
    return frozenset(ops)


_SAC_SAVE_OPS = _build_sac_save_ops()


def _sac_policy_fn(ctx, op, *args, **kwargs):
    """SAC policy: keep matmul/SDPA outputs, recompute everything else.

    Defined at module level (not a per-call closure) so torch.compile/Dynamo can
    trace it as a constant global when composed with the checkpoint higher-order
    op — a nested closure here trips ``AsPythonConstantNotImplementedError``.
    """
    if op in _SAC_SAVE_OPS:
        return _CheckpointPolicy.MUST_SAVE
    return _CheckpointPolicy.PREFER_RECOMPUTE


class OplmBlock(nn.Module):
    """One repeating encoder block: attention + FFN sublayers, configurable norms.

    Wires the four `norm_strategy` variants (`pre`, `sandwich`, `hybrid`,
    `post_sdpa`), optional Canon depthwise convolutions at positions A/B/C/D,
    the residual-stream scaling factor `alpha`, and an opt-in gradient
    checkpoint dispatch.
    """

    # `alpha` is registered as a persistent buffer in __init__ (via the string
    # name, invisible to the type checker); declare its type so `self.alpha`
    # resolves to Tensor rather than the `nn.Module.__getattr__` union.
    alpha: torch.Tensor

    def __init__(self, config: OplmConfig, layer_idx: int) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.norm_strategy = config.norm_strategy
        self.residual_scaling = config.residual_scaling
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.gradient_checkpointing_mode = str(
            getattr(config, "gradient_checkpointing_mode", "full")
        )

        if config.residual_scaling == "sqrt_num_layers":
            alpha_val = 1.0 / math.sqrt(config.num_hidden_layers)
        elif config.residual_scaling == "none":
            alpha_val = 1.0
        else:
            raise ValueError(
                f"Unknown residual_scaling {config.residual_scaling!r}; "
                "expected 'sqrt_num_layers' or 'none'."
            )
        # Register as a persistent buffer (scalar tensor) rather than a plain Python float.
        #
        # Why a buffer at all: torch.compile + DDP (DDPOptimizer) lifts plain-float
        # module attributes as graph inputs and may place them in subgraph outputs when
        # partitioning at bucket boundaries. aot_autograd then fails with
        # "AttributeError: 'float' has no attribute 'meta'" because it expects every
        # output value to be an FX Node. A buffer is a proper tensor throughout.
        #
        # Why persistent (not persistent=False): HuggingFace's from_pretrained fast-init
        # path creates model tensors uninitialized (torch.empty semantics) and only
        # restores persistent buffers from the saved state dict. Non-persistent buffers
        # stay as garbage after loading, producing near-zero alpha and broken outputs.
        self.register_buffer("alpha", torch.tensor(alpha_val), persistent=True)

        # Optional learnable residual gates: multiplicative refinements applied on
        # top of the fixed `alpha` scale at each residual write. `scalar` adds one
        # parameter per sublayer write; `channel` adds a `(hidden_size,)` vector.
        # Initialized directly to `residual_gate_init` (not via _init_weights, which
        # only matches Linear/Embedding/Conv1d/norm modules). 1D so they land in the
        # no-decay AdamW group, including the auxiliary AdamW path under Muon.
        self.residual_gate = getattr(config, "residual_gate", "none")
        if self.residual_gate not in {"none", "scalar", "channel"}:
            raise ValueError(
                f"Unknown residual_gate {self.residual_gate!r}; "
                "expected one of 'none', 'scalar', 'channel'."
            )
        if self.residual_gate != "none":
            gate_init = float(getattr(config, "residual_gate_init", 1.0))
            gate_shape = (1,) if self.residual_gate == "scalar" else (config.hidden_size,)
            self.attn_gate = nn.Parameter(torch.full(gate_shape, gate_init))
            self.ffn_gate = nn.Parameter(torch.full(gate_shape, gate_init))

        if config.norm_strategy not in {"pre", "sandwich", "hybrid", "post_sdpa"}:
            raise ValueError(
                f"Unknown norm_strategy {config.norm_strategy!r}; "
                "expected one of 'pre', 'sandwich', 'hybrid', 'post_sdpa'."
            )

        # Attention pre-norm: present under every strategy except hybrid.
        if config.norm_strategy != "hybrid":
            self.attn_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )

        # Attention module self-configures v_norm under hybrid; the layer index
        # drives the ResFormer value-residual wiring (layer 0 exposes v1).
        self.attention = OplmAttention(config, layer_idx)
        self.value_residual_enabled = getattr(config, "value_residual", "none") != "none"

        # FFN pre-norm and FFN module are always present.
        self.ffn_norm: nn.Module = make_norm(
            config.norm_type, config.hidden_size, eps=config.norm_eps
        )
        self.ffn = make_ffn(config)

        # Strategy-specific post-norms.
        if config.norm_strategy == "sandwich":
            self.attn_post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
            self.ffn_post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
        elif config.norm_strategy == "post_sdpa":
            self.attn_post_norm = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )

        # Canon convs at A/B/C/D: only when canon is enabled and the position is selected.
        if getattr(config, "canon_enabled", False):
            positions = set(config.canon_positions)
            unknown = positions - set(_CANON_POSITIONS)
            if unknown:
                raise ValueError(
                    f"canon_positions contains unknown entries {sorted(unknown)}; "
                    f"expected subset of {list(_CANON_POSITIONS)}."
                )
            kernel_sizes = config.canon_kernel_sizes
            if not isinstance(kernel_sizes, list) or len(kernel_sizes) != config.num_hidden_layers:
                raise ValueError(
                    "config.canon_kernel_sizes must be a list of length num_hidden_layers; "
                    "resolve it via resolve_canon_kernel_sizes() before instantiating OplmBlock."
                )
            kernel_size = kernel_sizes[layer_idx]
            activation = getattr(config, "canon_activation", "none")
            if "A" in positions:
                self.conv_a = CanonConv(config.hidden_size, kernel_size, activation=activation)
            if "B" in positions:
                self.conv_b = CanonConv(config.hidden_size, kernel_size, activation=activation)
            if "C" in positions:
                self.conv_c = CanonConv(config.hidden_size, kernel_size, activation=activation)
            if "D" in positions:
                self.conv_d = CanonConv(config.hidden_size, kernel_size, activation=activation)

    def _gate_attn(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Apply the learned attention residual gate (no-op when `residual_gate="none"`)."""
        if self.residual_gate == "none":
            return attn_out
        return self.attn_gate * attn_out

    def _gate_ffn(self, ffn_out: torch.Tensor) -> torch.Tensor:
        """Apply the learned FFN residual gate (no-op when `residual_gate="none"`)."""
        if self.residual_gate == "none":
            return ffn_out
        return self.ffn_gate * ffn_out

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool,
        value_residual: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ):
        # Canon A: additive into the residual stream (preserves the residual identity).
        if hasattr(self, "conv_a"):
            x = x + self.conv_a(x, attention_mask)

        # Attention sublayer. Hybrid feeds raw `x` (QKV-norm lives inside the
        # attention module); every other strategy applies the outer pre-norm.
        a_in = x if self.norm_strategy == "hybrid" else self.attn_norm(x)
        if hasattr(self, "conv_b"):
            a_in = self.conv_b(a_in, attention_mask)

        attn_result = self.attention(
            a_in, attention_mask, output_attentions, value_residual=value_residual
        )
        if self.value_residual_enabled:
            attn_out, attn_weights, v1 = attn_result
        else:
            attn_out, attn_weights = attn_result
            v1 = None
        if hasattr(self, "conv_c"):
            attn_out = self.conv_c(attn_out, attention_mask)

        if self.norm_strategy in {"sandwich", "post_sdpa"}:
            attn_out = self.attn_post_norm(attn_out)
        h = x + self.alpha * self._gate_attn(attn_out)

        # FFN sublayer.
        h_norm = self.ffn_norm(h)
        f_in = h_norm
        if hasattr(self, "conv_d"):
            f_in = self.conv_d(f_in, attention_mask)
        ffn_out = self.ffn(f_in)

        if self.norm_strategy == "sandwich":
            ffn_out = self.ffn_post_norm(ffn_out)
            y = h + self.alpha * self._gate_ffn(ffn_out)
        elif self.norm_strategy == "hybrid":
            # Hybrid reuses Norm(h) as both FFN input and FFN-side residual stream.
            y = h_norm + self.alpha * self._gate_ffn(ffn_out)
        else:  # "pre" or "post_sdpa"
            y = h + self.alpha * self._gate_ffn(ffn_out)

        if self.value_residual_enabled:
            return y, attn_weights, v1
        return y, attn_weights

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
        """Run one transformer block.

        Args:
            x: `(B, T, D)` residual-stream input.
            attention_mask: `(B, T)` mask with `1` at real tokens, `0` at pads.
            output_attentions: When `True`, return the per-block attention
                weights from `OplmAttention` (forces the SDPA fallback).
            value_residual: `(B, H, T, d_head)` layer-0 values v1 for the
                ResFormer value residual (later layers only); `None` otherwise.

        Returns:
            `(y, attn_weights_or_None)` — `y` has shape `(B, T, D)`;
            `attn_weights_or_None` has shape `(B, H, T, T)` when requested.
            When the value residual is enabled, a third element carries the
            layer-0 values v1 (`layer_idx == 0` only, else `None`).
        """
        if self.gradient_checkpointing and self.training:
            if self.gradient_checkpointing_mode == "selective":
                if not hasattr(torch_checkpoint, "create_selective_checkpoint_contexts"):
                    raise RuntimeError(
                        "Selective activation checkpointing requires torch>=2.4; "
                        "set gradient_checkpointing_mode='full'."
                    )
                context_fn = functools.partial(
                    torch_checkpoint.create_selective_checkpoint_contexts,
                    _sac_policy_fn,
                )
                return torch_checkpoint.checkpoint(
                    self._forward_impl,
                    x,
                    attention_mask,
                    output_attentions,
                    value_residual,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
            # "full" — recompute the entire block on backward.
            return torch_checkpoint.checkpoint(
                self._forward_impl,
                x,
                attention_mask,
                output_attentions,
                value_residual,
                use_reentrant=False,
            )
        return self._forward_impl(x, attention_mask, output_attentions, value_residual)


class OplmStack(nn.Module):
    """Encoder backbone: token embedding, N × OplmBlock, final norm.

    Forward returns `(last_hidden, hidden_states_or_None, attentions_or_None)`:

    * `last_hidden`: `(B, T, D)` post-final-norm activations.
    * `hidden_states`: `(L + 1)`-tuple of `(B, T, D)` tensors (the
      post-embedding state, then the output of each block, pre-final-norm).
      `None` when `output_hidden_states=False`.
    * `attentions`: `L`-tuple of `(B, H, T, T)` tensors (each entry is `None`
      when a block returned no weights). `None` when `output_attentions=False`.
    """

    def __init__(self, config: OplmConfig) -> None:
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.gradient_checkpointing_mode = str(
            getattr(config, "gradient_checkpointing_mode", "full")
        )
        self.value_residual_enabled = getattr(config, "value_residual", "none") != "none"

        self.embed_tokens = OplmEmbedding(config)
        self.layers = nn.ModuleList(
            [OplmBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)

    def set_gradient_checkpointing(self, enabled: bool, mode: str | None = None) -> None:
        """Toggle gradient checkpointing on every block in the stack.

        Args:
            enabled: Whether activation checkpointing fires during training.
            mode: Optional `"full"` | `"selective"` flavor. When given, it is
                propagated to the stack and every block; when `None`, the
                existing mode is left untouched.
        """
        self.gradient_checkpointing = enabled
        if mode is not None:
            self.gradient_checkpointing_mode = mode
        for block in self.layers:
            block.gradient_checkpointing = enabled  # ty: ignore[unresolved-attribute]  # nn.Module setattr
            if mode is not None:
                block.gradient_checkpointing_mode = mode  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        tuple[torch.Tensor | None, ...] | None,
    ]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `input_ids` or `inputs_embeds`.")

        # Materialize/validate the pad mask before the embedding lookup so it can
        # be threaded into mask dropout (which needs per-row real-token counts).
        if inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            attention_mask = prepare_attention_mask(
                attention_mask, batch_size, seq_len, inputs_embeds.device
            )
            # The inputs_embeds path bypasses mask dropout: <mask> positions
            # cannot be inferred once token IDs are gone.
            x = inputs_embeds
        else:
            assert input_ids is not None  # guaranteed by the exactly-one check above
            batch_size, seq_len = input_ids.shape
            attention_mask = prepare_attention_mask(
                attention_mask, batch_size, seq_len, input_ids.device
            )
            x = self.embed_tokens(input_ids, attention_mask)

        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states else None
        attentions: tuple[torch.Tensor | None, ...] | None = () if output_attentions else None

        # ResFormer value residual: layer 0 returns its values v1, which are fed
        # to every later block. Stays None (and unused) when disabled.
        v1: torch.Tensor | None = None
        for block in self.layers:
            result = block(x, attention_mask, output_attentions, value_residual=v1)
            if self.value_residual_enabled:
                x, attn, v = result
                if v1 is None:
                    v1 = v
            else:
                x, attn = result
            if hidden_states is not None:
                hidden_states = hidden_states + (x,)
            if attentions is not None:
                attentions = attentions + (attn,)

        last_hidden = self.final_norm(x)
        return last_hidden, hidden_states, attentions

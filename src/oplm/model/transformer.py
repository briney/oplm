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
from .loop_modules import LoopCycleGate, LoopInputInjector, LoopStepConditioner
from .looping import resolve_loop_schedule, validate_loop_extensions
from .masking import prepare_attention_mask, segment_block_mask
from .norm import make_norm

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

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
# memory. See docs/training.md, "What a training step does", for the memory/compute
# tradeoff rationale.
def _build_sac_save_ops() -> frozenset:
    """Collect the ops SAC keeps in memory (matmuls, SDPA backends, FlexAttention).

    Built lazily through getattr so a missing SDPA overload (or a torch without
    FlexAttention's higher-order op) simply drops out of the set (those ops then
    fall to PREFER_RECOMPUTE) rather than raising at import time.
    """
    aten = torch.ops.aten
    candidates = (
        "mm",  # Linear (2D) / matmuls
        "addmm",  # Linear with bias (MLM head, etc.)
        "bmm",  # batched matmul (manual-attention path)
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_attention_math",  # CPU / math backend
    )
    ops = set()
    for name in candidates:
        overload_packet = getattr(aten, name, None)
        if overload_packet is not None:
            ops.add(overload_packet.default)
    # FlexAttention runs as a higher-order op, not an aten overload: without it here every
    # layer's attention is recomputed in backward (torchtitan's SAC list adds it the same way).
    flex = getattr(torch.ops.higher_order, "flex_attention", None)
    if flex is not None:
        ops.add(flex)
    return frozenset(ops)


_SAC_SAVE_OPS = _build_sac_save_ops()


def _sac_policy_fn(ctx, op, *args, **kwargs):
    """SAC policy: keep matmul/attention-kernel outputs, recompute everything else.

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

    def __init__(self, config: OplmConfig, layer_idx: int, *, num_loop_passes: int = 1) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.num_loop_passes = num_loop_passes
        self.loop_untie_norms = bool(getattr(config, "loop_untie_norms", False)) and (
            num_loop_passes > 1
        )
        self.norm_strategy = config.norm_strategy
        self.residual_scaling = config.residual_scaling
        self.canon_residual = bool(getattr(config, "canon_residual", True))
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.gradient_checkpointing_mode = str(
            getattr(config, "gradient_checkpointing_mode", "full")
        )

        if config.residual_scaling == "sqrt_num_layers":
            alpha_val = 1.0 / math.sqrt(config.num_hidden_layers)
        elif config.residual_scaling == "esmc":
            # ESM-C's TransformerStack divides each branch by sqrt(L / 36).
            alpha_val = 1.0 / math.sqrt(config.num_hidden_layers / 36)
        elif config.residual_scaling == "none":
            alpha_val = 1.0
        else:
            raise ValueError(
                f"Unknown residual_scaling {config.residual_scaling!r}; "
                "expected 'sqrt_num_layers', 'esmc', or 'none'."
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
            if self.loop_untie_norms:
                self.loop_attn_norms = nn.ModuleList(
                    [
                        make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)
                        for _ in range(num_loop_passes - 1)
                    ]
                )

        # Attention module self-configures v_norm under hybrid; the layer index
        # drives the ResFormer value-residual wiring (layer 0 exposes v1).
        self.attention = OplmAttention(config, layer_idx, num_loop_passes=num_loop_passes)
        self.value_residual_enabled = getattr(config, "value_residual", "none") != "none"

        # FFN pre-norm and FFN module are always present.
        self.ffn_norm: nn.Module = make_norm(
            config.norm_type, config.hidden_size, eps=config.norm_eps
        )
        if self.loop_untie_norms:
            self.loop_ffn_norms = nn.ModuleList(
                [
                    make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)
                    for _ in range(num_loop_passes - 1)
                ]
            )
        self.ffn = make_ffn(config, layer_idx)

        # Strategy-specific post-norms.
        if config.norm_strategy == "sandwich":
            self.attn_post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
            self.ffn_post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
            if self.loop_untie_norms:
                self.loop_attn_post_norms = nn.ModuleList(
                    [
                        make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)
                        for _ in range(num_loop_passes - 1)
                    ]
                )
                self.loop_ffn_post_norms = nn.ModuleList(
                    [
                        make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)
                        for _ in range(num_loop_passes - 1)
                    ]
                )
        elif config.norm_strategy == "post_sdpa":
            self.attn_post_norm = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
            if self.loop_untie_norms:
                self.loop_attn_post_norms = nn.ModuleList(
                    [
                        make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)
                        for _ in range(num_loop_passes - 1)
                    ]
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
            # A acts on the attention pre-norm stream, C on the FFN pre-norm
            # stream (paper-exact insertion sites; see docs/architecture.md, Canon semantics).
            if "A" in positions:
                self.conv_a = CanonConv(config.hidden_size, kernel_size, activation=activation)
            if "C" in positions:
                self.conv_c = CanonConv(config.hidden_size, kernel_size, activation=activation)
            # Canon-B lives inside OplmAttention (it convolves the projected
            # Q/K/V); Canon-D lives inside the FFN (it convolves the activation
            # branch at intermediate_size). Neither is instantiated here.

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

    def _select_norm(
        self, base: nn.Module, extras: nn.ModuleList | None, pass_idx: int
    ) -> nn.Module:
        if pass_idx > 0 and extras is not None:
            return extras[pass_idx - 1]
        return base

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool,
        value_residual: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        block_mask: BlockMask | None = None,
        return_qk: bool = False,
        pass_idx: int = 0,
    ) -> tuple[torch.Tensor | None, ...]:
        # Attention sublayer. Hybrid feeds raw `x` (QKV-norm lives inside the
        # attention module); every other strategy applies the outer pre-norm.
        # Canon is restricted to norm_strategy="pre", so when conv_a exists
        # `attn_base` is always the normed stream.
        if self.norm_strategy == "hybrid":
            attn_base = x
        else:
            attn_norm = self._select_norm(
                self.attn_norm,
                self.loop_attn_norms if self.loop_untie_norms else None,
                pass_idx,
            )
            attn_base = attn_norm(x)

        # Canon-A: residual conv on the attention pre-norm stream (paper-exact);
        # Canon-B (the conv on Q/K/V) is applied inside the attention module.
        attn_in = attn_base
        if hasattr(self, "conv_a"):
            attn_in = self.conv_a(
                attn_base, attention_mask, segment_ids, residual=self.canon_residual
            )

        attn_result = self.attention(
            attn_in,
            attention_mask,
            output_attentions,
            value_residual=value_residual,
            segment_ids=segment_ids,
            block_mask=block_mask,
            return_qk=return_qk,
            pass_idx=pass_idx,
        )
        q_out: torch.Tensor | None = None
        k_out: torch.Tensor | None = None
        if self.value_residual_enabled:
            if return_qk:
                attn_out, attn_weights, v1, q_out, k_out = attn_result
            else:
                attn_out, attn_weights, v1 = attn_result
        else:
            v1 = None
            if return_qk:
                attn_out, attn_weights, q_out, k_out = attn_result
            else:
                attn_out, attn_weights = attn_result
        assert attn_out is not None

        if self.norm_strategy in {"sandwich", "post_sdpa"}:
            attn_post_norm = self._select_norm(
                self.attn_post_norm,
                self.loop_attn_post_norms if self.loop_untie_norms else None,
                pass_idx,
            )
            attn_out = attn_post_norm(attn_out)
        h = x + self.alpha * self._gate_attn(attn_out)

        # FFN sublayer.
        ffn_norm = self._select_norm(
            self.ffn_norm,
            self.loop_ffn_norms if self.loop_untie_norms else None,
            pass_idx,
        )
        h_norm = ffn_norm(h)
        # Canon-C: residual conv on the FFN pre-norm stream (paper-exact);
        # Canon-D runs inside the FFN on the activation branch at intermediate
        # width and is applied by the FFN module itself.
        mlp_in = h_norm
        if hasattr(self, "conv_c"):
            mlp_in = self.conv_c(h_norm, attention_mask, segment_ids, residual=self.canon_residual)
        ffn_out = self.ffn(mlp_in, attention_mask, segment_ids)

        if self.norm_strategy == "sandwich":
            ffn_post_norm = self._select_norm(
                self.ffn_post_norm,
                self.loop_ffn_post_norms if self.loop_untie_norms else None,
                pass_idx,
            )
            ffn_out = ffn_post_norm(ffn_out)
            y = h + self.alpha * self._gate_ffn(ffn_out)
        elif self.norm_strategy == "hybrid":
            # Hybrid reuses Norm(h) as both FFN input and FFN-side residual stream.
            y = h_norm + self.alpha * self._gate_ffn(ffn_out)
        else:  # "pre" or "post_sdpa"
            y = h + self.alpha * self._gate_ffn(ffn_out)

        if self.value_residual_enabled:
            if return_qk:
                return y, attn_weights, v1, q_out, k_out
            return y, attn_weights, v1
        if return_qk:
            return y, attn_weights, q_out, k_out
        return y, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        value_residual: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        block_mask: BlockMask | None = None,
        return_qk: bool = False,
        pass_idx: int = 0,
    ) -> tuple[torch.Tensor | None, ...]:
        """Run one transformer block.

        Args:
            x: `(B, T, D)` residual-stream input.
            attention_mask: `(B, T)` mask with `1` at real tokens, `0` at pads.
            output_attentions: When `True`, return the per-block attention
                weights from `OplmAttention` (forces the SDPA fallback).
            value_residual: `(B, H, T, d_head)` layer-0 values v1 for the
                ResFormer value residual (later layers only); `None` otherwise.
            segment_ids: `(B, T)` segment ids of a packed batch, or `None` (see `OplmAttention`).
            block_mask: FlexAttention block mask shared by every block of this forward, or `None`.
            return_qk: When `True`, also return post-RoPE `q, k` tensors from `OplmAttention`.
            pass_idx: Zero-based core loop pass index (used when `loop_untie_norms=True`).

        Returns:
            Flat tuple starting with `(y, attn_weights_or_None)`, followed by
            `v1_or_None` (when `value_residual` is enabled) and/or `q, k` (when `return_qk=True`).
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
                    segment_ids,
                    block_mask,
                    return_qk,
                    pass_idx,
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
                segment_ids,
                block_mask,
                return_qk,
                pass_idx,
                use_reentrant=False,
            )
        return self._forward_impl(
            x,
            attention_mask,
            output_attentions,
            value_residual,
            segment_ids,
            block_mask,
            return_qk,
            pass_idx,
        )


class OplmStack(nn.Module):
    """Encoder backbone: embedding, shared-block execution schedule, final norm.

    Forward returns `(last_hidden, hidden_states_or_None, attentions_or_None)`:

    D is executed depth; physical blocks remain registered exactly once.
    `layer_execution_order` maps each execution to its physical block index.

    * `last_hidden`: `(B, T, D)` post-final-norm activations.
    * `hidden_states`: `(D + 1)`-tuple of `(B, T, D)` tensors (the
      post-embedding state, then the output of each block, pre-final-norm).
      `None` when `output_hidden_states=False`.
    * `attentions`: `D`-tuple of `(B, H, T, T)` tensors (each entry is `None`
      when a block returned no weights). `None` when `output_attentions=False`.
    """

    def __init__(self, config: OplmConfig) -> None:
        super().__init__()
        self.config = config
        num_loops = getattr(config, "num_loops", 1)
        loop_strategy = getattr(config, "loop_strategy", "stack")
        self.loop_schedule = resolve_loop_schedule(
            config.num_hidden_layers,
            num_loops=num_loops,
            loop_strategy=loop_strategy,
            loop_start=getattr(config, "loop_start", 0),
            loop_end=getattr(config, "loop_end", None),
        )
        self.layer_execution_order = self.loop_schedule.execution_order
        loop_input_injection = getattr(config, "loop_input_injection", "none")
        loop_gate = getattr(config, "loop_gate", "none")
        loop_gate_init = getattr(config, "loop_gate_init", 1.0)
        loop_untie_norms = getattr(config, "loop_untie_norms", False)
        loop_step_conditioning = getattr(config, "loop_step_conditioning", "none")
        self._has_loop_extensions = validate_loop_extensions(
            num_loops=num_loops,
            loop_strategy=loop_strategy,
            loop_input_injection=loop_input_injection,
            loop_gate=loop_gate,
            loop_gate_init=loop_gate_init,
            loop_untie_norms=loop_untie_norms,
            loop_step_conditioning=loop_step_conditioning,
        )

        self.num_hidden_layers = config.num_hidden_layers
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.gradient_checkpointing_mode = str(
            getattr(config, "gradient_checkpointing_mode", "full")
        )
        self.value_residual_enabled = getattr(config, "value_residual", "none") != "none"

        self.embed_tokens = OplmEmbedding(config)
        core_set = set(self.loop_schedule.core)
        self.layers = nn.ModuleList(
            [
                OplmBlock(
                    config,
                    layer_idx,
                    num_loop_passes=(
                        num_loops if loop_strategy == "stack" and layer_idx in core_set else 1
                    ),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.loop_injector: LoopInputInjector | None = (
            LoopInputInjector(
                loop_input_injection,
                config.hidden_size,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
            )
            if loop_input_injection != "none"
            else None
        )
        self.loop_step_cond: LoopStepConditioner | None = (
            LoopStepConditioner(
                config.hidden_size,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
            )
            if loop_step_conditioning == "sinusoidal"
            else None
        )
        self.loop_gate_module: LoopCycleGate | None = (
            LoopCycleGate(
                loop_gate,
                num_loops,
                config.hidden_size,
                init_val=float(loop_gate_init),
            )
            if loop_gate != "none"
            else None
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
        segment_ids: torch.Tensor | None = None,
        return_qk_layers: tuple[int, ...] | None = None,
        return_hidden_layers: tuple[int, ...] | None = None,
    ) -> (
        tuple[
            torch.Tensor,
            tuple[torch.Tensor, ...] | None,
            tuple[torch.Tensor | None, ...] | None,
        ]
        | tuple[
            torch.Tensor,
            tuple[torch.Tensor, ...] | None,
            tuple[torch.Tensor | None, ...] | None,
            tuple[tuple[torch.Tensor, torch.Tensor], ...],
        ]
    ):
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

        if output_hidden_states and return_hidden_layers is not None:
            raise ValueError("Cannot pass both output_hidden_states=True and return_hidden_layers")
        effective_depth = len(self.layer_execution_order)
        if return_hidden_layers is not None:
            invalid_h = [idx for idx in return_hidden_layers if idx < 0 or idx >= effective_depth]
            if invalid_h:
                raise ValueError(
                    f"return_hidden_layers indices {invalid_h} out of range [0, {effective_depth})"
                )
            hidden_layer_set: set[int] | None = set(return_hidden_layers)
            tapped_hidden: list[torch.Tensor] | None = []
            hidden_states: tuple[torch.Tensor, ...] | None = None
        else:
            hidden_layer_set = None
            tapped_hidden = None
            hidden_states = (x,) if output_hidden_states else None

        attentions: tuple[torch.Tensor | None, ...] | None = () if output_attentions else None
        if return_qk_layers is not None:
            invalid = [idx for idx in return_qk_layers if idx < 0 or idx >= effective_depth]
            if invalid:
                raise ValueError(
                    f"return_qk_layers indices {invalid} out of range [0, {effective_depth})"
                )
            qk_layer_set: set[int] | None = set(return_qk_layers)
        else:
            qk_layer_set = None
        qk_states: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if qk_layer_set is not None else None
        )

        # ResFormer: retain the first execution of layer 0 as the global value
        # reference, including when that physical block is revisited.
        # Stays None (and unused) when disabled.
        v1: torch.Tensor | None = None
        # One FlexAttention block mask per forward, shared by every block (and every loop
        # revisit); read from config at call time so tools can switch kernels on a live model.
        block_mask = None
        if (
            segment_ids is not None
            and not output_attentions
            and getattr(self.config, "packed_attention", "sdpa") == "flex"
        ):
            block_mask = segment_block_mask(segment_ids)

        def _run_block(
            cur_x: torch.Tensor,
            exec_idx: int,
            layer_index: int,
            pass_idx: int,
            *,
            gate_prev: torch.Tensor | None = None,
            gate_idx: int | None = None,
        ) -> torch.Tensor:
            nonlocal v1, hidden_states, attentions
            block = self.layers[layer_index]
            want_qk = qk_layer_set is not None and exec_idx in qk_layer_set
            result = block(
                cur_x,
                attention_mask,
                output_attentions,
                value_residual=v1,
                segment_ids=segment_ids,
                block_mask=block_mask,
                return_qk=want_qk,
                pass_idx=pass_idx,
            )
            if self.value_residual_enabled:
                if want_qk:
                    next_x, attn, v, q_out, k_out = result
                else:
                    next_x, attn, v = result
                if v1 is None:
                    v1 = v
            else:
                if want_qk:
                    next_x, attn, q_out, k_out = result
                else:
                    next_x, attn = result
            assert next_x is not None
            if gate_prev is not None and gate_idx is not None and self.loop_gate_module is not None:
                next_x = self.loop_gate_module(gate_prev, next_x, gate_idx)
            if want_qk and qk_states is not None:
                assert q_out is not None and k_out is not None
                qk_states.append((q_out, k_out))
            if hidden_states is not None:
                hidden_states = hidden_states + (next_x,)
            elif hidden_layer_set is not None and exec_idx in hidden_layer_set:
                assert tapped_hidden is not None
                tapped_hidden.append(next_x)
            if attentions is not None:
                attentions = attentions + (attn,)
            return next_x

        if not self._has_loop_extensions:
            for exec_idx, layer_index in enumerate(self.layer_execution_order):
                x = _run_block(x, exec_idx, layer_index, pass_idx=0)
        else:
            sched = self.loop_schedule
            exec_idx = 0
            for layer_index in sched.prelude:
                x = _run_block(x, exec_idx, layer_index, pass_idx=0)
                exec_idx += 1
            prelude_anchor = x
            core_last_pos = len(sched.core) - 1
            for pass_idx in range(sched.num_loops):
                z_prev = x
                if pass_idx >= 1:
                    if self.loop_injector is not None:
                        x = self.loop_injector(z_prev, prelude_anchor)
                    if self.loop_step_cond is not None:
                        x = self.loop_step_cond(x, pass_idx, sched.num_loops)
                for core_pos, layer_index in enumerate(sched.core):
                    apply_gate = pass_idx >= 1 and core_pos == core_last_pos
                    x = _run_block(
                        x,
                        exec_idx,
                        layer_index,
                        pass_idx=pass_idx,
                        gate_prev=z_prev if apply_gate else None,
                        gate_idx=(pass_idx - 1) if apply_gate else None,
                    )
                    exec_idx += 1
            for layer_index in sched.coda:
                x = _run_block(x, exec_idx, layer_index, pass_idx=0)
                exec_idx += 1

        if tapped_hidden is not None:
            hidden_states = tuple(tapped_hidden)
        last_hidden = self.final_norm(x)
        if qk_states is not None:
            return last_hidden, hidden_states, attentions, tuple(qk_states)
        return last_hidden, hidden_states, attentions

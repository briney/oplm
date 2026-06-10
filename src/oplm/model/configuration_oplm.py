"""OplmConfig — PretrainedConfig subclass carrying every model hyperparameter."""

from __future__ import annotations

import math
import warnings
from typing import Any

from transformers import PretrainedConfig

from .conv import resolve_canon_kernel_sizes
from .ffn import round_up_to

# When loading custom code from a local directory with trust_remote_code=True,
# HuggingFace copies only a module's *direct* relative imports (depth-1; see
# transformers.dynamic_module_utils.get_cached_module_file). `conv` imports
# `masking` transitively, so import it directly here to guarantee it is bundled
# alongside this file. Referenced in `_REMOTE_CODE_DEPS` to mark it used.
from .masking import prepare_attention_mask

_REMOTE_CODE_DEPS = (prepare_attention_mask,)

__all__ = ["OplmConfig"]


_VALID_NORM_TYPES = ("layernorm", "rmsnorm")
_VALID_NORM_STRATEGIES = ("pre", "sandwich", "hybrid", "post_sdpa")
_VALID_QK_NORM_MODES = ("channel", "l2")
_VALID_RESIDUAL_SCALINGS = ("sqrt_num_layers", "none")
_VALID_RESIDUAL_GATES = ("none", "scalar", "channel")
_VALID_ATTN_OUTPUT_GATES = ("none", "sigmoid", "silu")
_VALID_FFN_ACTIVATIONS = ("swiglu", "geglu")
_VALID_MLM_HEAD_ACTIVATIONS = ("gelu", "silu", "relu")
_VALID_CANON_ACTIVATIONS = ("none", "silu", "gelu")
_VALID_CLASSIFIER_POOLS = ("mean", "cls")
_VALID_CANON_POSITIONS = frozenset({"A", "B", "C", "D"})
_VALID_GRADIENT_CHECKPOINTING_MODES = ("full", "selective")

_DEFAULT_VOCAB_SIZE = 33


class OplmConfig(PretrainedConfig):
    """Configuration for the OPLM family of encoder-only protein language models.

    Maps 1:1 to the `model:` block of the project YAML schema and to the
    constructor kwargs of every `Oplm*` class. See
    `docs/MODEL_ARCHITECTURE.md` §12 for the field reference and validation rules.
    """

    model_type = "oplm"

    def __init__(
        self,
        *,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
        rope_dim: int | None = None,
        nope_dim: int = 0,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        norm_strategy: str = "pre",
        qk_norm: bool = True,
        qk_norm_mode: str = "channel",
        qk_norm_l2_scale_init: float | None = None,
        post_embed_norm: bool = False,
        mask_dropout: bool = False,
        mask_dropout_reference_ratio: float = 0.12,
        residual_scaling: str = "sqrt_num_layers",
        residual_gate: str = "none",
        residual_gate_init: float = 1.0,
        attn_output_gate: str = "none",
        init_scale_output_projections: bool = True,
        ffn_activation: str = "swiglu",
        ffn_bias: bool = False,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        mlm_head_activation: str = "gelu",
        canon_enabled: bool = False,
        canon_positions: list[str] | None = None,
        canon_kernel_sizes: int | list[int] | dict[str, Any] = 4,
        canon_activation: str = "none",
        initializer_range: float = 0.02,
        classifier_pool: str = "mean",
        classifier_dropout: float = 0.0,
        num_labels: int = 2,
        pre_head_norm: bool = False,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_mode: str = "full",
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        mask_token_id: int = 32,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = head_dim if head_dim is None else int(head_dim)
        self.intermediate_size = (
            intermediate_size if intermediate_size is None else int(intermediate_size)
        )
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_dim = rope_dim if rope_dim is None else int(rope_dim)
        self.nope_dim = int(nope_dim)
        self.norm_type = norm_type
        self.norm_eps = float(norm_eps)
        self.norm_strategy = norm_strategy
        self.qk_norm = bool(qk_norm)
        self.qk_norm_mode = qk_norm_mode
        self.qk_norm_l2_scale_init = (
            qk_norm_l2_scale_init if qk_norm_l2_scale_init is None else float(qk_norm_l2_scale_init)
        )
        self.post_embed_norm = bool(post_embed_norm)
        self.mask_dropout = bool(mask_dropout)
        self.mask_dropout_reference_ratio = float(mask_dropout_reference_ratio)
        self.residual_scaling = residual_scaling
        self.residual_gate = residual_gate
        self.residual_gate_init = float(residual_gate_init)
        self.attn_output_gate = attn_output_gate
        self.init_scale_output_projections = bool(init_scale_output_projections)
        self.ffn_activation = ffn_activation
        self.ffn_bias = bool(ffn_bias)
        self.attention_dropout = float(attention_dropout)
        self.hidden_dropout = float(hidden_dropout)
        self.mlm_head_activation = mlm_head_activation
        self.canon_enabled = bool(canon_enabled)
        self.canon_positions = list(canon_positions) if canon_positions is not None else []
        self.canon_kernel_sizes = canon_kernel_sizes
        self.canon_activation = canon_activation
        self.initializer_range = float(initializer_range)
        self.classifier_pool = classifier_pool
        self.classifier_dropout = float(classifier_dropout)
        # `num_labels` is a property on PretrainedConfig that derives from
        # `id2label` (set in super().__init__). Forward it via kwargs below
        # instead of assigning here.
        self.pre_head_norm = bool(pre_head_norm)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_checkpointing_mode = str(gradient_checkpointing_mode)
        self.unk_token_id = int(unk_token_id)
        self.mask_token_id = int(mask_token_id)

        self._resolve_derived_fields()
        self._validate()

        # `num_labels` is a property on PretrainedConfig backed by
        # `id2label`/`label2id`; forward through kwargs (not as a direct attr)
        # so the base class's bookkeeping stays consistent.
        kwargs.setdefault("num_labels", int(num_labels))

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Derived-field resolution
    # ------------------------------------------------------------------

    def _resolve_derived_fields(self) -> None:
        """Fill in any `None` derived fields from the source dimensions."""
        if self.head_dim is None:
            if self.num_attention_heads == 0:
                raise ValueError("num_attention_heads must be > 0.")
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.intermediate_size is None:
            # SwiGLU convention: ~8/3 * D rounded up to a tensor-core friendly 256.
            self.intermediate_size = round_up_to(int(8 * self.hidden_size / 3), 256)

        if self.rope_dim is None:
            # Default to full RoPE on every head channel.
            self.rope_dim = self.head_dim
            self.nope_dim = 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Raise `ValueError` on any field combination the model cannot handle."""
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0; got {self.num_attention_heads}.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})."
            )
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_attention_heads "
                f"({self.num_attention_heads}) must equal hidden_size ({self.hidden_size})."
            )

        if self.rope_dim < 0:
            raise ValueError(f"rope_dim must be >= 0; got {self.rope_dim}.")
        if self.nope_dim < 0:
            raise ValueError(f"nope_dim must be >= 0; got {self.nope_dim}.")
        if self.rope_dim + self.nope_dim != self.head_dim:
            raise ValueError(
                f"rope_dim ({self.rope_dim}) + nope_dim ({self.nope_dim}) must equal "
                f"head_dim ({self.head_dim})."
            )
        if self.rope_dim % 2 != 0:
            raise ValueError(
                f"rope_dim must be even (each RoPE rotation consumes a channel pair); "
                f"got {self.rope_dim}."
            )

        if self.norm_type not in _VALID_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {_VALID_NORM_TYPES}; got {self.norm_type!r}."
            )
        if self.norm_strategy not in _VALID_NORM_STRATEGIES:
            raise ValueError(
                f"norm_strategy must be one of {_VALID_NORM_STRATEGIES}; "
                f"got {self.norm_strategy!r}."
            )
        if self.qk_norm_mode not in _VALID_QK_NORM_MODES:
            raise ValueError(
                f"qk_norm_mode must be one of {_VALID_QK_NORM_MODES}; got {self.qk_norm_mode!r}."
            )
        if self.qk_norm_l2_scale_init is not None and self.qk_norm_l2_scale_init <= 0:
            raise ValueError(
                f"qk_norm_l2_scale_init must be positive when set; "
                f"got {self.qk_norm_l2_scale_init}."
            )
        if not 0 <= self.mask_dropout_reference_ratio < 1:
            raise ValueError(
                f"mask_dropout_reference_ratio must satisfy 0 <= ratio < 1; "
                f"got {self.mask_dropout_reference_ratio}."
            )
        if self.residual_scaling not in _VALID_RESIDUAL_SCALINGS:
            raise ValueError(
                f"residual_scaling must be one of {_VALID_RESIDUAL_SCALINGS}; "
                f"got {self.residual_scaling!r}."
            )
        if self.residual_gate not in _VALID_RESIDUAL_GATES:
            raise ValueError(
                f"residual_gate must be one of {_VALID_RESIDUAL_GATES}; got {self.residual_gate!r}."
            )
        if not math.isfinite(self.residual_gate_init):
            raise ValueError(f"residual_gate_init must be finite; got {self.residual_gate_init}.")
        if self.attn_output_gate not in _VALID_ATTN_OUTPUT_GATES:
            raise ValueError(
                f"attn_output_gate must be one of {_VALID_ATTN_OUTPUT_GATES}; "
                f"got {self.attn_output_gate!r}."
            )
        if self.ffn_activation not in _VALID_FFN_ACTIVATIONS:
            raise ValueError(
                f"ffn_activation must be one of {_VALID_FFN_ACTIVATIONS}; "
                f"got {self.ffn_activation!r}."
            )
        if self.mlm_head_activation not in _VALID_MLM_HEAD_ACTIVATIONS:
            raise ValueError(
                f"mlm_head_activation must be one of {_VALID_MLM_HEAD_ACTIVATIONS}; "
                f"got {self.mlm_head_activation!r}."
            )
        if self.canon_activation not in _VALID_CANON_ACTIVATIONS:
            raise ValueError(
                f"canon_activation must be one of {_VALID_CANON_ACTIVATIONS}; "
                f"got {self.canon_activation!r}."
            )
        if self.classifier_pool not in _VALID_CLASSIFIER_POOLS:
            raise ValueError(
                f"classifier_pool must be one of {_VALID_CLASSIFIER_POOLS}; "
                f"got {self.classifier_pool!r}."
            )
        if self.gradient_checkpointing_mode not in _VALID_GRADIENT_CHECKPOINTING_MODES:
            raise ValueError(
                f"gradient_checkpointing_mode must be one of "
                f"{_VALID_GRADIENT_CHECKPOINTING_MODES}; "
                f"got {self.gradient_checkpointing_mode!r}."
            )

        if self.canon_enabled:
            if not self.canon_positions:
                raise ValueError("canon_positions must be non-empty when canon_enabled=True.")
            bad_positions = [p for p in self.canon_positions if p not in _VALID_CANON_POSITIONS]
            if bad_positions:
                raise ValueError(
                    f"canon_positions entries must be a subset of "
                    f"{sorted(_VALID_CANON_POSITIONS)}; got invalid entries {bad_positions}."
                )
            if len(set(self.canon_positions)) != len(self.canon_positions):
                raise ValueError(
                    f"canon_positions must not contain duplicates; got {self.canon_positions}."
                )
            # Cache the resolved per-layer list back onto the field so downstream
            # code (OplmBlock) can index it directly without re-resolving.
            self.canon_kernel_sizes = resolve_canon_kernel_sizes(
                self.canon_kernel_sizes, self.num_hidden_layers
            )

        if self.vocab_size != _DEFAULT_VOCAB_SIZE:
            warnings.warn(
                f"vocab_size={self.vocab_size} differs from the OPLM default "
                f"({_DEFAULT_VOCAB_SIZE}); custom vocabularies are not yet supported.",
                UserWarning,
                stacklevel=3,
            )

"""FLOP estimation for transformer encoder models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oplm.model import OplmConfig as OplmModelConfig


def estimate_flops_per_token(config: OplmModelConfig) -> int:
    """Estimate training FLOPs per token (forward + backward ~= 3x forward).

    Counts the dominant linear projections (factor of 2 per matmul). Omits
    attention-score FLOPs, normalization, and embedding lookups by design.
    """
    h = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    intermediate = config.intermediate_size  # always resolved post-__init__
    assert intermediate is not None, "intermediate_size is resolved in OplmConfig.__init__"

    # Attention projections: Q, K, V, O each h -> h  => 2 * h * (4h)
    attn_proj_flops = 2 * h * (4 * h)
    # Gated FFN (swiglu/geglu) has 3 projections of size h * intermediate;
    # non-gated relu2 has 2.
    num_ffn_proj = 2 if config.ffn_activation == "relu2" else 3
    ffn_flops = num_ffn_proj * 2 * h * intermediate

    per_layer = attn_proj_flops + ffn_flops
    backbone_flops = num_hidden_layers * per_layer

    # MLM head: dense projection + vocab projection
    head_flops = 2 * h * h + 2 * h * vocab_size

    forward_flops = backbone_flops + head_flops
    return 3 * forward_flops

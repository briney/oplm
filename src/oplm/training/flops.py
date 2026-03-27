"""FLOP estimation for transformer encoder models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oplm.config import ModelConfig


def estimate_flops_per_token(cfg: ModelConfig) -> int:
    """Estimate training FLOPs per token for the given model config.

    Counts multiply-accumulate operations (factor of 2 per matmul) for the
    dominant linear projections. Follows the Chinchilla/PaLM convention:
    training FLOPs ~= 3x forward FLOPs (forward + backward).

    Deliberately omits attention score computation (sequence-length dependent,
    small fraction of total), normalization, and embedding lookup FLOPs.

    Args:
        cfg: Model configuration.

    Returns:
        Estimated training FLOPs per token (forward + backward).
    """
    h = cfg.hidden_dim
    num_layers = cfg.num_layers
    vocab_size = cfg.vocab_size
    head_dim = cfg.head_dim if cfg.head_dim is not None else h // cfg.num_heads
    q_dim = cfg.num_heads * head_dim
    kv_dim = cfg.num_kv_heads * head_dim

    ffn_dim: int
    if cfg.ffn_dim is not None:
        ffn_dim = cfg.ffn_dim
    elif cfg.ffn_activation == "swiglu":
        import math

        ffn_dim = int(math.ceil(8 / 3 * h / 256) * 256)
    else:
        ffn_dim = 4 * h

    # Per-layer: attention projections (Q, K, V, O)
    attn_proj_flops = 2 * h * (q_dim + 2 * kv_dim + q_dim)

    # Per-layer: FFN (SwiGLU has 3 projections; others have 2)
    ffn_proj_count = 3 if cfg.ffn_activation == "swiglu" else 2
    ffn_flops = ffn_proj_count * 2 * h * ffn_dim

    per_layer = attn_proj_flops + ffn_flops
    encoder_flops = num_layers * per_layer

    # MLM head: dense projection + vocab projection
    mlm_head_flops = 2 * h * h + 2 * h * vocab_size

    forward_flops = encoder_flops + mlm_head_flops

    # Training: forward + backward ~= 3x forward
    return 3 * forward_flops

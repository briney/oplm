"""Structured configuration system for OPLM.

Uses OmegaConf for YAML serialization, CLI overrides, and type-safe merging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


def round_multiple(x: float, multiple: int) -> int:
    """Round x up to the nearest multiple."""
    return int(math.ceil(x / multiple) * multiple)


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Core dimensions
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    head_dim: int | None = None
    ffn_dim: int | None = None
    ffn_activation: str = "swiglu"
    vocab_size: int = 33
    max_seq_len: int = 2048

    # Attention features
    shared_kv: bool = False
    qk_norm: bool = True
    output_gate: bool = False
    query_dependent_gate: bool = False
    post_sdpa_norm: bool = False

    # Positional encoding
    rope_theta: float = 10000.0
    partial_rope: bool = False
    nope_dim: int | None = None
    rope_dim: int | None = None

    # Cross-layer value residuals (Proust)
    value_residual: bool = False
    value_residual_lambda_init: float = 0.5

    # Value embeddings (Proust)
    num_value_embeds: int = 0
    value_embed_gate_dim: int = 16

    # Depthwise convolutions
    conv_positions: str = ""
    conv_kernel_size: int = 7
    conv_activation: bool = True

    # Attention residuals (depth-wise, Kimi)
    attn_residual: bool = False
    attn_residual_block_size: int = 8

    # Normalization
    norm_eps: float = 1e-6
    post_embed_norm: bool = False

    # Training features
    gradient_checkpointing: bool = False
    tie_embeddings: bool = False
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        """Compute derived fields and validate configuration."""
        # Derived fields
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.ffn_dim is None:
            if self.ffn_activation == "swiglu":
                self.ffn_dim = round_multiple(8 / 3 * self.hidden_dim, 256)
            else:
                self.ffn_dim = 4 * self.hidden_dim
        if self.rope_dim is None:
            self.rope_dim = 32 if self.partial_rope else self.head_dim
        if self.nope_dim is None:
            self.nope_dim = self.head_dim - self.rope_dim

        # Validation
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.partial_rope and self.nope_dim + self.rope_dim != self.head_dim:
            raise ValueError(
                f"nope_dim ({self.nope_dim}) + rope_dim ({self.rope_dim}) "
                f"must equal head_dim ({self.head_dim})"
            )
        if self.conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size ({self.conv_kernel_size}) must be odd")
        if self.conv_positions and not all(c in "ACD" for c in self.conv_positions):
            raise ValueError(
                f"conv_positions ({self.conv_positions!r}) must only contain 'A', 'C', 'D'"
            )
        if self.attn_residual and self.num_layers % self.attn_residual_block_size != 0:
            raise ValueError(
                f"attn_residual_block_size ({self.attn_residual_block_size}) "
                f"must divide num_layers ({self.num_layers})"
            )
        valid_activations = ("swiglu", "relu_squared", "gelu")
        if self.ffn_activation not in valid_activations:
            raise ValueError(
                f"ffn_activation must be one of {valid_activations}, got {self.ffn_activation!r}"
            )


@dataclass
class TrainConfig:
    """Training configuration (stub)."""

    lr: float = 1e-4
    batch_size: int = 32
    max_steps: int = 100_000
    seed: int = 42
    output_dir: str = "outputs"
    config_path: str | None = None


@dataclass
class DataConfig:
    """Data configuration (stub)."""

    dataset: str = ""
    max_length: int = 1024
    mask_prob: float = 0.15


@dataclass
class OplmConfig:
    """Root configuration composing model, training, and data configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(argv: list[str]) -> OplmConfig:
    """Load config from defaults, optional YAML file, and CLI overrides.

    Args:
        argv: Command-line arguments (e.g. sys.argv[1:]).
            Supports ``--config <path>`` for YAML files and dotlist overrides
            like ``model.num_layers=32``.

    Returns:
        Fully resolved and validated OplmConfig.
    """
    base: DictConfig = OmegaConf.structured(OplmConfig)

    # Extract --config flag
    config_path: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        else:
            remaining.append(argv[i])
            i += 1

    # Merge YAML file if provided
    if config_path is not None:
        yaml_cfg = OmegaConf.load(config_path)
        base = OmegaConf.merge(base, yaml_cfg)

    # Merge CLI dotlist overrides
    if remaining:
        cli_cfg = OmegaConf.from_dotlist(remaining)
        base = OmegaConf.merge(base, cli_cfg)

    # Convert to dataclass instances (triggers __post_init__ validation)
    cfg: OplmConfig = OmegaConf.to_object(base)  # type: ignore[assignment]
    return cfg

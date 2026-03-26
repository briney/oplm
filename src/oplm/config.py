"""Structured configuration system for OPLM.

Uses OmegaConf for YAML serialization, CLI overrides, and type-safe merging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

from omegaconf import DictConfig, OmegaConf

AVAILABLE_PRESETS = ("small", "medium", "base", "large", "xlarge")


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
class TrainDatasetEntry:
    """Parsed configuration for a single training dataset.

    Populated by :func:`parse_train_configs`, not directly from YAML.
    """

    name: str
    path: str
    fraction: float


@dataclass
class DataConfig:
    """Data configuration for training datasets and loading."""

    # Training dataset(s). Accepts a str path (single dataset) or a dict of
    # {name: {path, fraction}} (multiple datasets). Parsed at runtime via
    # parse_train_configs(). See configs/data/base.yaml for syntax examples.
    train: Any = None

    # Sequence and masking
    max_length: int = 1024
    mask_prob: float = 0.15

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Shard iteration behavior (only affects sharded parquet directories)
    shuffle_shards: bool = True
    shuffle_rows: bool = True


@dataclass
class OplmConfig:
    """Root configuration composing model, training, and data configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def get_preset_config(preset: str) -> DictConfig:
    """Load a model size preset by name.

    Args:
        preset: One of ``"small"``, ``"medium"``, ``"base"``, ``"large"``, ``"xlarge"``.

    Returns:
        DictConfig loaded from the preset YAML.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    if preset not in AVAILABLE_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Available presets: {', '.join(AVAILABLE_PRESETS)}"
        )
    preset_dir = files("oplm.configs.model.presets")
    yaml_text = preset_dir.joinpath(f"{preset}.yaml").read_text()
    return OmegaConf.create(yaml_text)


# Fields in ModelConfig that are derived from other fields in __post_init__.
# These must be reset to None before OmegaConf.to_object() when they were not
# explicitly set by the user, so that __post_init__ recomputes them from the
# (potentially overridden) source dimensions.
_DERIVED_MODEL_FIELDS = ("head_dim", "ffn_dim", "rope_dim", "nope_dim")


def parse_train_configs(raw: Any) -> list[TrainDatasetEntry]:
    """Normalize a raw ``data.train`` config value into structured dataset entries.

    Supports three forms:

    * ``None`` or empty dict → empty list (no training data)
    * String path → single dataset at 100% sampling
    * Dict of ``{name: {path, fraction}}`` → multiple datasets with fractions

    Fractions are normalized to sum to 1.0. Omitted fractions share the
    remaining mass equally among unspecified entries.

    Args:
        raw: The ``data.train`` value from config — a string, dict, or None.

    Returns:
        List of :class:`TrainDatasetEntry` with normalized fractions.

    Raises:
        ValueError: If the config structure is invalid or fractions are negative.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        if not raw:
            return []
        return [TrainDatasetEntry(name="default", path=raw, fraction=1.0)]

    if isinstance(raw, dict):
        if len(raw) == 0:
            return []

        entries: list[TrainDatasetEntry] = []
        for name, value in raw.items():
            if value is None:
                continue
            if isinstance(value, str):
                entries.append(TrainDatasetEntry(name=str(name), path=value, fraction=-1.0))
            elif isinstance(value, dict):
                path = value.get("path")
                if path is None:
                    continue
                frac = value.get("fraction")
                entries.append(
                    TrainDatasetEntry(
                        name=str(name),
                        path=str(path),
                        fraction=float(frac) if frac is not None else -1.0,
                    )
                )
            else:
                raise ValueError(
                    f"Invalid train config for {name!r}: expected str or dict, "
                    f"got {type(value).__name__}"
                )

        if len(entries) == 0:
            return []
        if len(entries) == 1:
            entries[0].fraction = 1.0
            return entries

        # Validate specified fractions
        for e in entries:
            if e.fraction != -1.0 and e.fraction < 0:
                raise ValueError(f"data.train.{e.name}.fraction must be >= 0")

        # Fill unspecified fractions: share remaining mass equally
        specified_total = sum(e.fraction for e in entries if e.fraction >= 0)
        unspecified = [e for e in entries if e.fraction < 0]
        if unspecified:
            remaining = max(0.0, 1.0 - specified_total)
            default_frac = remaining / len(unspecified) if remaining > 0 else 0.0
            for e in unspecified:
                e.fraction = default_frac

        # Normalize to sum to 1.0
        total = sum(e.fraction for e in entries)
        if total <= 0:
            eq = 1.0 / len(entries)
            for e in entries:
                e.fraction = eq
        else:
            for e in entries:
                e.fraction = e.fraction / total

        return entries

    raise ValueError(f"Invalid data.train config type: {type(raw).__name__}")


def load_config(argv: list[str]) -> OplmConfig:
    """Load config from defaults, optional preset, optional YAML file, and CLI overrides.

    Merge order (later overrides earlier): defaults → preset → YAML file → CLI overrides.

    Args:
        argv: Command-line arguments (e.g. sys.argv[1:]).
            Supports ``--preset <name>`` for size presets, ``--config <path>``
            for YAML files, and dotlist overrides like ``model.num_layers=32``.

    Returns:
        Fully resolved and validated OplmConfig.
    """
    base: DictConfig = OmegaConf.structured(OplmConfig)

    # Disable struct mode to allow dynamic keys under data.train
    # (data.train can be a string path or a nested dict of datasets).
    OmegaConf.set_struct(base, False)

    # Extract --config and --preset flags
    config_path: str | None = None
    preset: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif argv[i] == "--preset" and i + 1 < len(argv):
            preset = argv[i + 1]
            i += 2
        else:
            remaining.append(argv[i])
            i += 1

    # Collect explicit overrides to track which model fields the user set
    overrides: list[DictConfig] = []
    if preset is not None:
        overrides.append(get_preset_config(preset))
    if config_path is not None:
        overrides.append(OmegaConf.load(config_path))
    if remaining:
        overrides.append(OmegaConf.from_dotlist(remaining))

    # Merge all overrides into base
    for ov in overrides:
        base = OmegaConf.merge(base, ov)

    # Find which model fields were explicitly provided by the user
    explicit_model_keys: set[str] = set()
    for ov in overrides:
        ov_dict = OmegaConf.to_container(ov, resolve=True)
        if isinstance(ov_dict, dict) and "model" in ov_dict:
            explicit_model_keys.update(ov_dict["model"].keys())

    # Reset derived fields not explicitly set so __post_init__ recomputes
    # them from the (potentially overridden) source dimensions.
    for fname in _DERIVED_MODEL_FIELDS:
        if fname not in explicit_model_keys:
            base.model[fname] = None

    # Convert to dataclass instances (triggers __post_init__ validation)
    cfg: OplmConfig = OmegaConf.to_object(base)  # type: ignore[assignment]
    return cfg

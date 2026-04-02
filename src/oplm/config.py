"""Structured configuration system for OPLM.

Uses OmegaConf for YAML serialization, CLI overrides, and type-safe merging.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

AVAILABLE_PRESETS = ("small", "medium", "base", "large", "xlarge")
_VALID_CONV_KERNEL_SCHEDULES = ("static", "block_step")


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
    max_seq_len: int = 512

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
    conv_kernel_schedule: str = "static"
    conv_kernel_increment: int = 2
    conv_kernel_block_size: int = 1
    conv_kernel_max_size: int | None = None
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
    # Reserved for a future model-construction dtype surface. Runtime precision
    # is currently controlled by ``train.mixed_precision``.
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
        if self.conv_kernel_schedule not in _VALID_CONV_KERNEL_SCHEDULES:
            raise ValueError(
                "conv_kernel_schedule must be one of "
                f"{_VALID_CONV_KERNEL_SCHEDULES}, got {self.conv_kernel_schedule!r}"
            )
        if self.conv_kernel_schedule == "block_step":
            if self.conv_kernel_increment < 0 or self.conv_kernel_increment % 2 != 0:
                raise ValueError(
                    "conv_kernel_increment "
                    f"({self.conv_kernel_increment}) must be a non-negative even integer"
                )
            if self.conv_kernel_block_size < 1:
                raise ValueError(
                    f"conv_kernel_block_size ({self.conv_kernel_block_size}) must be >= 1"
                )
            if self.conv_kernel_max_size is not None:
                if self.conv_kernel_max_size % 2 == 0:
                    raise ValueError(
                        f"conv_kernel_max_size ({self.conv_kernel_max_size}) must be odd"
                    )
                if self.conv_kernel_max_size < self.conv_kernel_size:
                    raise ValueError(
                        "conv_kernel_max_size "
                        f"({self.conv_kernel_max_size}) must be >= conv_kernel_size "
                        f"({self.conv_kernel_size})"
                    )
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

    def conv_kernel_size_for_layer(self, layer_idx: int) -> int:
        """Return the effective convolution kernel size for a given layer."""
        if self.conv_kernel_schedule == "static":
            return self.conv_kernel_size

        kernel_size = (
            self.conv_kernel_size
            + (layer_idx // self.conv_kernel_block_size) * self.conv_kernel_increment
        )
        if self.conv_kernel_max_size is not None:
            kernel_size = min(kernel_size, self.conv_kernel_max_size)
        return kernel_size


_VALID_SCHEDULERS = ("warmup_linear", "warmup_cosine", "wsd_linear", "wsd_cosine")
_VALID_OPTIMIZERS = ("adamw", "muon")
_VALID_MIXED_PRECISION = ("bf16", "fp16", "no")
_VALID_MUON_ADJUST_LR_FNS = ("match_rms_adamw", "original")


@dataclass
class TrainConfig:
    """Training configuration."""

    # Duration
    max_steps: int = 50_000
    max_epochs: int | None = None

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-4
    min_lr: float = 0.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "warmup_linear"
    warmup_steps: int = 5_000
    stable_fraction: float = 0.0

    # Logging
    log_every: int = 10
    eval_every: int = 10_000
    wandb_project: str = "oplm"
    wandb_run_name: str | None = None
    wandb_enabled: bool = True

    # Checkpointing
    save_every: int = 10_000
    save_total_limit: int = 3
    resume_from: str | None = None

    # Infrastructure
    seed: int = 42
    output_dir: str = "outputs"
    # Provenance field populated by ``load_config()`` when a YAML file is used.
    config_path: str | None = None
    mixed_precision: str = "bf16"

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {_VALID_OPTIMIZERS}, got {self.optimizer!r}"
            )
        if self.muon_adjust_lr_fn not in _VALID_MUON_ADJUST_LR_FNS:
            raise ValueError(
                f"muon_adjust_lr_fn must be one of {_VALID_MUON_ADJUST_LR_FNS}, "
                f"got {self.muon_adjust_lr_fn!r}"
            )
        if self.scheduler not in _VALID_SCHEDULERS:
            raise ValueError(
                f"scheduler must be one of {_VALID_SCHEDULERS}, got {self.scheduler!r}"
            )
        if self.mixed_precision not in _VALID_MIXED_PRECISION:
            raise ValueError(
                f"mixed_precision must be one of {_VALID_MIXED_PRECISION}, "
                f"got {self.mixed_precision!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) must be <= lr ({self.lr})")
        if self.muon_momentum < 0:
            raise ValueError(f"muon_momentum must be >= 0, got {self.muon_momentum}")
        if self.muon_ns_steps < 1:
            raise ValueError(f"muon_ns_steps must be >= 1, got {self.muon_ns_steps}")
        if not 0.0 <= self.stable_fraction < 1.0:
            raise ValueError(f"stable_fraction must be in [0, 1), got {self.stable_fraction}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )


@dataclass
class TrainDatasetEntry:
    """Parsed configuration for a single training dataset.

    Populated by :func:`parse_train_configs`, not directly from YAML.
    """

    name: str
    path: str
    fraction: float


@dataclass
class EvalDatasetEntry:
    """Parsed configuration for a single evaluation dataset.

    Populated by :func:`parse_eval_configs`, not directly from YAML.
    """

    name: str
    path: str
    type: str  # "sequence", "structure", "proteingym", ...
    eval_every: int | None = None  # Per-dataset override; None → use train.eval_every
    metrics: list[str] | None = None  # Override default metrics; None → use type defaults
    extra: dict[str, Any] = field(default_factory=dict)  # Task-specific config


@dataclass
class DataConfig:
    """Data configuration for training datasets and loading."""

    # Training dataset(s). Accepts a str path (single dataset) or a dict of
    # {name: {path, fraction}} (multiple datasets). Parsed at runtime via
    # parse_train_configs(). See configs/data/base.yaml for syntax examples.
    train: Any = None

    # Evaluation dataset(s). Accepts a dict of {name: {path, type, ...}}.
    # Parsed at runtime via parse_eval_configs().
    eval: Any = None

    # Sequence and masking
    # Deprecated compatibility alias for model.max_seq_len. load_config()
    # mirrors the resolved model value back into this field so serialized
    # configs stay coherent while the alias remains supported.
    max_length: int = 512
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
    return cast("DictConfig", OmegaConf.create(yaml_text))


# Fields in ModelConfig that are derived from other fields in __post_init__.
# These must be reset to None before OmegaConf.to_object() when they were not
# explicitly set by the user, so that __post_init__ recomputes them from the
# (potentially overridden) source dimensions.
_DERIVED_MODEL_FIELDS = ("head_dim", "ffn_dim", "rope_dim", "nope_dim")
_NESTED_VALUE_MISSING = object()


def _lookup_nested_mapping_value(mapping: Any, path: tuple[str, ...]) -> Any:
    """Return a nested mapping value or a sentinel when the path is absent."""
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _NESTED_VALUE_MISSING
        current = current[key]
    return current


def _normalize_sequence_length_config(
    base: DictConfig,
    override_dicts: list[Any],
) -> None:
    """Canonicalize sequence length onto model.max_seq_len.

    `data.max_length` remains a deprecated compatibility alias. When the alias
    is provided on its own, copy it into `model.max_seq_len` and emit a
    deprecation warning. If both keys are provided and disagree, fail fast.
    The resolved model length is always mirrored back into `data.max_length`
    so downstream config serialization stays coherent.
    """
    explicit_model_max_seq_len = any(
        _lookup_nested_mapping_value(ov, ("model", "max_seq_len")) is not _NESTED_VALUE_MISSING
        for ov in override_dicts
    )
    explicit_data_max_length = any(
        _lookup_nested_mapping_value(ov, ("data", "max_length")) is not _NESTED_VALUE_MISSING
        for ov in override_dicts
    )

    if explicit_data_max_length:
        if explicit_model_max_seq_len and base.model.max_seq_len != base.data.max_length:
            raise ValueError(
                "Conflicting sequence length settings: `model.max_seq_len` and deprecated "
                "`data.max_length` are both set but differ. Use `model.max_seq_len` as the "
                "canonical sequence-length setting."
            )
        warnings.warn(
            "`data.max_length` is deprecated; use `model.max_seq_len` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        base.model.max_seq_len = base.data.max_length

    base.data.max_length = base.model.max_seq_len


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


def parse_eval_configs(raw: Any, default_eval_every: int) -> list[EvalDatasetEntry]:
    """Normalize a raw ``data.eval`` config value into structured eval dataset entries.

    Supports two forms:

    * ``None`` or empty dict → empty list (no eval data)
    * Dict of ``{name: {path, type, eval_every?, metrics?}}`` → multiple eval datasets

    Args:
        raw: The ``data.eval`` value from config — a dict or None.
        default_eval_every: Fallback ``eval_every`` from ``train.eval_every``.

    Returns:
        List of :class:`EvalDatasetEntry` with resolved eval_every values.

    Raises:
        ValueError: If the config structure is invalid or required fields are missing.
    """
    if raw is None:
        return []

    if isinstance(raw, dict):
        if len(raw) == 0:
            return []

        entries: list[EvalDatasetEntry] = []
        for name, value in raw.items():
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(
                    f"Invalid eval config for {name!r}: expected dict with 'path' and 'type', "
                    f"got {type(value).__name__}"
                )

            path = value.get("path")
            if path is None:
                raise ValueError(f"Eval dataset {name!r} is missing required 'path' field")

            eval_type = value.get("type")
            if eval_type is None:
                raise ValueError(f"Eval dataset {name!r} is missing required 'type' field")

            eval_every = value.get("eval_every")
            if eval_every is not None:
                eval_every = int(eval_every)

            raw_metrics = value.get("metrics")
            metrics: list[str] | None = None
            if raw_metrics is not None:
                metrics = [str(m) for m in raw_metrics]

            if "extra" in value:
                raise ValueError(
                    f"Eval dataset {name!r} uses deprecated nested 'extra' config. "
                    "Put task-specific keys directly on the dataset entry instead."
                )

            # Collect task-specific config keys into extra dict
            _KNOWN_EVAL_KEYS = {"path", "type", "eval_every", "metrics"}
            extra = {k: v for k, v in value.items() if k not in _KNOWN_EVAL_KEYS}

            entries.append(
                EvalDatasetEntry(
                    name=str(name),
                    path=str(path),
                    type=str(eval_type),
                    eval_every=eval_every if eval_every is not None else default_eval_every,
                    metrics=metrics,
                    extra=extra,
                )
            )

        return entries

    raise ValueError(f"Invalid data.eval config type: {type(raw).__name__}")


def load_config(argv: list[str]) -> OplmConfig:
    """Load config from defaults, optional preset, optional YAML file, and CLI overrides.

    Merge order (later overrides earlier): defaults → preset → YAML file → CLI overrides.

    Args:
        argv: Command-line arguments (e.g. sys.argv[1:]).
            Supports ``--preset <name>`` for size presets, ``--config <path>``
            for YAML files, and dotlist overrides like ``model.num_layers=32``.

    Returns:
        Fully resolved and validated OplmConfig. If a YAML file was used,
        ``cfg.train.config_path`` is populated with its absolute path.
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
            config_path = str(Path(argv[i + 1]).expanduser().resolve())
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
        overrides.append(cast("DictConfig", OmegaConf.load(config_path)))
    if remaining:
        overrides.append(OmegaConf.from_dotlist(remaining))

    # Merge all overrides into base
    for ov in overrides:
        base = cast("DictConfig", OmegaConf.merge(base, ov))

    override_dicts = [OmegaConf.to_container(ov, resolve=True) for ov in overrides]

    # Find which model fields were explicitly provided by the user
    explicit_model_keys: set[str] = set()
    for ov_dict in override_dicts:
        model_dict = ov_dict.get("model") if isinstance(ov_dict, dict) else None
        if isinstance(model_dict, dict):
            explicit_model_keys.update(model_dict.keys())

    _normalize_sequence_length_config(base, override_dicts)

    # Reset derived fields not explicitly set so __post_init__ recomputes
    # them from the (potentially overridden) source dimensions.
    for fname in _DERIVED_MODEL_FIELDS:
        if fname not in explicit_model_keys:
            base.model[fname] = None

    # Convert to dataclass instances (triggers __post_init__ validation)
    cfg: OplmConfig = OmegaConf.to_object(base)  # type: ignore[assignment]
    cfg.train.config_path = config_path
    return cfg

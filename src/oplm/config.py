"""Structured configuration system for OPLM.

Uses OmegaConf for YAML serialization, CLI overrides, and type-safe merging.
"""

from __future__ import annotations

import math
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
    pre_norm: bool = True
    post_norm: bool = False
    sandwich_norm: bool = False

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
    stable_steps: int = 0

    # Logging
    log_every: int = 10
    # Default eval cadence for datasets that omit `every`. Same grammar as a
    # data.eval.<name>.every block: exactly one of {steps, tokens}. Parsed into a
    # ScheduleSpec by the Evaluator via oplm.config.parse_schedule_block.
    eval_default_every: Any = field(default_factory=lambda: {"steps": 10_000})
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
        if self.stable_steps < 0:
            raise ValueError(f"stable_steps must be >= 0, got {self.stable_steps}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )


@dataclass
class TrainDatasetEntry:
    """Parsed configuration for a single training dataset.

    Populated by :func:`oplm.data.config.parse_train_configs`, not directly from YAML.
    """

    name: str
    path: str
    fraction: float


_VALID_SCHEDULE_UNITS = ("steps", "tokens")  # "epochs" deferred — EVAL_HARNESS.md §4.6
_SCHEDULE_KEYS = frozenset({*_VALID_SCHEDULE_UNITS, "at_start", "at_end"})


@dataclass(frozen=True)
class ScheduleSpec:
    """Parsed, behavior-free eval cadence built from an ``every: {unit: N}`` block.

    Carries no behavior so it can live in ``oplm.config`` without importing
    ``oplm.eval``. The eval harness turns it into a concrete ``Schedule`` via
    ``oplm.eval.schedule.build_schedule``.
    """

    unit: str  # one of _VALID_SCHEDULE_UNITS
    n: int  # positive
    at_start: bool = False
    at_end: bool = True


def parse_schedule_block(raw: Any, label: str) -> ScheduleSpec:
    """Parse an ``every: {unit: N, at_start?, at_end?}`` mapping into a ScheduleSpec.

    Args:
        raw: The cadence mapping (a dataset's ``every`` or ``train.eval_default_every``).
        label: Human-readable source used in error messages.

    Raises:
        ValueError: If ``raw`` is not a mapping, names ``epochs`` (deferred), does not
            name exactly one valid unit, has unknown keys, the unit value is not a
            positive int, or ``at_start`` / ``at_end`` are not actual bools.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"{label}: cadence must be a mapping like {{steps: N}} or {{tokens: N}}, "
            f"got {type(raw).__name__}"
        )
    if "epochs" in raw:
        raise ValueError(
            f"{label}: epoch cadence is not yet supported (see docs/EVAL_HARNESS.md "
            f"§4.6); use {{steps: N}} or {{tokens: N}}"
        )
    unknown = [k for k in raw if k not in _SCHEDULE_KEYS]
    if unknown:
        raise ValueError(f"{label}: unknown keys in cadence block: {sorted(unknown)}")
    unit_keys = [k for k in raw if k in _VALID_SCHEDULE_UNITS]
    if len(unit_keys) != 1:
        raise ValueError(
            f"{label}: cadence must name exactly one of {list(_VALID_SCHEDULE_UNITS)}, "
            f"got {sorted(unit_keys)}"
        )
    unit = unit_keys[0]
    n = raw[unit]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"{label}: cadence {unit!r} must be a positive int, got {n!r}")
    # The schema says bools; parse them strictly. bool(raw.get(...)) would coerce the
    # YAML string "false" to True, silently inverting the flag — validate the type.
    at_start = raw.get("at_start", False)
    at_end = raw.get("at_end", True)
    for flag_name, flag_value in (("at_start", at_start), ("at_end", at_end)):
        if not isinstance(flag_value, bool):
            raise ValueError(
                f"{label}: {flag_name!r} must be a bool, got {flag_value!r} "
                f"({type(flag_value).__name__})"
            )
    return ScheduleSpec(unit=unit, n=n, at_start=at_start, at_end=at_end)


@dataclass
class EvalDatasetEntry:
    """Parsed configuration for a single evaluation dataset.

    Populated by :func:`oplm.data.config.parse_eval_configs`, not directly from YAML.
    """

    name: str
    path: str
    type: str  # registry key: "sequence", "structure", ...
    schedule: ScheduleSpec  # was: eval_every: int | None
    metrics: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data configuration for training datasets and loading."""

    # Training dataset(s). Accepts a str path (single dataset) or a dict of
    # {name: {path, fraction}} (multiple datasets). Parsed at runtime via
    # oplm.data.config.parse_train_configs(). See configs/data/base.yaml for syntax.
    train: Any = None

    # Evaluation dataset(s). Accepts a dict of {name: {path, type, ...}}.
    # Parsed at runtime via oplm.data.config.parse_eval_configs().
    eval: Any = None

    # Sequence masking (see docs/DATA_TOOLING.md §4.5)
    mask_prob: float = 0.15  # fraction of eligible positions selected for masking
    mask_token_prob: float = 0.8  # of masked positions -> <mask>
    random_token_prob: float = 0.1  # of masked positions -> random canonical AA
    weighted_masking: bool = False  # honor the masking_weights column when True (§4.5.1)

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Shard iteration behavior (only affects sharded parquet directories)
    shuffle_shards: bool = True
    shuffle_rows: bool = True

    def __post_init__(self) -> None:
        """Validate the masking-split probabilities."""
        if not 0.0 <= self.mask_token_prob <= 1.0:
            raise ValueError(f"mask_token_prob must be in [0, 1], got {self.mask_token_prob}")
        if not 0.0 <= self.random_token_prob <= 1.0:
            raise ValueError(f"random_token_prob must be in [0, 1], got {self.random_token_prob}")
        # The remainder (1 - mask_token_prob - random_token_prob) keeps the
        # original token, so the two probabilities must not exceed 1 together.
        if self.mask_token_prob + self.random_token_prob > 1.0 + 1e-9:
            raise ValueError(
                "mask_token_prob + random_token_prob must be <= 1, got "
                f"{self.mask_token_prob} + {self.random_token_prob} = "
                f"{self.mask_token_prob + self.random_token_prob}"
            )


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


def _reject_removed_sequence_length_alias(override_dicts: list[Any]) -> None:
    """Reject removed sequence-length overrides before merging them into the config."""
    has_removed_alias = any(
        _lookup_nested_mapping_value(ov, ("data", "max_length")) is not _NESTED_VALUE_MISSING
        for ov in override_dicts
    )
    if has_removed_alias:
        raise ValueError(
            "`data.max_length` has been removed. Use `model.max_seq_len` as the "
            "sequence-length setting."
        )


def _reject_removed_eval_every_alias(override_dicts: list[Any]) -> None:
    """Reject the removed steps-only ``train.eval_every`` override."""
    present = any(
        _lookup_nested_mapping_value(ov, ("train", "eval_every")) is not _NESTED_VALUE_MISSING
        for ov in override_dicts
    )
    if present:
        raise ValueError(
            "`train.eval_every` has been removed. Use "
            "`train.eval_default_every: {steps: N}` (or {tokens: N}) for the global "
            "default eval cadence."
        )


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

    override_dicts = [OmegaConf.to_container(ov, resolve=True) for ov in overrides]
    _reject_removed_sequence_length_alias(override_dicts)
    _reject_removed_eval_every_alias(override_dicts)

    # Merge all overrides into base
    for ov in overrides:
        base = cast("DictConfig", OmegaConf.merge(base, ov))

    # Find which model fields were explicitly provided by the user
    explicit_model_keys: set[str] = set()
    for ov_dict in override_dicts:
        model_dict = ov_dict.get("model") if isinstance(ov_dict, dict) else None
        if isinstance(model_dict, dict):
            explicit_model_keys.update(model_dict.keys())

    # Reset derived fields not explicitly set so __post_init__ recomputes
    # them from the (potentially overridden) source dimensions.
    for fname in _DERIVED_MODEL_FIELDS:
        if fname not in explicit_model_keys:
            base.model[fname] = None

    # Convert to dataclass instances (triggers __post_init__ validation)
    cfg: OplmConfig = OmegaConf.to_object(base)  # type: ignore[assignment]
    cfg.train.config_path = config_path
    return cfg

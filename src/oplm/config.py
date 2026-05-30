"""Structured configuration system for OPLM.

Uses OmegaConf for YAML serialization, CLI overrides, and type-safe merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

AVAILABLE_PRESETS = ("small", "medium", "base", "large", "xlarge")


_VALID_SCHEDULERS = ("warmup_linear", "warmup_cosine", "wsd_linear", "wsd_cosine")
_VALID_OPTIMIZERS = ("adamw", "muon")
_VALID_MIXED_PRECISION = ("bf16", "fp16", "no")
_VALID_MUON_ADJUST_LR_FNS = ("match_rms_adamw", "original")

# Effective default for ``train.eval_every`` when left null. The field itself
# defaults to None (not this mapping) so an override replaces it cleanly:
# OmegaConf deep-merges dicts, so a non-null ``{steps: N}`` default would linger
# and collide with a ``{tokens: M}`` override (two units → parse error). The
# Evaluator coalesces None → this constant. See parse_schedule_block.
DEFAULT_EVAL_CADENCE = {"steps": 10_000}


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
    # data.eval.<name>.every block: exactly one of {steps, tokens}. Defaults to
    # None so a CLI/config override replaces it cleanly (OmegaConf deep-merges
    # dicts); the Evaluator coalesces None → DEFAULT_EVAL_CADENCE and parses it
    # into a ScheduleSpec via oplm.config.parse_schedule_block.
    eval_every: Any = None
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
        raw: The cadence mapping (a dataset's ``every`` or ``train.eval_every``).
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

    # Resolved into oplm.model.OplmConfig (the HF PretrainedConfig) by
    # load_config. Untyped so OmegaConf can carry arbitrary HF field keys,
    # mirroring how DataConfig.train/.eval are Any.
    model: Any = field(default_factory=dict)
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


# Per-concern default layers, merged at the config root in this order. Each
# file is section-wrapped (top-level `model:` / `train:` / `data:`).
_BASE_CONFIG_LAYERS = (
    ("oplm.configs.model", "base.yaml"),
    ("oplm.configs.train", "base.yaml"),
    ("oplm.configs.data", "base.yaml"),
)


def _load_packaged_yaml(package: str, filename: str) -> DictConfig:
    """Load a YAML resource shipped inside the package as a DictConfig."""
    text = files(package).joinpath(filename).read_text()
    return cast("DictConfig", OmegaConf.create(text))


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
            "`data.max_length` has been removed. Use `model.max_position_embeddings` "
            "as the sequence-length setting."
        )


def load_config(argv: list[str]) -> OplmConfig:
    """Load config from defaults, optional preset, optional YAML file, and CLI overrides.

    Merge order (later overrides earlier): defaults → preset → YAML file → CLI overrides.

    Args:
        argv: Command-line arguments (e.g. sys.argv[1:]).
            Supports ``--preset <name>`` for size presets, ``--config <path>``
            for YAML files, ``--name <run>`` for the W&B run name, and dotlist
            overrides like ``model.num_hidden_layers=32``.

    Returns:
        Fully resolved and validated OplmConfig. If a YAML file was used,
        ``cfg.train.config_path`` is populated with its absolute path. ``--name``
        sets ``cfg.train.wandb_run_name`` unless that field was explicitly set in
        the YAML or via a CLI override (an explicit value always wins); when both
        are unset it stays ``None`` (W&B assigns a random name).
    """
    base: DictConfig = OmegaConf.structured(OplmConfig)

    # Disable struct mode to allow dynamic keys under data.train
    # (data.train can be a string path or a nested dict of datasets).
    OmegaConf.set_struct(base, False)

    # Authoritative YAML defaults layer (model → train → data). These are the
    # human-editable home for defaults and override the dataclass fallbacks.
    for package, filename in _BASE_CONFIG_LAYERS:
        base = cast("DictConfig", OmegaConf.merge(base, _load_packaged_yaml(package, filename)))

    # Extract --config, --preset, and --name flags
    config_path: str | None = None
    preset: str | None = None
    run_name: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = str(Path(argv[i + 1]).expanduser().resolve())
            i += 2
        elif argv[i] == "--preset" and i + 1 < len(argv):
            preset = argv[i + 1]
            i += 2
        elif argv[i] == "--name" and i + 1 < len(argv):
            run_name = argv[i + 1]
            i += 2
        else:
            remaining.append(argv[i])
            i += 1

    # Collect user overrides (preset → --config → CLI dotlist), applied on top
    # of the authoritative base layer.
    overrides: list[DictConfig] = []
    if preset is not None:
        overrides.append(get_preset_config(preset))
    if config_path is not None:
        overrides.append(cast("DictConfig", OmegaConf.load(config_path)))
    if remaining:
        overrides.append(OmegaConf.from_dotlist(remaining))

    override_dicts = [OmegaConf.to_container(ov, resolve=True) for ov in overrides]
    _reject_removed_sequence_length_alias(override_dicts)

    # Whether the user explicitly set the W&B run name (YAML or CLI override).
    # An explicit value — even ``null`` — wins over the ``--name`` flag below.
    user_set_run_name = any(
        _lookup_nested_mapping_value(ov, ("train", "wandb_run_name")) is not _NESTED_VALUE_MISSING
        for ov in override_dicts
    )

    # Merge all overrides into base
    for ov in overrides:
        base = cast("DictConfig", OmegaConf.merge(base, ov))

    # Build train/data dataclasses (triggers their __post_init__ validation);
    # `model` is an Any field so it round-trips as a plain dict here.
    cfg: OplmConfig = OmegaConf.to_object(base)  # type: ignore[assignment]

    # Instantiate the HF model config from the merged `model` subtree. HF owns
    # derivation (head_dim, intermediate_size, rope_dim/nope_dim) and validation.
    # Derived fields are omitted unless set, resolving to None → derived inside
    # OplmModelConfig.__init__. Unknown / old / mistyped `model.*` keys flow into
    # **model_dict → PretrainedConfig **kwargs and are silently retained.
    from oplm.model import OplmConfig as OplmModelConfig

    model_dict = OmegaConf.to_container(base.model, resolve=True) or {}
    cfg.model = OplmModelConfig(**model_dict)  # type: ignore[arg-type]

    cfg.train.config_path = config_path

    # Propagate --name to the W&B run name unless explicitly set in YAML/CLI.
    if run_name is not None and not user_set_run_name:
        cfg.train.wandb_run_name = run_name

    return cfg

"""Structured configuration system for OPLM.

Uses OmegaConf for YAML serialization, CLI overrides, and type-safe merging.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

AVAILABLE_PRESETS = ("50M", "170M", "400M", "800M", "1B", "3B", "6B", "12B")


_VALID_SCHEDULERS = ("warmup_linear", "warmup_cosine", "wsd_linear", "wsd_cosine")
_VALID_OPTIMIZERS = ("adamw", "muon")
_VALID_MIXED_PRECISION = ("bf16", "fp16", "no")
_VALID_PARALLELISM = ("ddp", "hsdp")
_VALID_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
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

    # Optimizer.
    # NOTE: these are conservative *library* fallbacks for bare TrainConfig() /
    # direct construction. OPLM's production training default (what `oplm train`
    # loads) is μP + Muon with lr 0.01 — see configs/train/base.yaml and docs/MUP.md.
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
    mup_depth_lr_exponent: float = 0.0
    mup_depth_reference_layers: int = 24
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
    # Always write a checkpoint when training completes, even if the final step
    # is not a multiple of save_every. The save is skipped only when the final
    # step already triggered a periodic save (avoids a redundant re-write).
    save_final: bool = True
    resume_from: str | None = None
    # Time-based checkpoint cadence: also save every N wall-clock minutes, in
    # addition to (not instead of) the step-based save_every cadence. None
    # disables it (checkpoints only happen on the save_every step cadence).
    save_every_minutes: int | None = None
    # Retain (never rotate away) every checkpoint whose step is a multiple of
    # this value, in addition to the rolling save_total_limit window. None
    # disables this exemption (all checkpoints are subject to rotation).
    keep_every_n_steps: int | None = None
    # Retain (never rotate away) a checkpoint at least this many wall-clock
    # hours after the previous permanent checkpoint. None disables this
    # exemption. Enforced by the trainer via checkpoint.mark_permanent.
    keep_every_n_hours: float | None = None
    # Automatically resume from the latest committed checkpoint under
    # output_dir at trainer start, without an explicit --resume_from.
    auto_resume: bool = False
    # When resuming, also restore the data-loader position (sampler/shard
    # offsets) so training resumes from the same data point rather than
    # restarting the data stream from the beginning of the current epoch.
    resume_data_position: bool = True
    # Timeout (minutes) for the process-group / distributed backend, passed to
    # Accelerator's InitProcessGroupKwargs. Longer timeouts tolerate slow
    # checkpoint writes or requeue delays without triggering a collective abort.
    dist_timeout_minutes: int = 15
    # Optional remote URI (e.g. s3://bucket/prefix) that committed checkpoints
    # are additionally synced to, for durability beyond local/shared storage.
    # None disables remote sync.
    remote_checkpoint_uri: str | None = None

    # Parallelism strategy.
    #   "ddp"  — one full model replica per rank, gradients all-reduced (default;
    #            what every run did before Phase 5).
    #   "hsdp" — FSDP2 (`fully_shard`) over a 2-D device mesh: shard within a node,
    #            replicate across nodes. Required before ~24B; needs world size > 1.
    # Checkpoints are parallelism-agnostic (DCP `get_state_dict`), so the same
    # checkpoint resumes under either setting -- see docs/TRAIN.md.
    parallelism: str = "ddp"

    # Infrastructure
    seed: int = 42
    output_dir: str = "outputs"
    # Provenance field populated by ``load_config()`` when a YAML file is used.
    config_path: str | None = None
    mixed_precision: str = "bf16"
    # torch.compile
    # Compiles the model with torch.compile before DDP wrapping. Requires an
    # initial compilation step on the first forward pass (may take several minutes
    # for large models). Disabled by default.
    # compile_dynamic controls the ``dynamic`` argument to torch.compile:
    #   True  — single dynamic graph (variable shapes without recompilation; default).
    #   False — static graph per shape (best with pad_to_multiple_of to limit shapes).
    #   None  — Dynamo auto-selects (single-element tensors → static, others → dynamic).
    compile: bool = False
    # Compilation mode passed to torch.compile(mode=...).
    # "default"          — balanced; safe for all hardware.
    # "reduce-overhead"  — uses CUDA graphs to reduce kernel-launch overhead;
    #                      best for small batch sizes.
    # "max-autotune"     — tries more optimization strategies; longest compile
    #                      time, best peak throughput on Blackwell.
    compile_mode: str = "default"
    # dynamic flag for torch.compile: True (one dynamic graph), False (static per
    # shape — pair with pad_to_multiple_of to bound the shape space), None (auto).
    compile_dynamic: bool | None = True

    # Throughput / MFU logging
    # Steps to exclude from steady-state throughput (compile/warmup transient).
    throughput_warmup_steps: int = 50
    # Device peak TFLOPs for MFU calculation. None → log achieved TFLOPs only.
    peak_tflops: float | None = None

    # Training-stability diagnostics (deep-model μP debugging; see docs/LR_SWEEP.md).
    # When True, the Trainer attaches a StabilityDiagnosticsCallback. It logs the
    # pre-clip global grad norm every training log (free, no hooks), and every
    # `stability_probe_every` logs runs one eager diagnostic forward on the
    # unwrapped model recording per-depth residual-stream RMS, output-logit RMS,
    # and attention entropy under `diag/*`. Off by default; enable it for the deep
    # stability probe and control runs.
    stability_diagnostics: bool = False
    # Cadence (in training-log emissions) of the diagnostic probe forward. The
    # probe runs eagerly on the *unwrapped* model, off the compiled training step,
    # so `torch.compile` stays on (there are no forward hooks). 0 logs only the
    # grad norm. Consulted only when stability_diagnostics is True.
    stability_probe_every: int = 25

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
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {_VALID_COMPILE_MODES}, got {self.compile_mode!r}"
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
        if self.throughput_warmup_steps < 0:
            raise ValueError(
                f"throughput_warmup_steps must be >= 0, got {self.throughput_warmup_steps}"
            )
        if self.peak_tflops is not None and self.peak_tflops <= 0:
            raise ValueError(f"peak_tflops must be > 0, got {self.peak_tflops}")
        if not math.isfinite(self.mup_depth_lr_exponent) or self.mup_depth_lr_exponent < 0:
            raise ValueError(
                f"mup_depth_lr_exponent must be finite and >= 0, got {self.mup_depth_lr_exponent}"
            )
        if self.mup_depth_reference_layers < 1:
            raise ValueError(
                f"mup_depth_reference_layers must be >= 1, got {self.mup_depth_reference_layers}"
            )
        if self.stability_probe_every < 0:
            raise ValueError(
                f"stability_probe_every must be >= 0, got {self.stability_probe_every}"
            )
        if self.save_every_minutes is not None and self.save_every_minutes <= 0:
            raise ValueError(
                f"save_every_minutes must be > 0 when set, got {self.save_every_minutes}"
            )
        if self.keep_every_n_steps is not None and self.keep_every_n_steps <= 0:
            raise ValueError(
                f"keep_every_n_steps must be > 0 when set, got {self.keep_every_n_steps}"
            )
        if self.keep_every_n_hours is not None and self.keep_every_n_hours <= 0:
            raise ValueError(
                f"keep_every_n_hours must be > 0 when set, got {self.keep_every_n_hours}"
            )
        if self.dist_timeout_minutes <= 0:
            raise ValueError(f"dist_timeout_minutes must be > 0, got {self.dist_timeout_minutes}")
        if self.parallelism not in _VALID_PARALLELISM:
            raise ValueError(
                f"parallelism must be one of {_VALID_PARALLELISM}, got {self.parallelism!r}"
            )
        if self.parallelism == "hsdp" and self.mixed_precision == "fp16":
            # Under FSDP2 every gradient is a sharded DTensor, and torch.amp.GradScaler's
            # unscale_/inf-check runs per rank over local shards without a cross-rank
            # reduction -- ranks can then disagree about whether to skip a step, which
            # desynchronizes the run (a hang or silently divergent replicas, not a clean
            # failure). bf16, the default, needs no scaler; fully_shard's own
            # MixedPrecisionPolicy handles the param/reduce dtypes.
            raise ValueError(
                "parallelism='hsdp' does not support mixed_precision='fp16': the fp16 "
                "GradScaler's inf-check is not shard-aware and would let ranks diverge. "
                "Use mixed_precision='bf16' (default) or 'no'."
            )
        if self.parallelism == "hsdp" and self.stability_diagnostics and self.stability_probe_every:
            # StabilityDiagnosticsCallback's probe forward is deliberately main-process
            # only, on the documented assumption that "the extra forward has no
            # collectives" -- true under DDP, false under FSDP2, where every forward
            # all-gathers sharded parameters. Running it on rank 0 alone would hang every
            # other rank until dist_timeout_minutes, then crash-loop through the requeue.
            # Refuse up front instead (grad-norm-only diagnostics stay available via
            # stability_probe_every=0).
            raise ValueError(
                "parallelism='hsdp' is incompatible with stability_diagnostics=true and "
                "stability_probe_every > 0: the diagnostic probe runs a main-process-only "
                "forward, which under FSDP2 issues an all-gather that would hang every "
                "other rank. Set train.stability_probe_every=0 to keep the (collective-free) "
                "grad-norm diagnostic, or run the probe under train.parallelism='ddp'."
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

    # Collation padding: None = pad to batch-longest sequence; N = pad to the
    # smallest multiple of N that is >= batch-longest. Useful for compile+static
    # regimes (dynamic=False) to reduce unique shapes and improve throughput.
    pad_to_multiple_of: int | None = None

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Shard iteration behavior (only affects sharded parquet directories)
    shuffle_shards: bool = True
    shuffle_rows: bool = True

    def __post_init__(self) -> None:
        """Validate the masking-split probabilities and padding config."""
        if self.pad_to_multiple_of is not None:
            # Reject bool explicitly: isinstance(True, int) is True in Python, so
            # a bare int check would silently accept True/False.
            if isinstance(self.pad_to_multiple_of, bool):
                raise ValueError(
                    f"pad_to_multiple_of must be an int, got bool ({self.pad_to_multiple_of!r})"
                )
            if not isinstance(self.pad_to_multiple_of, int):
                got = type(self.pad_to_multiple_of).__name__
                raise ValueError(f"pad_to_multiple_of must be an int, got {got}")
            if self.pad_to_multiple_of < 1:
                raise ValueError(f"pad_to_multiple_of must be >= 1, got {self.pad_to_multiple_of}")
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
        preset: One of ``"50M"``, ``"170M"``, ``"400M"``, ``"800M"``, ``"1B"``, ``"3B"``,
            ``"6B"``, ``"12B"``.

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


def _reject_unknown_model_keys(model_dict: dict[str, Any], model_config_cls: type) -> None:
    """Raise on `model.*` keys that no `OplmConfig` field or HF metadata accepts.

    `PretrainedConfig.__init__` silently absorbs unknown `**kwargs`, so a typo
    like `model.cannon_enabled=true` would otherwise be retained while the real
    `canon_enabled` stays at its default — silently disabling an ablation. The
    allowed set is the constructor signature plus the HF metadata keys a default
    instance emits via `to_dict()` (so serialized configs still round-trip).

    Args:
        model_dict: The merged `model` subtree about to construct the HF config.
        model_config_cls: The `OplmConfig` (HF `PretrainedConfig`) class.

    Raises:
        ValueError: If `model_dict` contains any key outside the allowed set.
    """
    params = inspect.signature(model_config_cls.__init__).parameters
    allowed = {
        name
        for name, p in params.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    # HF metadata keys (model_type, transformers_version, architectures, dtype,
    # id2label, ...) are not constructor params but round-trip through a
    # serialized config; a default instance enumerates exactly that set.
    allowed |= set(model_config_cls().to_dict())

    unknown = sorted(set(model_dict) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown model config key(s): {unknown}. Check for typos "
            "(e.g. 'canon_enabled', not 'cannon_enabled'); valid keys come from "
            "OplmConfig.__init__ and HuggingFace PretrainedConfig metadata."
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
        are unset it stays ``None`` (W&B assigns a random name). ``--name`` also
        seeds ``cfg.train.output_dir`` to ``./<name>`` unless ``train.output_dir``
        was explicitly set in the YAML or via a CLI override.
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

    # Whether the user explicitly set the output directory (YAML or CLI override).
    # An explicit value wins over the ``--name``-derived default below.
    user_set_output_dir = any(
        _lookup_nested_mapping_value(ov, ("train", "output_dir")) is not _NESTED_VALUE_MISSING
        for ov in override_dicts
    )

    # Merge all overrides into base
    for ov in overrides:
        base = cast("DictConfig", OmegaConf.merge(base, ov))

    # Build train/data dataclasses (triggers their __post_init__ validation);
    # `model` is an Any field so it round-trips as a plain dict here.
    cfg: OplmConfig = OmegaConf.to_object(base)  # ty: ignore[invalid-assignment]  # OmegaConf union

    # Instantiate the HF model config from the merged `model` subtree. HF owns
    # derivation (head_dim, intermediate_size, rope_dim/nope_dim) and validation.
    # Derived fields are omitted unless set, resolving to None → derived inside
    # OplmModelConfig.__init__. Unknown / mistyped `model.*` keys are rejected
    # here (run-config path) rather than silently absorbed into
    # PretrainedConfig **kwargs; from_pretrained() stays permissive for
    # checkpoint compatibility.
    from oplm.model import OplmConfig as OplmModelConfig

    model_dict = cast("dict[str, Any]", OmegaConf.to_container(base.model, resolve=True) or {})
    _reject_unknown_model_keys(model_dict, OplmModelConfig)
    cfg.model = OplmModelConfig(**model_dict)

    # Cross-config divisibility check: pad_to_multiple_of must evenly divide
    # max_position_embeddings so padded lengths can never overflow position IDs.
    # This is the only place both cfg.data and the resolved cfg.model are available.
    if cfg.data.pad_to_multiple_of is not None:
        max_pos = cfg.model.max_position_embeddings
        ptm = cfg.data.pad_to_multiple_of
        if max_pos % ptm != 0:
            raise ValueError(
                f"pad_to_multiple_of ({ptm}) must evenly divide "
                f"model.max_position_embeddings ({max_pos}); "
                f"{max_pos} % {ptm} = {max_pos % ptm}"
            )

    cfg.train.config_path = config_path

    # Propagate --name to the W&B run name unless explicitly set in YAML/CLI.
    if run_name is not None and not user_set_run_name:
        cfg.train.wandb_run_name = run_name

    # Propagate --name to the output directory (./<name>) unless explicitly set.
    if run_name is not None and not user_set_output_dir:
        cfg.train.output_dir = run_name

    return cfg


def serialize_config(cfg: OplmConfig) -> str:
    """Serialize a resolved :class:`OplmConfig` to reloadable YAML text.

    The ``model`` subtree is an HF ``OplmConfig`` (serialized via ``to_dict``);
    ``train`` and ``data`` are dataclasses. The result round-trips through
    ``load_config(["--config", <path>])``. Shared by the top-level run config dump
    and the per-checkpoint ``config.yaml``.
    """
    from dataclasses import asdict

    model_dict = cfg.model.to_dict()
    # `auto_map` is HF `register_for_auto_class`/`save_pretrained` plumbing for
    # trust_remote_code loading (see oplm/__init__.py's registration calls), not
    # something a user ever sets. `PretrainedConfig.register_for_auto_class` stamps
    # it onto the config instance *in place* the first time any `save_pretrained`
    # call runs against it -- e.g. this very checkpoint's own `hf/` export just
    # below in `save_checkpoint`, or an earlier checkpoint's, since `cfg.model` is
    # one shared, mutable object across every checkpoint in a run. Once stamped, it
    # sticks for the rest of the process, so every checkpoint after the first one
    # saved would otherwise carry it into config.yaml -- and fail to round-trip
    # through `load_config` (`_reject_unknown_model_keys` rejects it: a freshly
    # constructed default instance's own `to_dict()` never has it either, since it
    # is only ever added by `save_pretrained`, not by construction).
    model_dict.pop("auto_map", None)

    config_dict = {
        "model": model_dict,
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
    }
    return OmegaConf.to_yaml(OmegaConf.create(config_dict))

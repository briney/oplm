"""Schema and resolution for the ``slurm:`` block of an oplm training config.

The block is general: it describes how to turn a training config into Slurm job scripts and
carries no μP or sweep concepts. ``load_config`` tolerates it as an unknown top-level key
(``OmegaConf.set_struct(base, False)``), and ``serialize_config`` omits it, so a generated
per-cell ``run.yaml`` never carries cluster settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

# PEP 695 (`class PhaseTable[T]:`) is Python 3.12+; this project supports 3.11 and CI runs
# 3.11/3.12/3.13, so use the TypeVar + Generic form.
T = TypeVar("T")

_DEFAULT_KEY = "default"
# Stands in for "any preset" when a phase entry carries a bare value rather than a preset map,
# e.g. `time_limit: {default: "168:00:00", analyze: "01:00:00"}`.
_ANY_KEY = "*"


@dataclass(frozen=True)
class PhaseTable(Generic[T]):
    """A setting that may be a scalar, per-preset, or per-phase-and-preset.

    Four accepted YAML forms::

        nodes: 4                                          # one value for every job
        nodes: {170M: 1, 400M: 4}                         # per preset
        nodes: {default: {170M: 1}, bridge: {170M: 4}}    # per phase, then per preset
        time_limit: {default: "168:00:00", analyze: "1:00:00"}   # per phase, no preset dimension

    The presence of a ``default`` key is what distinguishes the last two forms from the second.
    A phase entry that is not itself a mapping applies to every preset in that phase.
    """

    name: str
    scalar: T | None
    tables: dict[str, dict[str, T]]

    @classmethod
    def from_value(cls, value: object, *, name: str) -> PhaseTable[T]:
        """Build a table from any of the four accepted forms."""
        if not isinstance(value, Mapping):
            return cls(name=name, scalar=value, tables={})
        if _DEFAULT_KEY in value:
            tables: dict[str, dict[str, T]] = {}
            for phase, sub in value.items():
                if isinstance(sub, Mapping):
                    inner = {str(preset): entry for preset, entry in sub.items()}
                    tables[str(phase)] = inner  # ty: ignore[invalid-assignment]  # untyped YAML
                else:
                    # untyped YAML values can't be narrowed to T
                    tables[str(phase)] = {_ANY_KEY: sub}  # ty: ignore[invalid-assignment]
            return cls(name=name, scalar=None, tables=tables)
        return cls(
            name=name,
            scalar=None,
            tables={_DEFAULT_KEY: {str(preset): entry for preset, entry in value.items()}},
        )

    def resolve(self, *, phase: str | None, preset: str | None) -> T:
        """Resolve the value for one (phase, preset) pair.

        Looks in the phase's own table first, then the ``default`` table; within each, an exact
        preset match wins over the ``*`` wildcard.

        Raises:
            KeyError: If no entry covers ``preset``.
        """
        if self.scalar is not None:
            return self.scalar
        for table in (self.tables.get(phase or ""), self.tables.get(_DEFAULT_KEY)):
            if table is None:
                continue
            if preset is not None and preset in table:
                return table[preset]
            if _ANY_KEY in table:
                return table[_ANY_KEY]
        raise KeyError(f"{self.name} has no entry for phase={phase!r} preset={preset!r}")


def _require(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"slurm config is missing required field {key!r}")
    return raw[key]


def _reject_bool(value: Any, description: str) -> None:
    """Raise if ``value`` is a ``bool``.

    ``bool`` is an ``int`` subclass, so a bare type/range check silently accepts it (and YAML
    coerces bare ``true``/``false``/``yes``/``no``/``on``/``off`` to booleans), which would
    otherwise coerce to 0 or 1 without ever tripping a "not an int" guard.
    """
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a positive int, got {value!r}")


def _positive_int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"slurm config is missing required field {key!r}")
    _reject_bool(value, f"slurm config {key}")
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"slurm config {key} must be a positive int, got {value!r}")
    return value


def _coerce_positive_int(value: Any, description: str) -> int:
    """Coerce ``value`` to a positive int for fields that accept int-like YAML scalars.

    Unlike ``_positive_int`` (which requires a literal ``int``), this also accepts values
    coercible via ``int()`` (e.g. numeric strings), but still rejects ``bool`` outright and
    turns a non-numeric value into an actionable message instead of a raw ``ValueError`` from
    ``int()``.
    """
    _reject_bool(value, description)
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a positive int, got {value!r}") from exc
    if coerced < 1:
        raise ValueError(f"{description} must be a positive int, got {value!r}")
    return coerced


@dataclass(frozen=True)
class SlurmConfig:
    """Validated ``slurm:`` block."""

    partition: str
    time_limit: PhaseTable[str]
    nodes: PhaseTable[int]
    max_batch_size: dict[str, int]
    log_dir: Path
    env_file: Path
    container_image: Path
    container_mounts: tuple[str, ...]
    install: str
    gpus_per_node: int = 8
    cpus_per_task: int = 128
    mem: str = "0"
    exclusive: bool = True
    max_concurrent: int = 4
    account: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SlurmConfig:
        """Validate and build from the raw YAML mapping.

        Raises:
            ValueError: On a missing required field or an out-of-range value.
        """
        max_batch: dict[str, int] = {}
        bad: list[str] = []
        for preset, value in dict(raw.get("max_batch_size", {})).items():
            try:
                max_batch[str(preset)] = _coerce_positive_int(
                    value, f"slurm config max_batch_size[{preset!r}]"
                )
            except ValueError as exc:
                bad.append(str(exc))
        if bad:
            raise ValueError("; ".join(bad))
        return cls(
            partition=str(_require(raw, "partition")),
            time_limit=PhaseTable.from_value(_require(raw, "time_limit"), name="time_limit"),
            nodes=PhaseTable.from_value(_require(raw, "nodes"), name="nodes"),
            max_batch_size=max_batch,
            log_dir=Path(str(_require(raw, "log_dir"))),
            env_file=Path(str(_require(raw, "env_file"))),
            container_image=Path(str(_require(raw, "container_image"))),
            container_mounts=tuple(str(mount) for mount in _require(raw, "container_mounts")),
            install=str(_require(raw, "install")),
            gpus_per_node=_positive_int(raw, "gpus_per_node", 8),
            cpus_per_task=_positive_int(raw, "cpus_per_task", 128),
            mem=str(raw.get("mem", "0")),
            exclusive=bool(raw.get("exclusive", True)),
            max_concurrent=_positive_int(raw, "max_concurrent", 4),
            account=str(raw["account"]) if raw.get("account") is not None else None,
        )


@dataclass(frozen=True)
class BatchPlan:
    """Per-device batch and accumulation for one training job."""

    per_device_batch: int
    gradient_accumulation_steps: int
    world_size: int


def resolve_batch_plan(*, global_examples: int, world_size: int, max_batch_size: int) -> BatchPlan:
    """Derive per-device batch and accumulation from the global batch and world size.

    Picks the smallest accumulation that yields an integer per-device batch no larger than
    ``max_batch_size``. Node counts are chosen for wall time, so per-device batch is derived
    rather than configured; an infeasible combination is an error, never a silently adjusted
    global batch.

    Takes ``world_size`` rather than a node count so both callers land on an exact global batch:
    Slurm passes ``nodes * gpus_per_node``, while ``--local`` passes its actual process count.

    ``accum == base`` (i.e. ``per_device_batch == 1``) always divides evenly and always satisfies
    the cap once ``max_batch_size >= 1``, so the search never exhausts its range: at worst it
    degenerates to per-device batch 1 with accumulation equal to the full per-replica batch.

    Args:
        global_examples: Target global batch in examples per optimizer step.
        world_size: Total training processes.
        max_batch_size: Memory cap on per-device batch for this preset.

    Returns:
        The resolved plan.

    Raises:
        ValueError: If any argument is non-positive or a ``bool``, or if the global batch is not
            divisible by the world size.
    """
    _reject_bool(global_examples, "resolve_batch_plan global_examples")
    _reject_bool(world_size, "resolve_batch_plan world_size")
    _reject_bool(max_batch_size, "resolve_batch_plan max_batch_size")
    if global_examples < 1 or world_size < 1 or max_batch_size < 1:
        raise ValueError(
            "global_examples, world_size, and max_batch_size must all be >= 1; got "
            f"{global_examples=}, {world_size=}, {max_batch_size=}"
        )
    if global_examples % world_size != 0:
        raise ValueError(
            f"global batch {global_examples} is not divisible by world size {world_size}"
        )
    base = global_examples // world_size
    # accum == base always divides base evenly and yields per_device == 1, which satisfies
    # max_batch_size >= 1 (checked above), so this search always has a match: at worst it
    # degenerates to per-device batch 1 with accumulation equal to the full per-replica batch.
    accum = next(
        accum
        for accum in range(1, base + 1)
        if base % accum == 0 and base // accum <= max_batch_size
    )
    return BatchPlan(
        per_device_batch=base // accum,
        gradient_accumulation_steps=accum,
        world_size=world_size,
    )


def load_slurm_config(config_path: Path) -> SlurmConfig:
    """Read the ``slurm:`` block out of an oplm training config.

    Raises:
        ValueError: If the config has no ``slurm:`` block.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config_path)
    block = OmegaConf.select(raw, "slurm")
    if block is None:
        raise ValueError(f"{config_path} has no `slurm:` block")
    container = OmegaConf.to_container(block, resolve=True)
    # OmegaConf.to_container's declared return type doesn't match any dict.__init__ overload.
    return SlurmConfig.from_mapping(dict(container))  # ty: ignore[no-matching-overload]

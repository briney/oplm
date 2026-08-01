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


def _positive_int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"slurm config is missing required field {key!r}")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"slurm config {key} must be a positive int, got {value!r}")
    return value


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
        max_batch = {
            str(preset): int(value) for preset, value in dict(raw.get("max_batch_size", {})).items()
        }
        bad = sorted(preset for preset, value in max_batch.items() if value < 1)
        if bad:
            raise ValueError(f"slurm config max_batch_size must be >= 1; bad presets: {bad}")
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
    return SlurmConfig.from_mapping(dict(container))  # ty: ignore[no-matching-overload]

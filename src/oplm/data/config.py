"""Data-config parsing helpers.

Parsing for the train/eval dataset specifications declared in
:mod:`oplm.config` (``TrainDatasetEntry`` / ``EvalDatasetEntry`` are imported
from there, not redefined here). Normalizes train-dataset fractions and folds
unknown eval keys into ``extra``.

This module is the single home for the parsing helpers: they used to live in
:mod:`oplm.config` and have been moved here so the loose-YAML → typed-entry
translation sits with the rest of the data tooling.
"""

from __future__ import annotations

from typing import Any

from oplm.config import EvalDatasetEntry, ScheduleSpec, TrainDatasetEntry, parse_schedule_block

# Eval-entry keys with dedicated handling; everything else is folded into
# ``EvalDatasetEntry.extra`` so task-specific config travels with the entry.
_KNOWN_EVAL_KEYS = frozenset({"path", "type", "every", "metrics"})


def parse_train_configs(raw: Any) -> list[TrainDatasetEntry]:
    """Normalize a raw ``data.train`` config value into structured dataset entries.

    Accepts:

    * ``None`` / empty string / empty mapping → empty list (no training data).
    * A single path string → one entry (``name="train"``, ``fraction=1.0``).
    * A mapping ``{name: {path: str, fraction?: float}}`` → one entry per key. A
      bare-string value (``{name: path}``) is accepted as shorthand for
      ``{path: <string>}`` with an omitted fraction.

    Fractions are normalized to sum to ``1.0``. Entries that omit ``fraction``
    split the remaining mass (``1 - sum(specified)``) equally among themselves.

    Args:
        raw: The ``data.train`` value from config — a string, mapping, or ``None``.

    Returns:
        List of :class:`~oplm.config.TrainDatasetEntry` with normalized fractions.

    Raises:
        ValueError: If a specified fraction is negative, the resolved fractions do
            not sum to a positive value, an entry is missing ``path``, or the raw
            value is neither a string nor a mapping.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        path = raw.strip()
        if not path:
            return []
        return [TrainDatasetEntry(name="train", path=path, fraction=1.0)]

    if isinstance(raw, dict):
        return _parse_train_mapping(raw)

    raise ValueError(
        f"data.train must be a path string or a {{name: {{path, fraction}}}} "
        f"mapping, got {type(raw).__name__}"
    )


def _parse_train_mapping(raw: dict[Any, Any]) -> list[TrainDatasetEntry]:
    """Parse the ``{name: {path, fraction?}}`` form into normalized entries."""
    if not raw:
        return []

    names: list[str] = []
    paths: list[str] = []
    # None marks an omitted fraction (resolved by splitting the remainder).
    fractions: list[float | None] = []
    for name, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str):
            path: Any = value
            frac: Any = None
        elif isinstance(value, dict):
            path = value.get("path")
            frac = value.get("fraction")
        else:
            raise ValueError(
                f"data.train.{name}: expected a path string or a "
                f"{{path, fraction}} mapping, got {type(value).__name__}"
            )

        if path is None or (isinstance(path, str) and not path.strip()):
            raise ValueError(f"data.train.{name} is missing required 'path'")

        names.append(str(name))
        paths.append(str(path).strip())
        fractions.append(float(frac) if frac is not None else None)

    if not names:
        return []

    for name, frac in zip(names, fractions, strict=True):
        if frac is not None and frac < 0:
            raise ValueError(f"data.train.{name}.fraction must be >= 0, got {frac}")

    specified_sum = sum(f for f in fractions if f is not None)
    n_omitted = sum(1 for f in fractions if f is None)
    if n_omitted:
        # Omitted entries share whatever mass the specified ones leave behind.
        share = max(0.0, 1.0 - specified_sum) / n_omitted
        resolved = [share if f is None else f for f in fractions]
    else:
        resolved = [f for f in fractions if f is not None]

    total = sum(resolved)
    if total <= 0:
        raise ValueError(f"data.train fractions must sum to > 0, got {total}")

    return [
        TrainDatasetEntry(name=name, path=path, fraction=frac / total)
        for name, path, frac in zip(names, paths, resolved, strict=True)
    ]


def parse_eval_configs(raw: Any, default_schedule: ScheduleSpec) -> list[EvalDatasetEntry]:
    """Normalize a raw ``data.eval`` config value into structured eval entries.

    Accepts ``None`` / empty mapping (→ empty list) or a mapping
    ``{name: {path, type, every?, metrics?, **extra}}``. ``path`` and ``type``
    are required; ``every`` is an ``{steps: N}`` / ``{tokens: N}`` cadence block
    (see :func:`oplm.config.parse_schedule_block`) that falls back to
    ``default_schedule`` when omitted; any keys outside the known set are folded
    into :attr:`~oplm.config.EvalDatasetEntry.extra` so task-specific config
    travels with the entry.

    Args:
        raw: The ``data.eval`` value from config — a mapping or ``None``.
        default_schedule: Fallback cadence (``train.eval_every``) applied
            when an entry omits ``every``.

    Returns:
        List of :class:`~oplm.config.EvalDatasetEntry`.

    Raises:
        ValueError: If ``raw`` is not a mapping, an entry is not a mapping, an
            entry is missing ``path`` or ``type``, an entry uses a nested
            ``extra`` block, an entry uses the removed ``eval_every`` key, an
            entry's ``every`` block is malformed, or ``metrics`` is not a list.
    """
    if raw is None:
        return []

    if not isinstance(raw, dict):
        raise ValueError(
            f"data.eval must be a {{name: {{path, type, ...}}}} mapping, got {type(raw).__name__}"
        )

    entries: list[EvalDatasetEntry] = []
    for name, value in raw.items():
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Eval dataset {name!r} must be a mapping with 'path' and 'type', "
                f"got {type(value).__name__}"
            )

        path = value.get("path")
        if path is None:
            raise ValueError(f"Eval dataset {name!r} is missing required 'path'")

        eval_type = value.get("type")
        if eval_type is None:
            raise ValueError(f"Eval dataset {name!r} is missing required 'type'")

        if "extra" in value:
            raise ValueError(
                f"Eval dataset {name!r} uses a nested 'extra' block; put "
                "task-specific keys directly on the dataset entry instead."
            )

        if "eval_every" in value:
            raise ValueError(
                f"Eval dataset {name!r} uses the removed `eval_every` key. "
                f"Use `every: {{steps: N}}` (or {{tokens: N}})."
            )

        raw_every = value.get("every")
        schedule = (
            parse_schedule_block(raw_every, f"data.eval.{name}.every")
            if raw_every is not None
            else default_schedule
        )

        raw_metrics = value.get("metrics")
        if raw_metrics is None:
            metrics = None
        elif isinstance(raw_metrics, (list, tuple)):
            metrics = [str(m) for m in raw_metrics]
        else:
            # A bare string is iterable, so the naive `[str(m) for m in "loss"]` yields
            # ["l", "o", "s", "s"]. Require an actual list/tuple of metric names; reject
            # strings (and anything else) explicitly. (OmegaConf is already resolved to
            # plain containers here — see the isinstance(value, dict) checks above.)
            raise ValueError(
                f"Eval dataset {name!r}: `metrics` must be a list of names "
                f"(e.g. [loss, accuracy]), got {raw_metrics!r}"
            )
        extra = {k: v for k, v in value.items() if k not in _KNOWN_EVAL_KEYS}

        entries.append(
            EvalDatasetEntry(
                name=str(name),
                path=str(path),
                type=str(eval_type),
                schedule=schedule,
                metrics=metrics,
                extra=extra,
            )
        )

    return entries

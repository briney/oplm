"""Config-parsing tests for the eval cadence grammar — pure, no model."""

from __future__ import annotations

from typing import Any

import pytest

from oplm.config import ScheduleSpec, load_config, parse_schedule_block
from oplm.data.config import parse_eval_configs

# Sentinel default cadence applied when an entry omits ``every``.
_DEFAULT = ScheduleSpec("steps", 12_345)


def _parse_one(entry: dict[str, Any]) -> Any:
    """Parse a single-entry ``data.eval`` mapping and return the lone entry."""
    entries = parse_eval_configs({"ds": entry}, _DEFAULT)
    assert len(entries) == 1
    return entries[0]


# --- parse_schedule_block ------------------------------------------------------


def test_schedule_block_steps() -> None:
    """A bare ``steps`` block parses with default flags (at_start off, at_end on)."""
    assert parse_schedule_block({"steps": 2000}, "x") == ScheduleSpec(
        "steps", 2000, at_start=False, at_end=True
    )


def test_schedule_block_tokens_with_flags() -> None:
    """Explicit ``at_start`` / ``at_end`` bools are honored."""
    spec = parse_schedule_block({"tokens": 1_000_000, "at_start": True, "at_end": False}, "x")
    assert spec == ScheduleSpec("tokens", 1_000_000, at_start=True, at_end=False)


def test_schedule_block_two_units_raises() -> None:
    """Naming two units is ambiguous and rejected."""
    with pytest.raises(ValueError, match="exactly one of"):
        parse_schedule_block({"steps": 1, "tokens": 2}, "x")


def test_schedule_block_epochs_deferred() -> None:
    """Epoch cadence is explicitly deferred with a clear message."""
    with pytest.raises(ValueError, match="not yet supported"):
        parse_schedule_block({"epochs": 1}, "x")


@pytest.mark.parametrize("bad_n", [0, -5, True])
def test_schedule_block_non_positive_or_bool_n_raises(bad_n: Any) -> None:
    """The unit value must be a positive, non-bool int."""
    with pytest.raises(ValueError, match="positive int"):
        parse_schedule_block({"steps": bad_n}, "x")


@pytest.mark.parametrize("flag", ["at_start", "at_end"])
def test_schedule_block_string_bool_rejected(flag: str) -> None:
    """A YAML string like ``"false"`` must not coerce to ``True`` (issue #5)."""
    with pytest.raises(ValueError, match=flag):
        parse_schedule_block({"steps": 10, flag: "false"}, "x")


# --- parse_eval_configs --------------------------------------------------------


def test_entry_every_steps() -> None:
    """A per-entry ``every`` block becomes the entry's schedule."""
    entry = _parse_one({"path": "p", "type": "sequence", "every": {"steps": 2000}})
    assert entry.schedule == ScheduleSpec("steps", 2000, at_start=False, at_end=True)


def test_entry_every_tokens_with_flags() -> None:
    """Token cadence with flags round-trips through ``parse_eval_configs``."""
    entry = _parse_one(
        {
            "path": "p",
            "type": "sequence",
            "every": {"tokens": 1_000_000, "at_start": True, "at_end": False},
        }
    )
    assert entry.schedule == ScheduleSpec("tokens", 1_000_000, at_start=True, at_end=False)


def test_entry_omitted_every_uses_default() -> None:
    """An entry without ``every`` inherits the supplied default schedule."""
    entry = _parse_one({"path": "p", "type": "sequence"})
    assert entry.schedule is _DEFAULT


def test_entry_metrics_bare_string_rejected() -> None:
    """``metrics`` must be a list — a bare string must not split into characters (issue #8)."""
    with pytest.raises(ValueError, match="must be a list"):
        _parse_one({"path": "p", "type": "sequence", "metrics": "loss"})


def test_entry_metrics_list_parses() -> None:
    """A real list of metric names is preserved verbatim."""
    entry = _parse_one({"path": "p", "type": "sequence", "metrics": ["loss", "accuracy"]})
    assert entry.metrics == ["loss", "accuracy"]


def test_entry_removed_eval_every_rejected() -> None:
    """The removed per-entry ``eval_every`` key points users at ``every``."""
    with pytest.raises(ValueError, match="every"):
        _parse_one({"path": "p", "type": "sequence", "eval_every": 500})


def test_entry_unknown_key_folds_into_extra() -> None:
    """Task-specific keys outside the known set travel in ``extra``."""
    entry = _parse_one({"path": "p", "type": "structure", "contact_threshold": 8.0})
    assert entry.extra == {"contact_threshold": 8.0}


def test_entry_every_malformed_propagates() -> None:
    """A malformed ``every`` block surfaces the schedule-parser error."""
    with pytest.raises(ValueError, match="exactly one of"):
        _parse_one({"path": "p", "type": "sequence", "every": {"steps": 1, "tokens": 2}})


# --- load_config ---------------------------------------------------------------


def test_load_config_rejects_removed_train_eval_every() -> None:
    """The removed global ``train.eval_every`` override points at ``eval_default_every``."""
    with pytest.raises(ValueError, match="eval_default_every"):
        load_config(["train.eval_every=500"])


def test_load_config_accepts_eval_default_every() -> None:
    """The new ``train.eval_default_every`` override resolves and parses as a cadence."""
    cfg = load_config(["train.eval_default_every={steps: 7}"])
    assert parse_schedule_block(cfg.train.eval_default_every, "train.eval_default_every") == (
        ScheduleSpec("steps", 7)
    )

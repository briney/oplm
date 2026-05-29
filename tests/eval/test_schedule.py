"""Pure scheduling tests — no model, no GPU. See docs/EVAL_HARNESS.md §4.5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from oplm.config import ScheduleSpec
from oplm.eval.schedule import EveryNSteps, EveryNTokens, build_schedule

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.eval import EvalContext


def test_every_n_steps_fires_on_multiple(ctx: Callable[..., EvalContext]) -> None:
    """A step schedule is due exactly at multiples of ``n`` (delta 1)."""
    sched = EveryNSteps(1000)
    assert sched.is_due(ctx(1000, steps_delta=1))
    assert not sched.is_due(ctx(999, steps_delta=1))
    assert not sched.is_due(ctx(1001, steps_delta=1))


def test_every_n_steps_at_start(ctx: Callable[..., EvalContext]) -> None:
    """``at_start`` fires on the first optimizer step; the default does not."""
    assert EveryNSteps(1000, at_start=True).is_due(ctx(1, steps_delta=1))
    assert not EveryNSteps(1000).is_due(ctx(1, steps_delta=1))


def test_every_n_steps_at_end(ctx: Callable[..., EvalContext]) -> None:
    """``at_end`` (default true) fires on the final step even when off-cadence."""
    assert EveryNSteps(1000).is_due(ctx(1500, is_final=True))


def test_every_n_tokens_crosses_multiple(ctx: Callable[..., EvalContext]) -> None:
    """A token schedule fires the step its cumulative count crosses a multiple."""
    sched = EveryNTokens(1_000_000)
    assert sched.is_due(ctx(step=10, tokens_seen=1_000_030, tokens_delta=80))
    # Next step stays within the same multiple → not due again.
    assert not sched.is_due(ctx(step=11, tokens_seen=1_000_110, tokens_delta=80))


def test_every_n_tokens_fires_once_when_delta_exceeds_n(
    ctx: Callable[..., EvalContext],
) -> None:
    """A single step whose delta spans several multiples fires exactly once."""
    sched = EveryNTokens(1_000_000)
    assert sched.is_due(ctx(step=1, tokens_seen=2_500_000, tokens_delta=2_500_000))


def test_step_schedule_does_not_refire_after_resume(
    ctx: Callable[..., EvalContext],
) -> None:
    """A schedule that fired at step 1000 is not due again at step 1001 (resume safety)."""
    sched = EveryNSteps(1000)
    assert sched.is_due(ctx(1000, steps_delta=1))
    assert not sched.is_due(ctx(1001, steps_delta=1))


def test_cadence_larger_than_run_never_fires_without_at_end(
    ctx: Callable[..., EvalContext],
) -> None:
    """``n`` beyond the run length never fires when both at_start/at_end are off."""
    sched = EveryNSteps(100_000, at_end=False)
    assert not any(sched.is_due(ctx(step, steps_delta=1)) for step in range(1, 50_001))


def test_cadence_larger_than_run_fires_only_at_end(
    ctx: Callable[..., EvalContext],
) -> None:
    """With ``at_end`` (default) the same oversized cadence fires only on the final step."""
    sched = EveryNSteps(100_000)
    assert not sched.is_due(ctx(50_000, steps_delta=1, is_final=False))
    assert sched.is_due(ctx(50_000, steps_delta=1, is_final=True))


def test_build_schedule_steps() -> None:
    """``build_schedule`` maps a ``steps`` spec to ``EveryNSteps`` preserving flags."""
    sched = build_schedule(ScheduleSpec("steps", 5))
    assert sched == EveryNSteps(5, at_start=False, at_end=True)


def test_build_schedule_tokens() -> None:
    """``build_schedule`` maps a ``tokens`` spec to ``EveryNTokens`` preserving flags."""
    sched = build_schedule(ScheduleSpec("tokens", 7, at_start=True, at_end=False))
    assert sched == EveryNTokens(7, at_start=True, at_end=False)


def test_build_schedule_unsupported_unit_raises() -> None:
    """An unsupported unit is rejected with an actionable message."""
    with pytest.raises(ValueError, match="Unsupported schedule unit"):
        build_schedule(ScheduleSpec("furlongs", 3))
